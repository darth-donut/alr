from alr.training.pl_mixup import PLMixupTrainer, temp_ds_transform
from alr.training import Trainer
from alr.training.samplers import RandomFixedLengthSampler
from alr.utils import manual_seed, timeop
from alr.data.datasets import Dataset
from alr.data import UnlabelledDataset, DataManager, TransformedDataset, disable_augmentation
from alr.training.utils import PLPredictionSaver
from alr import ALRModel
from alr.acquisition import AcquisitionFunction

import torch
import pickle
import torch.utils.data as torchdata
import torchvision as tv
from collections import defaultdict
from ignite.engine import create_supervised_evaluator
from pathlib import Path
from torch import nn
import numpy as np
from torch.nn import functional as F
from ignite.engine import Engine
import random

def save_rnd_state(prefix):
    with open(prefix + "_state.pkl", "wb") as fp:
        payload = {
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'rand': random.getstate()
        }
        pickle.dump(payload, fp)

def load_rnd_state(prefix):
    with open(prefix + "_state.pkl", "rb") as fp:
        payload = pickle.load(fp)
    np.random.set_state(payload['numpy'])
    torch.set_rng_state(payload['torch'])
    random.setstate(payload['rand'])

class RecordPseudoLabels:
    def __init__(self,
                 model: nn.Module,
                 pool_loader: torchdata.DataLoader,
                 patience: int,
                 max_epoch: int,
                 device):
        self.labels_E_N_C = []
        self.model = model
        self.pool_loader = pool_loader
        self.device = device
        self.patience = patience
        self.last_epoch = 0
        self.max_epoch = max_epoch

    def __call__(self, engine: Engine):
        self.model.eval()
        with torch.no_grad():
            labels = []
            for x in self.pool_loader:
                x = x.to(self.device)
                labels.append(self.model(x).exp_().cpu())
            self.labels_E_N_C.append(torch.cat(labels))
            self.last_epoch = engine.state.epoch

    @property
    def pseudo_labels(self):
        if self.last_epoch != self.max_epoch:
            print(f"Ignoring the last {self.patience} history")
            # early stopping happened
            return torch.stack(self.labels_E_N_C[:-self.patience])
        print("Using full history")
        return torch.stack(self.labels_E_N_C)


class LastKL(AcquisitionFunction):
    def __init__(self):
        super(LastKL, self).__init__()
        self.labels_E_N_C: torch.Tensor = None
        self.recent_score = None

    def __call__(self, X_pool: torchdata.Dataset, b: int):
        pool_size = len(X_pool)
        mc_preds = self.labels_E_N_C.double()
        E, N, C = mc_preds.shape
        # p = [N, C]
        p = mc_preds[-1]
        # reverse "mode-seeking" kl divergence
        scores = LastKL._kl_divergence(mc_preds[:-1], p.unsqueeze(0)).mean(0)
        assert torch.isfinite(scores).all()
        assert scores.shape == (pool_size,)
        result = torch.argsort(scores, descending=True).numpy()
        self.recent_score = scores.numpy()
        return result[:b]

    @staticmethod
    def _kl_divergence(p, q):
        return (p * torch.log(p / (q + 1e-5) + 1e-5)).sum(dim=-1)


def make_imbalanced_pool(dataset: torchdata.Dataset, classes, n):
    idxs = []
    count = {c: n for c in classes}
    for idx in np.random.permutation(len(dataset)):
        y = dataset[idx][1]
        if y in count:
            if count[y] > 0:
                idxs.append(idx)
                count[y] -= 1
        else:
            idxs.append(idx)
    return torchdata.Subset(dataset, idxs)


def uneven_split(dataset: torchdata.Dataset,
                 mapping: dict) -> tuple:
    count = {k: v for k, v in mapping.items()}
    original_idxs = set(range(len(dataset)))
    idxs = []
    for idx in np.random.permutation(len(dataset)):
        if all(i == 0 for i in count.values()):
            break
        y = dataset[idx][1]
        if count[y]:
            count[y] -= 1
            idxs.append(idx)
    return torchdata.Subset(
        dataset, idxs
    ), torchdata.Subset(
        dataset, list(original_idxs - set(idxs))
    )


class Net(ALRModel):
    def __init__(self, model):
        super(Net, self).__init__()
        self.model = model
        self.snap()

    def forward(self, x):
        return self.model(x)


def calc_calib_metrics(loader, model: nn.Module, log_dir: Path, device):
    if log_dir.exists():
        return
    evaluator = create_supervised_evaluator(
        model, metrics=None, device=device
    )
    pds = PLPredictionSaver(log_dir)
    pds.attach(evaluator)
    evaluator.run(loader)


def main(seed: int,
         alpha: float,
         b: int,
         iters: int,
         ):
    augment = True
    acq_name = "lastKL"
    print(f"Starting experiment with seed {seed}, augment = {augment} ...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    kwargs = dict(num_workers=4, pin_memory=True)

    # ========= CONSTANTS ===========
    BATCH_SIZE = 100
    # at least have this much points in one epoch (see RandomFixedLengthSampler)
    MIN_TRAIN_LENGTH = 20_000
    VAL_SIZE = 5_000
    # stage 1 and stage 2 patience
    PATIENCE = 10
    EPOCHS = 400

    ITERS = iters

    # ========= SETUP ===========
    manual_seed(42)
    mapping = {cls: 110 for cls in range(10)}
    mapping[0] = 10
    mapping[4] = 10
    mapping[9] = 10
    train, test = Dataset.CIFAR10.get(raw=True)
    train, pool = uneven_split(train, mapping)
    pool_idxs = pool.indices[:]
    class_count = defaultdict(int)
    for _, y in train:
        class_count[y] += 1
    print("Initial dataset class count:")
    print(class_count)

    pool, val = torchdata.random_split(pool, (len(pool) - VAL_SIZE, VAL_SIZE))
    pool_idxs = (pool_idxs, pool.indices)

    pool = make_imbalanced_pool(pool, [0, 4, 9], n=500)
    pool_idxs = (*pool_idxs, pool.indices)
    class_count = defaultdict(int)
    for _, y in pool:
        class_count[y] += 1
    print("Unlabelled class count:")
    print(class_count)

    manual_seed(seed)
    standard_augmentation = [
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.RandomHorizontalFlip(),
    ]
    regular_transform = [
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(*Dataset.CIFAR10.normalisation_params)
    ]
    accs = defaultdict(list)

    # apply augmentation to train and pool
    train = TransformedDataset(
        train, transform=regular_transform,
        augmentation=standard_augmentation
    )
    # disable pool augmentation when acquiring
    pool = UnlabelledDataset(
        TransformedDataset(
            pool, transform=regular_transform,
            augmentation=standard_augmentation
        )
    )
    val = TransformedDataset(val, transform=regular_transform)

    test_loader = torchdata.DataLoader(
        test, batch_size=512, shuffle=False, **kwargs,
    )
    val_loader = torchdata.DataLoader(
        val, batch_size=512, shuffle=False, **kwargs,
    )

    template = f"{acq_name}_{b}_alpha_{alpha}" + ("_aug" if augment else "") + f"_{seed}"
    metrics = Path("metrics") / template
    calib_metrics = Path("calib_metrics") / template
    saved_models = Path("saved_models") / template
    metrics.mkdir(parents=True, exist_ok=True)
    calib_metrics.mkdir(parents=True, exist_ok=True)
    saved_models.mkdir(parents=True, exist_ok=True)

    # since we need to know which points were taken for val dataset
    with open(metrics / "pool_idxs.pkl", "wb") as fp:
        pickle.dump(pool_idxs, fp)

    acq_fn = LastKL()
    # repeat no. 1
    r = 1

    model = Net(Dataset.CIFAR10.model).to(device)

    if Path(template + "_accs.pkl").exists():
        print("Resuming from last checkpoint ...")
        with open(template + "_accs.pkl", "rb") as fp:
            accs = pickle.load(fp)

    start_iter = 1
    files = sorted(list(metrics.glob(f"rep_{r}*.pkl")), key=lambda x: int(str(x).split("_")[-1][:-4]))
    if len(files):
        last_chkpt = files[-1]
        start_iter = int(last_chkpt.name.split("_")[-1][:-4]) + 1
        print(f"Resuming from iteration {start_iter}; ", end="")
        with open(last_chkpt, "rb") as fp:
            payload = pickle.load(fp)
            reload_idxs = payload['labelled_indices']
            # reload train and pool
            acq_ds = pool.label(reload_idxs)
            train = torchdata.ConcatDataset([train, acq_ds])
            print(f"train: {len(train)}, pool: {len(pool)}")
        load_rnd_state(template)

    dm = DataManager(train, pool, acq_fn)

    for i in range(start_iter, ITERS + 1):
        # finally, acquire points
        model.reset_weights()
        pool_loader = torchdata.DataLoader(
            dm.unlabelled,
            batch_size=512, shuffle=False,
            **kwargs,
        )
        pl_recorder = RecordPseudoLabels(
            model, pool_loader,
            patience=PATIENCE,
            max_epoch=EPOCHS,
            device=device,
        )
        trainer = Trainer(
            model, F.nll_loss, optimiser='Adam',
            patience=PATIENCE, reload_best=True, device=device,
        )
        train_loader = torchdata.DataLoader(
            dm.labelled, batch_size=BATCH_SIZE,
            sampler=RandomFixedLengthSampler(dm.labelled, MIN_TRAIN_LENGTH, shuffle=True),
            **kwargs
        )
        with timeop() as t:
            history = trainer.fit(
                train_loader,
                val_loader,
                epochs=EPOCHS,
                callbacks=[pl_recorder]
            )
        test_metrics = trainer.evaluate(test_loader)
        print(f"=== Iteration {i} of {ITERS} ({i / ITERS:.2%}) ===")
        print(f"\ttrain: {dm.n_labelled}; val: {len(val)}; "
              f"pool: {dm.n_unlabelled}; test: {len(test)}")
        print(f"\t[test] acc: {test_metrics['acc']:.4f}, time: {t}")
        accs[dm.n_labelled].append(test_metrics['acc'])

        # flush results frequently for the impatient
        with open(template + "_accs.pkl", "wb") as fp:
            pickle.dump(accs, fp)

        with dm.unlabelled.tmp_debug():
            pool_loader = torchdata.DataLoader(
                disable_augmentation(dm.unlabelled),
                batch_size=512, shuffle=False,
                **kwargs,
            )
            calc_calib_metrics(
                pool_loader, model, calib_metrics / "pool" / f"rep_{r}" / f"iter_{i}",
                device=device
            )
        calc_calib_metrics(
            test_loader, model, calib_metrics / "test" / f"rep_{r}" / f"iter_{i}",
            device=device
        )

        torch.save(model.state_dict(), saved_models / f"rep_{r}_iter_{i}.pt")


        acq_fn.labels_E_N_C = pl_recorder.pseudo_labels
        dm.acquire(b=b, transform=disable_augmentation)


        with open(metrics / f"rep_{r}_iter_{i}.pkl", "wb") as fp:
            payload = {
                'history': history,
                'test_metrics': test_metrics,
                'labelled_classes': dm.unlabelled.labelled_classes,
                'labelled_indices': dm.unlabelled.labelled_indices,
                'bald_scores': acq_fn.recent_score,
            }
            # critical section
            save_rnd_state(template)
            pickle.dump(payload, fp)
            # END critical section


if __name__ == '__main__':
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--alpha", type=float, default=0.4)
    args.add_argument("--iters", type=int)
    args = args.parse_args()

    main(
        seed=args.seed,
        alpha=args.alpha,
        b=20,
        iters=args.iters,
    )

