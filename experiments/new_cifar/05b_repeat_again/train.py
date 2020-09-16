# post-acquisition BALD-10 with imbalanced pool and initial set
from alr.training.pl_mixup import PLMixupTrainer, temp_ds_transform
from alr.utils import manual_seed, timeop
from alr.data.datasets import Dataset
from alr.data import UnlabelledDataset
from alr.training.utils import PLPredictionSaver
from alr import ALRModel

import torch
import pickle
import torch.utils.data as torchdata
import torchvision as tv
from collections import defaultdict
from ignite.engine import create_supervised_evaluator
from pathlib import Path
from torch import nn
import numpy as np
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


def calc_calib_metrics(loader, model: nn.Module, log_dir, device):
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
         augment: bool,
         metrics_path: str,
         every: int):
    acq_name = "lastKL"
    print(f"Starting experiment with seed {seed}, augment = {augment} ...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    kwargs = dict(num_workers=4, pin_memory=True)

    # ========= CONSTANTS ===========
    BATCH_SIZE = 100
    # at least have this much points in one epoch (see RandomFixedLengthSampler)
    MIN_TRAIN_LENGTH = 20_000
    VAL_SIZE = 5_000
    MIN_LABELLED = 16
    # stage 1 and stage 2 patience
    PATIENCE = (5, 25)
    # how many epochs before LR is reduced
    LR_PATIENCE = 10
    # stage 1 and stage 2 of PL mixup training
    EPOCHS = (100, 400)

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

    original_train = train

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
    pool = UnlabelledDataset(pool)
    test_loader = torchdata.DataLoader(
        test, batch_size=512, shuffle=False, **kwargs,
    )
    train_transform = test_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(
            *Dataset.CIFAR10.normalisation_params
        ),
    ])
    if augment:
        data_augmentation = Dataset.CIFAR10.get_augmentation
    else:
        data_augmentation = None
    accs = defaultdict(list)

    root = Path(metrics_path) / f"lastKL_20_alpha_0.4_aug_{seed}"
    files = sorted(list(root.glob("rep_1*.pkl")), key=lambda x: int(str(x).split("_")[-1][:-4]))
    indices = []
    for f in files:
        with open(f, "rb") as fp:
            indices.append(pickle.load(fp)['labelled_indices'])
    indices = indices[::every] + [indices[-1]]
    # let the for loop iterate + 1 more to evaluate the final 'indices'
    indices.append([])

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

    # repeat no. 1
    r = 1

    model = Net(Dataset.CIFAR10.model).to(device)

    start_idx = 1
    full_idxs = indices

    if Path(template + "_chkpt.pkl").exists():
        with open(template + "_chkpt.pkl", "rb") as fp:
            start_idx = pickle.load(fp) + 1
        print(f"Resuming from iteration {start_idx}")
        indices = indices[(start_idx-1):]
        load_rnd_state(template)

    if Path(template + "_accs.pkl").exists():
        print("Resuming from checkpoint ...")
        with open(template + "_accs.pkl", "rb") as fp:
            accs = pickle.load(fp)


    for i, idx in enumerate(indices, start_idx):
        model.reset_weights()
        trainer = PLMixupTrainer(
            model, 'SGD', train_transform, test_transform,
            {'lr': .1, 'momentum': .9, 'weight_decay': 1e-4},
            kwargs, log_dir=None,
            rfls_length=MIN_TRAIN_LENGTH, alpha=alpha,
            min_labelled=MIN_LABELLED,
            data_augmentation=data_augmentation,
            batch_size=BATCH_SIZE,
            patience=PATIENCE, lr_patience=LR_PATIENCE,
            device=device
        )
        with pool.tmp_debug():
            with timeop() as t:
                history = trainer.fit(
                    train, val, pool, epochs=EPOCHS
                )

        # eval
        test_metrics = trainer.evaluate(test_loader)
        print(f"=== Iteration {i} of {len(full_idxs)} ({i / len(full_idxs):.2%}) ===")
        print(f"\ttrain: {len(train)}; val: {len(val)}; "
              f"pool: {len(pool)}; test: {len(test)}")
        print(f"\t[test] acc: {test_metrics['acc']:.4f}, time: {t}")
        accs[len(train)].append(test_metrics['acc'])

        # save stuff

        # pool calib
        with pool.tmp_debug():
            pool_loader = torchdata.DataLoader(
                temp_ds_transform(test_transform, with_targets=True)(pool),
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

        with open(metrics / f"rep_{r}_iter_{i}.pkl", "wb") as fp:
            payload = {
                'history': history,
                'test_metrics': test_metrics,
                'labelled_classes': pool.labelled_classes,
                'labelled_indices': pool.labelled_indices,
            }
            pickle.dump(payload, fp)
        # torch.save(model.state_dict(), saved_models / f"rep_{r}_iter_{i}.pt")

        # finally, acquire points
        # since indices[i] is cumulative, reset the pool
        pool.reset()
        # now get the points that BALD acquired
        acq_ds = pool.label(idx)
        # and append it to train
        train = torchdata.ConcatDataset([original_train, acq_ds])

        # flush results frequently for the impatient
        with open(template + "_accs.pkl", "wb") as fp:
            pickle.dump(accs, fp)

        with open(template + "_chkpt.pkl", "wb") as fp:
            # critical section
            save_rnd_state(template)
            pickle.dump(i, fp)
            # END: critical section



if __name__ == '__main__':
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--alpha", type=float, default=0.4)
    args.add_argument("--augment", action='store_true')
    args = args.parse_args()

    main(
        seed=args.seed,
        alpha=args.alpha,
        b=20,
        augment=args.augment,
        metrics_path="kl_metrics",
        every=2,
    )

