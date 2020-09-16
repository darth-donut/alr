# ENSEMBLE AL on CIFAR10, IMBALANCED 0, 4, 9 minority
from alr.utils import manual_seed, eval_fwd_exp, timeop
from alr.data.datasets import Dataset
from alr.data import DataManager, UnlabelledDataset, TransformedDataset, disable_augmentation
from alr.acquisition import BALD, RandomAcquisition, _bald_score
from alr.training import Trainer
from alr.training.samplers import RandomFixedLengthSampler
from alr.training.utils import PLPredictionSaver

import torch
import pickle
import torch.utils.data as torchdata
from torch.nn import functional as F
from collections import defaultdict
from ignite.engine import create_supervised_evaluator
from pathlib import Path
import torchvision as tv
import numpy as np


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



class Ensemble:
    def __init__(self, models: list):
        # assumes models return log-softmax probabilities
        self.models = models

    def forward(self, x):
        return self.get_preds(x).mean(dim=0)

    def get_preds(self, x):
        preds = []
        with torch.no_grad():
            for m in self.models:
                m.eval()
                preds.append(m(x).exp())
        return torch.stack(preds)

    def evaluate(self, loader, device):
        with torch.no_grad():
            correct = 0
            total = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                correct += (self.forward(x).argmax(dim=1) == y).sum().item()
                total += y.size(0)
        return correct / total

    def __call__(self, x):
        return self.forward(x)

    def eval(self):
        self.train(mode=False)

    def train(self, mode=True):
        # should never ever be in training mode
        assert not mode

    def save_weights(self, prefix: str):
        for mi, m in enumerate(self.models, 1):
            torch.save(m.state_dict(), f"{prefix}_model_{mi}.pt")


def calc_calib_metrics(loader, model, log_dir, device):
    evaluator = create_supervised_evaluator(
        model, metrics=None, device=device
    )
    pds = PLPredictionSaver(log_dir)
    pds.attach(evaluator)
    evaluator.run(loader)


def main(acq_name, b, iters, repeats, seed, save_weights: bool):
    print(f"Seed: {seed}" + (", saving weights." if save_weights else ""))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    kwargs = dict(num_workers=4, pin_memory=True)

    # ========= CONSTANTS ===========
    BATCH_SIZE = 64
    # with early stopping, this'll probably be lesser
    EPOCHS = 400
    # at least have this much points in one epoch (see RandomFixedLengthSampler)
    MIN_TRAIN_LENGTH = 20_000
    VAL_SIZE = 5_000

    REPEATS = repeats
    ITERS = iters

    # ========= SETUP ===========
    standard_augmentation = [
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.RandomHorizontalFlip(),
    ]
    regular_transform = [
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(*Dataset.CIFAR10.normalisation_params)
    ]

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
    # apply augmentation to train and pool only
    train = TransformedDataset(
        train, transform=regular_transform,
        augmentation=standard_augmentation
    )
    # when acquiring(scoring) points, we'll temporarily disable augmentation
    pool = UnlabelledDataset(
        TransformedDataset(
            pool, transform=regular_transform,
            augmentation=standard_augmentation
        )
    )
    # no augmentation in validation set
    val = TransformedDataset(val, transform=regular_transform)

    val_loader = torchdata.DataLoader(
        val, batch_size=512, shuffle=False, **kwargs,
    )
    test_loader = torchdata.DataLoader(
        test, batch_size=512, shuffle=False, **kwargs,
    )
    accs = defaultdict(list)


    # ========== MISC =============
    template = f"{acq_name}_{b}_{seed}"
    metrics = Path("metrics") / template
    calib_metrics = Path("calib_metrics") / template
    saved_models = Path("saved_models") / template
    metrics.mkdir(parents=True)
    calib_metrics.mkdir(parents=True)
    saved_models.mkdir(parents=True)
    bald_scores = None

    dm = DataManager(train, pool, None)

    # since we need to know which points were taken for val dataset
    with open(metrics / "pool_idxs.pkl", "wb") as fp:
        pickle.dump(pool_idxs, fp)

    for r in range(1, REPEATS + 1):
        print(f"- [{acq_name} (b={b})] repeat #{r} of {REPEATS}-")
        dm.reset()  # this resets pool
        for i in range(1, ITERS + 1):
            models = []
            histories = []
            with timeop() as t:
                for m in range(5):
                    print(f"\tTraining model {m + 1} of 5")
                    model = Dataset.CIFAR10.model.to(device)
                    trainer = Trainer(
                        model, F.nll_loss, optimiser='Adam',
                        patience=10, reload_best=True, device=device,
                    )
                    train_loader = torchdata.DataLoader(
                        dm.labelled, batch_size=BATCH_SIZE,
                        sampler=RandomFixedLengthSampler(dm.labelled, MIN_TRAIN_LENGTH, shuffle=True),
                        **kwargs
                    )
                    histories.append(trainer.fit(train_loader, val_loader, epochs=EPOCHS))
                    models.append(model)
            # eval
            ensemble = Ensemble(models)
            test_acc = ensemble.evaluate(test_loader, device)
            print(f"=== Iteration {i} of {ITERS} ({i / ITERS:.2%}) ===")
            print(f"\ttrain: {dm.n_labelled}; val: {len(val)}; "
                  f"pool: {dm.n_unlabelled}; test: {len(test)}")
            print(f"\t[test] acc: {test_acc}, time: {t}")
            accs[dm.n_labelled].append(test_acc)

            # save stuff
            # pool calib
            with dm.unlabelled.tmp_debug():
                pool_loader = torchdata.DataLoader(
                    disable_augmentation(dm.unlabelled),
                    batch_size=512, shuffle=False,
                    **kwargs,
                )
                calc_calib_metrics(
                    pool_loader, ensemble, calib_metrics / "pool" / f"rep_{r}" / f"iter_{i}",
                    device=device
                )
            calc_calib_metrics(
                test_loader, ensemble, calib_metrics / "test" / f"rep_{r}" / f"iter_{i}",
                device=device
            )

            with open(metrics / f"rep_{r}_iter_{i}.pkl", "wb") as fp:
                payload = {
                    'history': histories,
                    'test_metrics': test_acc,
                    'labelled_classes': dm.unlabelled.labelled_classes,
                    'labelled_indices': dm.unlabelled.labelled_indices,
                    'bald_scores': bald_scores,

                }
                pickle.dump(payload, fp)

            if save_weights:
                ensemble.save_weights(prefix=str(saved_models / f"rep_{r}_iter_{i}"))

            # flush results frequently for the impatient
            with open(template + "_accs.pkl", "wb") as fp:
                pickle.dump(accs, fp)

            ## ACQUIRE ##
            if acq_name == "bald":
                acq_fn = BALD(ensemble.get_preds, device=device, batch_size=512, **kwargs)
            elif acq_name == "random":
                acq_fn = RandomAcquisition()
            else:
                raise Exception("Done goofed.")
            dm._a_fn = acq_fn

            # finally, acquire points
            acquired_idxs, _ = dm.acquire(b=b, transform=disable_augmentation)
            # if bald, store ALL bald scores and the acquired idx so we can map the top b scores
            # to the b acquired_idxs
            if acq_name == "bald":
                # acquired_idxs has the top b scores from recent_score
                bald_scores = (acquired_idxs, acq_fn.recent_score)


if __name__ == '__main__':
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--acq", choices=['bald', 'random'], default='bald')
    args.add_argument("--b", default=10, type=int, help="Batch acq size (default = 10)")
    args.add_argument("--iters", default=199, type=int)
    args.add_argument("--reps", default=1, type=int)
    args.add_argument("--seed", type=int)
    args.add_argument("--save", action='store_true')
    args = args.parse_args()

    main(args.acq, b=args.b, iters=args.iters, repeats=args.reps, seed=args.seed, save_weights=args.save)
