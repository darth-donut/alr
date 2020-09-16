from alr.utils import manual_seed, eval_fwd_exp, timeop
from alr.data.datasets import Dataset
from alr.data import DataManager, UnlabelledDataset
from alr.acquisition import BALD, RandomAcquisition, _bald_score
from alr.training import Trainer
from alr.training.samplers import RandomFixedLengthSampler
from alr.training.utils import PLPredictionSaver
from alr import MCDropout

import torch
import numpy as np
import pickle
import torch.utils.data as torchdata
from torch.nn import functional as F
from collections import defaultdict
from ignite.engine import create_supervised_evaluator
from pathlib import Path
from torch import nn


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


def calc_calib_metrics(loader, model: nn.Module, log_dir, device):
    evaluator = create_supervised_evaluator(
        model, metrics=None, device=device
    )
    pds = PLPredictionSaver(log_dir)
    pds.attach(evaluator)
    evaluator.run(loader)


def main(b, iters, repeats):
    acq_name = "random"
    manual_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    kwargs = dict(num_workers=4, pin_memory=True)

    # ========= CONSTANTS ===========
    BATCH_SIZE = 64
    # with early stopping, this'll probably be lesser
    EPOCHS = 200
    # at least have this much points in one epoch (see RandomFixedLengthSampler)
    MIN_TRAIN_LENGTH = 20_000
    VAL_SIZE = 5_000

    REPEATS = repeats
    ITERS = iters

    # ========= SETUP ===========
    mapping = {cls: 150 for cls in range(10)}
    mapping[1] = 15
    mapping[2] = 15
    mapping[8] = 15
    train, test = Dataset.CIFAR10.get()
    train, pool = uneven_split(train, mapping)

    class_count = defaultdict(int)
    for _, y in train:
        class_count[y] += 1
    print(class_count)

    pool, val = torchdata.random_split(pool, (len(pool) - VAL_SIZE, VAL_SIZE))
    pool = UnlabelledDataset(pool)
    val_loader = torchdata.DataLoader(
        val, batch_size=512, shuffle=False, **kwargs,
    )
    test_loader = torchdata.DataLoader(
        test, batch_size=512, shuffle=False, **kwargs,
    )
    accs = defaultdict(list)

    template = f"{acq_name}_{b}"
    metrics = Path("metrics") / template
    calib_metrics = Path("calib_metrics") / template
    saved_models = Path("saved_models") / template
    metrics.mkdir(parents=True)
    calib_metrics.mkdir(parents=True)
    saved_models.mkdir(parents=True)
    acquired_idxs = None

    # since we need to know which points were taken for val dataset
    with open(metrics / "pool_idxs.pkl", "wb") as fp:
        pickle.dump(pool._dataset.indices, fp)
    # need to know which were the initial samples too
    with open(metrics / "train_idxs.pkl", "wb") as fp:
        pickle.dump(train.indices, fp)

    for r in range(1, REPEATS + 1):
        print(f"- [{acq_name} (b={b})] repeat #{r} of {REPEATS}-")
        model = MCDropout(Dataset.CIFAR10.model, forward=20, fast=False).to(device)
        acq_fn = RandomAcquisition()

        dm = DataManager(train, pool, acq_fn)
        dm.reset()  # this resets pool

        for i in range(1, ITERS + 1):
            model.reset_weights()
            trainer = Trainer(
                model, F.nll_loss, optimiser='Adam',
                patience=10, reload_best=True, device=device
            )
            train_loader = torchdata.DataLoader(
                dm.labelled, batch_size=BATCH_SIZE,
                sampler=RandomFixedLengthSampler(dm.labelled, MIN_TRAIN_LENGTH, shuffle=True),
                **kwargs
            )
            with timeop() as t:
                history = trainer.fit(train_loader, val_loader, epochs=EPOCHS)

            # eval
            test_metrics = trainer.evaluate(test_loader)
            print(f"=== Iteration {i} of {ITERS} ({i / ITERS:.2%}) ===")
            print(f"\ttrain: {dm.n_labelled}; val: {len(val)}; "
                  f"pool: {dm.n_unlabelled}; test: {len(test)}")
            print(f"\t[test] acc: {test_metrics['acc']:.4f}, time: {t}")
            accs[dm.n_labelled].append(test_metrics['acc'])

            # save stuff

            # pool calib
            with dm.unlabelled.tmp_debug():
                pool_loader = torchdata.DataLoader(
                    dm.unlabelled, batch_size=512, shuffle=False,
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
                    'labelled_classes': dm.unlabelled.labelled_classes,
                    'labelled_indices': dm.unlabelled.labelled_indices,
                    'acquired_idxs': acquired_idxs,
                }
                pickle.dump(payload, fp)
            torch.save(model.state_dict(), saved_models / f"rep_{r}_iter_{i}.pt")

            # flush results frequently for the impatient
            with open(template + "_accs.pkl", "wb") as fp:
                pickle.dump(accs, fp)

            if i == ITERS: continue  # job done
            # finally, acquire points
            acquired_idxs, _ = dm.acquire(b=b)


if __name__ == '__main__':
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--b", default=10, type=int, help="Batch acq size (default = 10)")
    args.add_argument("--iters", default=199, type=int)
    args.add_argument("--reps", default=1, type=int)
    args = args.parse_args()

    main(b=args.b, iters=args.iters, repeats=args.reps)

