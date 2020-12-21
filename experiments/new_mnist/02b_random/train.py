# MNIST experiment 2 (BALD, RANDOM)
from alr.utils import manual_seed, eval_fwd_exp, timeop
from alr.data.datasets import Dataset
from alr.data import DataManager, UnlabelledDataset
from alr.acquisition import BALD, RandomAcquisition
from alr.training import Trainer
from alr.training.samplers import RandomFixedLengthSampler
from alr.training.utils import PLPredictionSaver
from alr import MCDropout

import torch
import pickle
import torch.utils.data as torchdata
from torch.nn import functional as F
from collections import defaultdict
from ignite.engine import create_supervised_evaluator
from pathlib import Path
import numpy as np


def calc_calib_metric(loader, model, device, log_dir):
    save_pl_metrics = create_supervised_evaluator(model, metrics=None, device=device)
    pps = PLPredictionSaver(
        log_dir=log_dir,
    )
    pps.attach(save_pl_metrics)
    save_pl_metrics.run(loader)


def uneven_split(dataset: torchdata.Dataset, mapping: dict) -> tuple:
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
    return torchdata.Subset(dataset, idxs), torchdata.Subset(
        dataset, list(original_idxs - set(idxs))
    )


def main(acq_name, b, iters, reps, seed):
    print(f"Starting experiment with seed {seed}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kwargs = dict(num_workers=4, pin_memory=True)

    # ========= CONSTANTS ===========
    BATCH_SIZE = 64
    # with early stopping, this'll probably be lesser
    EPOCHS = 50
    # at least have this much points in one epoch (see RandomFixedLengthSampler)
    MIN_TRAIN_LENGTH = 12_500
    VAL_SIZE = 5_000

    REPEATS = reps
    ITERS = iters

    # ========= SETUP ===========
    manual_seed(42)
    mapping = {cls: 11 for cls in range(10)}
    mapping[2] = 1
    mapping[4] = 1
    mapping[7] = 1
    train, test = Dataset.MNIST.get()
    train, pool = uneven_split(train, mapping)
    pool_idxs = pool.indices
    class_count = defaultdict(int)
    for _, y in train:
        class_count[y] += 1
    print(class_count)

    # use seed 42 to do uneven_split consistently across all experiments
    manual_seed(seed)

    pool, val = torchdata.random_split(pool, (len(pool) - VAL_SIZE, VAL_SIZE))
    pool_idxs = (pool_idxs, pool.indices)
    pool = UnlabelledDataset(pool)
    val_loader = torchdata.DataLoader(
        val,
        batch_size=1024,
        shuffle=False,
        **kwargs,
    )
    test_loader = torchdata.DataLoader(
        test,
        batch_size=1024,
        shuffle=False,
        **kwargs,
    )
    accs = defaultdict(list)

    template = f"{acq_name}_{b}_{seed}"
    metrics = Path("metrics") / template
    calib_metrics = Path("calib_metrics") / template
    saved_models = Path("saved_models") / template
    metrics.mkdir(parents=True)
    calib_metrics.mkdir(parents=True)
    saved_models.mkdir(parents=True)
    bald_scores = None

    with open(metrics / "pool_idxs.pkl", "wb") as fp:
        pickle.dump(pool_idxs, fp)

    for r in range(1, REPEATS + 1):
        print(f"- [{acq_name} (b={b})] repeat #{r} of {REPEATS}-")
        model = MCDropout(Dataset.MNIST.model, forward=20, fast=True).to(device)
        if acq_name == "bald":
            acq_fn = BALD(eval_fwd_exp(model), device=device, batch_size=1024, **kwargs)
        elif acq_name == "random":
            acq_fn = RandomAcquisition()
        else:
            raise Exception("Done goofed.")
        dm = DataManager(train, pool, acq_fn)
        dm.reset()  # this resets pool
        for i in range(1, ITERS + 1):
            model.reset_weights()
            trainer = Trainer(
                model,
                F.nll_loss,
                optimiser="Adam",
                patience=3,
                reload_best=True,
                device=device,
            )
            train_loader = torchdata.DataLoader(
                dm.labelled,
                batch_size=BATCH_SIZE,
                sampler=RandomFixedLengthSampler(
                    dm.labelled, MIN_TRAIN_LENGTH, shuffle=True
                ),
                **kwargs,
            )
            with timeop() as t:
                history = trainer.fit(train_loader, val_loader, epochs=EPOCHS)

            # eval
            test_metrics = trainer.evaluate(test_loader)
            print(f"=== Iteration {i} of {ITERS} ({i/ITERS:.2%}) ===")
            print(
                f"\ttrain: {dm.n_labelled}; val: {len(val)}; "
                f"pool: {dm.n_unlabelled}; test: {len(test)}"
            )
            print(f"\t[test] acc: {test_metrics['acc']:.4f}, time: {t}")
            accs[dm.n_labelled].append(test_metrics["acc"])

            # save stuff
            with dm.unlabelled.tmp_debug():
                pool_loader = torchdata.DataLoader(
                    dm.unlabelled,
                    batch_size=1024,
                    shuffle=False,
                    **kwargs,
                )
                calc_calib_metric(
                    pool_loader,
                    model,
                    device,
                    (calib_metrics / "pool" / f"rep_{r}" / f"iter_{i}"),
                )
            calc_calib_metric(
                test_loader,
                model,
                device,
                (calib_metrics / "test" / f"rep_{r}" / f"iter_{i}"),
            )

            with open(metrics / f"rep_{r}_iter_{i}.pkl", "wb") as fp:
                payload = {
                    "history": history,
                    "test_metrics": test_metrics,
                    "labelled_classes": dm.unlabelled.labelled_classes,
                    "labelled_indices": dm.unlabelled.labelled_indices,
                    "bald_scores": bald_scores,
                }
                pickle.dump(payload, fp)
            torch.save(model.state_dict(), saved_models / f"rep_{r}_iter_{i}.pt")

            # finally, acquire points
            acquired_idxs, _ = dm.acquire(b=b)
            if acq_name == "bald":
                # acquired_idxs has the top b scores from recent_score
                bald_scores = (acquired_idxs, acq_fn.recent_score)

            # flush results frequently for the impatient
            with open(template + "_accs.pkl", "wb") as fp:
                pickle.dump(accs, fp)


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--seed", type=int)
    args.add_argument("--acq", choices=["bald", "random"], default="bald")
    args.add_argument("--iters", type=int, default=24)
    args.add_argument("--reps", type=int, default=1)
    args.add_argument("--b", type=int, default=10)
    args = args.parse_args()
    main(args.acq, b=args.b, iters=args.iters, reps=args.reps, seed=args.seed)
