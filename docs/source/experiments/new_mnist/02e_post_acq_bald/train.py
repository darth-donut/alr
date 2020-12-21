# MNIST experiment 2 post-acquisition
# Dependency: BALD10 to complete because we just take BALD10's acquired points
# and add SSL training post-acquisition.
from alr.training.ephemeral_trainer import EphemeralTrainer
from alr.utils import manual_seed, timeop
from alr.data.datasets import Dataset
from alr.data import UnlabelledDataset
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


def main(threshold, metrics_path, seed):
    print(f"Starting experiment with seed {seed}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kwargs = dict(num_workers=4, pin_memory=True)

    # ========= CONSTANTS ===========
    BATCH_SIZE = 64
    # at least have this much points in one epoch (see RandomFixedLengthSampler)
    MIN_TRAIN_LENGTH = 12_500
    VAL_SIZE = 5_000
    SSL_ITERATIONS = 200
    EPOCHS = 200
    REPS = 1
    ITERS = 24

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

    original_train = train

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

    root = Path(metrics_path) / f"bald_10_{seed}"
    files = sorted(
        list(root.glob("rep_1*.pkl")), key=lambda x: int(str(x).split("_")[-1][:-4])
    )
    indices = []
    for f in files:
        with open(f, "rb") as fp:
            indices.append(pickle.load(fp)["labelled_indices"])
    assert len(indices) == ITERS

    template = f"preacq_10_{seed}"
    metrics = Path("metrics") / template
    calib_metrics = Path("calib_metrics") / template
    saved_models = Path("saved_models") / template
    metrics.mkdir(parents=True)
    calib_metrics.mkdir(parents=True)
    saved_models.mkdir(parents=True)

    with open(metrics / "pool_idxs.pkl", "wb") as fp:
        pickle.dump(pool_idxs, fp)

    for r in range(1, REPS + 1):
        model = MCDropout(Dataset.MNIST.model, forward=20, fast=True).to(device)
        print(f"=== repeat #{r} of {REPS} ===")
        # last pseudo-labeled dataset
        last_pld = None
        for i in range(1, ITERS + 1):
            model.reset_weights()
            # since we're collecting calibration metrics,
            # make pool return targets too. (i.e. debug mode)
            with pool.tmp_debug():
                trainer = EphemeralTrainer(
                    model,
                    pool,
                    F.nll_loss,
                    "Adam",
                    threshold=threshold,
                    min_labelled=0.1,
                    log_dir=None,
                    patience=(3, 7),
                    reload_best=True,
                    init_pseudo_label_dataset=last_pld,
                    device=device,
                    pool_loader_kwargs=kwargs,
                )
                train_loader = torchdata.DataLoader(
                    train,
                    batch_size=BATCH_SIZE,
                    sampler=RandomFixedLengthSampler(
                        train, MIN_TRAIN_LENGTH, shuffle=True
                    ),
                    **kwargs,
                )
                with timeop() as t:
                    history = trainer.fit(
                        train_loader,
                        val_loader,
                        iterations=SSL_ITERATIONS,
                        epochs=EPOCHS,
                    )
            last_pld = trainer.last_pseudo_label_dataset
            # eval on test set
            test_metrics = trainer.evaluate(test_loader)
            accs[len(train)].append(test_metrics["acc"])
            print(f"-- Iteration {i} of {ITERS} --")
            print(
                f"\ttrain: {len(train)}; pool: {len(pool)}\n"
                f"\t[test] acc: {test_metrics['acc']}; time: {t}"
            )

            # save stuff
            with pool.tmp_debug():
                pool_loader = torchdata.DataLoader(
                    pool,
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
                    "labelled_classes": pool.labelled_classes,
                    "labelled_indices": pool.labelled_indices,
                }
                pickle.dump(payload, fp)
            torch.save(model.state_dict(), saved_models / f"rep_{r}_iter_{i}.pth")

            # there's nothing in indices anymore. stop.
            if i == ITERS:
                continue

            # finally, acquire points
            # since indices[i] is cumulative, reset the pool
            pool.reset()
            # now get the points that BALD acquired
            acq_ds = pool.label(indices[i])
            # and append it to train
            train = torchdata.ConcatDataset([original_train, acq_ds])

            with open(f"{template}_accs.pkl", "wb") as fp:
                pickle.dump(accs, fp)


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--seed", type=int)
    args = args.parse_args()
    main(threshold=0.90, metrics_path="bald_metrics", seed=args.seed)
