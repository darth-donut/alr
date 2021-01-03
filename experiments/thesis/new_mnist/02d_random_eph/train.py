# MNIST experiment 2 (SSL + random acq)
from ignite.engine import create_supervised_evaluator

from alr.training.utils import PLPredictionSaver
from alr.utils import eval_fwd_exp, timeop, manual_seed
from alr import MCDropout
from alr.data.datasets import Dataset
from alr.data import UnlabelledDataset, DataManager
from alr.acquisition import BALD, RandomAcquisition
from alr.training.ephemeral_trainer import EphemeralTrainer
from alr.training.samplers import RandomFixedLengthSampler

import pickle
from collections import defaultdict
from pathlib import Path
import torch.utils.data as torchdata
import torch
import numpy as np
from torch.nn import functional as F


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


def main(threshold: float, b: int, seed: int):
    print(f"Starting experiment with seed {seed}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kwargs = dict(num_workers=4, pin_memory=True)

    BATCH_SIZE = 64
    REPS = 1
    ITERS = 24
    VAL_SIZE = 5_000
    MIN_TRAIN_LEN = 12_500
    SSL_ITERATIONS = 200
    EPOCHS = 200

    accs = defaultdict(list)

    template = f"thresh_{threshold}_b_{b}_{seed}"
    calib_metrics = Path("calib_metrics") / template
    saved_models = Path("saved_models") / template
    metrics = Path("metrics") / template
    calib_metrics.mkdir(parents=True)
    saved_models.mkdir(parents=True)
    metrics.mkdir(parents=True)
    bald_scores = None

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

    manual_seed(seed)
    val, pool = torchdata.random_split(pool, (VAL_SIZE, len(pool) - VAL_SIZE))
    pool_idxs = (pool_idxs, pool.indices)
    pool = UnlabelledDataset(pool)
    test_loader = torchdata.DataLoader(test, batch_size=512, shuffle=False, **kwargs)
    val_loader = torchdata.DataLoader(val, batch_size=512, shuffle=False, **kwargs)

    with open(metrics / "pool_idxs.pkl", "wb") as fp:
        pickle.dump(pool_idxs, fp)

    for r in range(1, REPS + 1):
        model = MCDropout(Dataset.MNIST.model, forward=20, fast=True).to(device)
        ra = RandomAcquisition()
        dm = DataManager(train, pool, ra)
        dm.reset()  # to reset pool
        print(f"=== repeat #{r} of {REPS} ===")
        # last pseudo-labeled dataset
        last_pld = None
        for i in range(1, ITERS + 1):
            model.reset_weights()
            # since we're collecting calibration metrics,
            # make pool return targets too. (i.e. debug mode)
            with dm.unlabelled.tmp_debug():
                trainer = EphemeralTrainer(
                    model,
                    dm.unlabelled,
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
                    dm.labelled,
                    batch_size=BATCH_SIZE,
                    sampler=RandomFixedLengthSampler(
                        dm.labelled, MIN_TRAIN_LEN, shuffle=True
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
            accs[dm.n_labelled].append(test_metrics["acc"])
            print(f"-- Iteration {i} of {ITERS} --")
            print(
                f"\ttrain: {dm.n_labelled}; pool: {dm.n_unlabelled}\n"
                f"\t[test] acc: {test_metrics['acc']}; time: {t}"
            )

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
            torch.save(model.state_dict(), saved_models / f"rep_{r}_iter_{i}.pth")

            # finally, acquire points
            dm.acquire(b)

            with open(f"{template}_accs.pkl", "wb") as fp:
                pickle.dump(accs, fp)


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--seed", type=int)
    args = args.parse_args()
    main(threshold=0.90, b=10, seed=args.seed)
