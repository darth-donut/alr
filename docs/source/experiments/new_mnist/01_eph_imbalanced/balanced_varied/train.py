# MNIST experiment 1
from alr.utils import timeop, manual_seed, stratified_partition
from alr import MCDropout
from alr.data.datasets import Dataset
from alr.data import UnlabelledDataset
from alr.training.ephemeral_trainer import EphemeralTrainer
from alr.training.samplers import RandomFixedLengthSampler

import numpy as np
from collections import defaultdict
import pickle
from pathlib import Path
import torch.utils.data as torchdata
import torch
from torch.nn import functional as F

def main(threshold: float, seed: int):
    print(f"Starting experiment with seed {seed}")
    manual_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    kwargs = dict(num_workers=4, pin_memory=True)

    BATCH_SIZE = 64
    REPS = 1
    VAL_SIZE = 5_000
    MIN_TRAIN_LEN = 12_500
    SSL_ITERATIONS = 200
    EPOCHS = 200

    template = f"thresh_{threshold}_{seed}"
    calib_metrics = Path("calib_metrics") / template
    saved_models = Path("saved_models") / template
    metrics = Path("metrics") / template
    calib_metrics.mkdir(parents=True)
    saved_models.mkdir(parents=True)
    metrics.mkdir(parents=True)

    train, test = Dataset.MNIST.get()
    train, pool = stratified_partition(train, classes=10, size=80)
    pool_idxs = pool.indices
    class_count = defaultdict(int)
    for _, y in train:
        class_count[y] += 1
    print(class_count)


    val, pool = torchdata.random_split(pool, (VAL_SIZE, len(pool) - VAL_SIZE))
    pool_idxs = (pool_idxs, pool.indices)
    pool = UnlabelledDataset(pool)
    test_loader = torchdata.DataLoader(test, batch_size=512, shuffle=False, **kwargs)
    val_loader = torchdata.DataLoader(val, batch_size=512, shuffle=False, **kwargs)

    with open(metrics / "pool_idxs.pkl", "wb") as fp:
        pickle.dump(pool_idxs, fp)

    for r in range(1, REPS + 1):
        model = MCDropout(Dataset.MNIST.model, forward=20, fast=True).to(device)
        print(f"=== repeat #{r} of {REPS} ===")
        # since we're collecting calibration metrics,
        # make pool return targets too. (i.e. debug mode)
        with pool.tmp_debug():
            trainer = EphemeralTrainer(
                model, pool, F.nll_loss, 'Adam', threshold=threshold,
                min_labelled=0.1,
                log_dir=(calib_metrics / f"rep_{r}"), patience=(3, 7),
                reload_best=True,
                device=device, pool_loader_kwargs=kwargs
            )
            train_loader = torchdata.DataLoader(
                train, batch_size=BATCH_SIZE,
                sampler=RandomFixedLengthSampler(train, MIN_TRAIN_LEN, shuffle=True),
                **kwargs
            )
            with timeop() as t:
                history = trainer.fit(
                    train_loader, val_loader,
                    iterations=SSL_ITERATIONS, epochs=EPOCHS
                )
        # eval on test set
        test_metrics = trainer.evaluate(test_loader)
        print(f"\ttrain: {len(train)}; pool: {len(pool)}\n"
              f"\t[test] acc: {test_metrics['acc']}; time: {t}")

        # save stuff
        with open(metrics / f"rep_{r}.pkl", "wb") as fp:
            payload = {
                'history': history, 'test_metrics': test_metrics,
                'labelled_classes': pool.labelled_classes,
                'labelled_indices': pool.labelled_indices,
            }
            pickle.dump(payload, fp)
        torch.save(model.state_dict(), saved_models / f"rep_{r}.pth")

        with open(template + "_accs.pkl", "wb") as fp:
            pickle.dump(test_metrics['acc'], fp)

if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--seed", type=int)
    args = args.parse_args()
    main(threshold=.90, seed=args.seed)

