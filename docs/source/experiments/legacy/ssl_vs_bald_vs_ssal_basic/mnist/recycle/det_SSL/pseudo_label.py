r"""
vanilla pseudo-labeling implementation
"""
from collections import defaultdict

from alr.utils import timeop, manual_seed
from alr.data.datasets import Dataset
from alr.data import UnlabelledDataset
from alr.training import VanillaPLTrainer
from alr.training.samplers import RandomFixedLengthSampler
from alr import MCDropout

import pickle
import numpy as np
import torch
import torch.utils.data as torchdata
from torch.nn import functional as F
from pathlib import Path


if __name__ == '__main__':
    manual_seed(42)
    kwargs = dict(num_workers=4, pin_memory=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    sizes = np.arange(20, 260, 10)
    N = len(sizes)
    # validation dataset size
    VAL_SIZE = 5_000
    # according to the paper:
    BATCH_SIZE = 32
    UNLABELLED_BATCH_SIZE = 256
    # at least prolong the epoch to have this many points (see RandomFixedLengthSampler)
    MIN_TRAIN_SIZE = 12_500
    # well, early stopping should kick-in before then.
    EPOCHS = 200
    REPEATS = 6

    # paths
    pl_metrics = Path("pl_metrics")
    metrics = Path("metrics")
    saved_models = Path("saved_models")
    metrics.mkdir()
    saved_models.mkdir()
    log_every = 2

    accs = defaultdict(list)

    for r in range(1, REPEATS + 1):
        for i, n in enumerate(sizes, 1):
            train, test = Dataset.MNIST.get()
            train, pool = torchdata.random_split(train, (n, len(train) - n))
            pool, val = torchdata.random_split(pool, (len(pool) - VAL_SIZE, VAL_SIZE))
            pool = UnlabelledDataset(pool, debug=True)
            model = MCDropout(Dataset.MNIST.model, forward=20, fast=True).to(device)

            print(f"=== Iteration {i} of {N} ({i/N:.2%}) ===")
            print(f"\ttrain: {len(train)}; pool: {len(pool)}; test: {len(test)}")

            if (i - 1) % log_every == 0 and r == 1:
                pl_log = str(pl_metrics / f"dsize_{n}")
            else:
                pl_log = None

            trainer = VanillaPLTrainer(
                model, labelled_loss=F.nll_loss,
                unlabelled_loss=F.nll_loss, optimiser='Adam',
                patience=3, reload_best=True,
                track_pl_metrics=pl_log,
                device=device,
            )

            train_loader = torchdata.DataLoader(
                train, batch_size=BATCH_SIZE,
                sampler=RandomFixedLengthSampler(train, length=MIN_TRAIN_SIZE, shuffle=True),
                **kwargs,
            )
            pool_loader = torchdata.DataLoader(
                pool, batch_size=UNLABELLED_BATCH_SIZE, shuffle=True, **kwargs,
            )
            val_loader = torchdata.DataLoader(
                val, batch_size=1024, shuffle=False, **kwargs,
            )
            test_loader = torchdata.DataLoader(
                test, batch_size=1024, shuffle=False, **kwargs,
            )

            with timeop() as t:
                history = trainer.fit(
                    train_loader, pool_loader, val_loader, epochs=EPOCHS,
                )

            test_metrics = trainer.evaluate(test_loader)
            accs[n].append(test_metrics['acc'])
            print(f"\t[train] loss, acc: ({history['stage2']['train_loss'][-1]}, {history['stage2']['train_acc'][-1]})\n"
                  f"\t[test] loss, acc: ({test_metrics['loss']}, {test_metrics['acc']})\n"
                  f"\ttime: {t}")

            if pl_log:
                torch.save(model.state_dict(), saved_models / f"repeat_{r}_dsize_{n}_weights.pth")

            payload = {
                'history': history,
                'test_metrics': test_metrics,
            }
            with open(metrics / f"repeat_{r}_dsize_{n}_metrics.pkl", "wb") as fp:
                pickle.dump(payload, fp)

    with open("accs.pkl", "wb") as fp:
        pickle.dump(accs, fp)

