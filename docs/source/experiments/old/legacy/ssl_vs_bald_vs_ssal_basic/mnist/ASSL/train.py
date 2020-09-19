r"""
vanilla pseudo-labeling + BALD/RA
"""

from alr.utils import manual_seed, eval_fwd_exp, timeop
from alr.data.datasets import Dataset
from alr.data import DataManager, UnlabelledDataset
from alr.acquisition import BALD, RandomAcquisition
from alr.training import VanillaPLTrainer, Trainer
from alr.training.samplers import RandomFixedLengthSampler
from alr import MCDropout

import torch
import pickle
import torch.utils.data as torchdata
from torch.nn import functional as F
from collections import defaultdict
from pathlib import Path


def main(acq_name, b, iters, log_every):
    manual_seed(42, det_cudnn=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    kwargs = dict(num_workers=4, pin_memory=True)

    # ========= CONSTANTS ===========
    # according to the paper:
    BATCH_SIZE = 32
    UNLABELLED_BATCH_SIZE = 256
    # with early stopping, this'll probably be lesser
    EPOCHS = 200
    # at least have this much points in one epoch (see RandomFixedLengthSampler)
    MIN_TRAIN_LENGTH = 12_500
    VAL_SIZE = 5_000

    REPEATS = 6
    ITERS = iters

    # ========= SETUP ===========
    train, pool, test = Dataset.MNIST.get_fixed()
    model = MCDropout(Dataset.MNIST.model, forward=20, fast=True).to(device)
    pool, val = torchdata.random_split(pool, (len(pool) - VAL_SIZE, VAL_SIZE))
    pool = UnlabelledDataset(pool, debug=True)
    val_loader = torchdata.DataLoader(
        val, batch_size=1024, shuffle=False, **kwargs,
    )
    test_loader = torchdata.DataLoader(
        test, batch_size=1024, shuffle=False, **kwargs,
    )
    if acq_name == "bald":
        acq_fn = BALD(eval_fwd_exp(model), device=device, batch_size=1024, **kwargs)
    elif acq_name == "random":
        acq_fn = RandomAcquisition()
    else:
        raise Exception("Done goofed.")

    dm = DataManager(train, pool, acq_fn)
    accs = defaultdict(list)

    # paths
    pl_metrics = Path("pl_metrics") / f"{acq_name}_{b}"
    metrics = Path("metrics") / f"{acq_name}_{b}"
    saved_models = Path("saved_models") / f"{acq_name}_{b}"
    metrics.mkdir(parents=True)
    saved_models.mkdir(parents=True)

    for r in range(1, REPEATS + 1):
        print(f"\n- [{acq_name} (b={b})] repeat #{r} of {REPEATS}-")
        dm.reset()
        for i in range(1, ITERS + 1):
            print(f"=== Iteration {i} of {ITERS} ({i/ITERS:.2%}) ===")
            print(f"\ttrain: {dm.n_labelled}; val: {len(val)}; "
                  f"pool: {dm.n_unlabelled}; test: {len(test)}")
            # ---- SSL training stage ----
            pool.debug = True
            model.reset_weights()
            if (i - 1) % log_every == 0 and r == 1:
                pl_log = str(pl_metrics / f"dsize_{dm.n_labelled}")
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
                dm.labelled, batch_size=BATCH_SIZE,
                sampler=RandomFixedLengthSampler(dm.labelled, MIN_TRAIN_LENGTH, shuffle=True),
                **kwargs
            )
            pool_loader = torchdata.DataLoader(
                dm.unlabelled, batch_size=UNLABELLED_BATCH_SIZE, shuffle=True, **kwargs,
            )
            with timeop() as t:
                history = trainer.fit(train_loader, pool_loader, val_loader, epochs=EPOCHS)

            test_metric = trainer.evaluate(test_loader)
            print(f"\t[test] acc: {test_metric['acc']}, time: {t}")
            accs[dm.n_labelled].append(test_metric['acc'])

            if pl_log:
                torch.save(model.state_dict(), saved_models / f"repeat_{r}_dsize_{dm.n_labelled}_weights.pth")

            payload = {
                'history': history,
                'test_metrics': test_metric,
            }
            with open(metrics / f"repeat_{r}_dsize_{dm.n_labelled}_metrics.pkl", "wb") as fp:
                pickle.dump(payload, fp)

            # ---- pure-supervised training stage ----
            model.reset_weights()
            sup_trainer = Trainer(
                model, F.nll_loss, optimiser='Adam',
                patience=3, reload_best=True, device=device
            )
            train_loader = torchdata.DataLoader(
                dm.labelled, batch_size=BATCH_SIZE,
                sampler=RandomFixedLengthSampler(dm.labelled, MIN_TRAIN_LENGTH, shuffle=True),
                **kwargs
            )
            sup_trainer.fit(train_loader, val_loader, epochs=EPOCHS)

            # ---- acquisition stage ----
            # done with training, turn it off
            pool.debug = False
            dm.acquire(b=b)

    with open(f"{acq_name}_accs_{b}.pkl", "wb") as fp:
        pickle.dump(accs, fp)


if __name__ == '__main__':
    main("bald", b=10, iters=24, log_every=2)
    main("random", b=10, iters=24, log_every=2)
    main("bald", b=1, iters=240, log_every=20)

