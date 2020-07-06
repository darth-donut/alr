r"""
some baseline comparisons for this experiment: BALD b = 10, b = 1, and RA b = 10
"""


from alr.utils import manual_seed, eval_fwd_exp, timeop
from alr.data.datasets import Dataset
from alr.data import DataManager, UnlabelledDataset
from alr.acquisition import BALD, RandomAcquisition
from alr.training import Trainer
from alr.training.samplers import RandomFixedLengthSampler
from alr import MCDropout

import torch
import pickle
import torch.utils.data as torchdata
from torch.nn import functional as F
from collections import defaultdict


def main(acq_name, b, iters):
    manual_seed(42, det_cudnn=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    kwargs = dict(num_workers=4, pin_memory=True)

    # ========= CONSTANTS ===========
    BATCH_SIZE = 64
    # with early stopping, this'll probably be lesser
    EPOCHS = 50
    # at least have this much points in one epoch (see RandomFixedLengthSampler)
    MIN_TRAIN_LENGTH = 12_500
    VAL_SIZE = 5_000

    REPEATS = 6
    ITERS = iters

    # ========= SETUP ===========
    train, pool, test = Dataset.MNIST.get_fixed()
    model = MCDropout(Dataset.MNIST.model, forward=20, fast=True).to(device)
    pool, val = torchdata.random_split(pool, (len(pool) - VAL_SIZE, VAL_SIZE))
    pool = UnlabelledDataset(pool)
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

    for r in range(1, REPEATS + 1):
        print(f"- [{acq_name} (b={b})] repeat #{r} of {REPEATS}-")
        dm.reset()
        for i in range(1, ITERS + 1):
            print(f"=== Iteration {i} of {ITERS} ({i/ITERS:.2%}) ===")
            print(f"\ttrain: {dm.n_labelled}; val: {len(val)}; "
                  f"pool: {dm.n_unlabelled}; test: {len(test)}")
            model.reset_weights()
            trainer = Trainer(
                model, F.nll_loss, optimiser='Adam',
                patience=3, reload_best=True, device=device
            )
            train_loader = torchdata.DataLoader(
                dm.labelled, batch_size=BATCH_SIZE,
                sampler=RandomFixedLengthSampler(dm.labelled, MIN_TRAIN_LENGTH, shuffle=True),
                **kwargs
            )
            with timeop() as t:
                trainer.fit(train_loader, val_loader, epochs=EPOCHS)
            test_metric = trainer.evaluate(test_loader)
            print(f"\t[test] acc: {test_metric['acc']}, time: {t}")
            accs[dm.n_labelled].append(test_metric['acc'])
            dm.acquire(b=b)

    with open(f"{acq_name}_accs_{b}.pkl", "wb") as fp:
        pickle.dump(accs, fp)


if __name__ == '__main__':
    main("bald", b=10, iters=24)
    main("random", b=10, iters=24)
    main("bald", b=1, iters=240)
