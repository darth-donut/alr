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
    pool, val = torchdata.random_split(pool, (len(pool) - VAL_SIZE, VAL_SIZE))
    pool = UnlabelledDataset(pool)
    pool_loader = torchdata.DataLoader(
        pool, batch_size=1024, shuffle=False, **kwargs,
    )
    val_loader = torchdata.DataLoader(
        val, batch_size=1024, shuffle=False, **kwargs,
    )
    test_loader = torchdata.DataLoader(
        test, batch_size=1024, shuffle=False, **kwargs,
    )
    accs = defaultdict(list)

    template = f"{acq_name}_{b}"
    metrics = Path("metrics") / template
    calib_metrics = Path("calib_metrics") / template
    saved_models = Path("saved_models") / template
    metrics.mkdir(parents=True)
    calib_metrics.mkdir(parents=True)
    saved_models.mkdir(parents=True)

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
                model, F.nll_loss, optimiser='Adam',
                patience=3, reload_best=True, device=device
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
            print(f"=== Iteration {i} of {ITERS} ({i/ITERS:.2%}) ===")
            print(f"\ttrain: {dm.n_labelled}; val: {len(val)}; "
                  f"pool: {dm.n_unlabelled}; test: {len(test)}")
            print(f"\t[test] acc: {test_metrics['acc']:.4f}, time: {t}")
            accs[dm.n_labelled].append(test_metrics['acc'])

            # save stuff
            with dm.unlabelled.tmp_debug():
                save_pl_metrics = create_supervised_evaluator(
                    model, metrics=None, device=device
                )
                pps = PLPredictionSaver(
                    log_dir=(calib_metrics / f"rep_{r}" / f"iter_{i}"),
                )
                pps.attach(save_pl_metrics)
                save_pl_metrics.run(pool_loader)

            with open(metrics / f"rep_{r}_iter_{i}.pkl", "wb") as fp:
                payload = {
                    'history': history, 'test_metrics': test_metrics,
                    'labelled_classes': dm.unlabelled.labelled_classes,
                    'labelled_indices': dm.unlabelled.labelled_indices,
                    # labelled_indices ignores the fact that there's also a val_dataset
                    # we need pool's indices to recover the true labelled_indices.
                    'pool_indices': pool._dataset.indices,
                }
                pickle.dump(payload, fp)
            torch.save(model.state_dict(), saved_models / f"rep_{r}_iter_{i}.pt")

            # finally, acquire points
            dm.acquire(b=b)

            # flush results frequently for the impatient
            with open(template + "_accs.pkl", "wb") as fp:
                pickle.dump(accs, fp)


if __name__ == '__main__':
    main("random", b=10, iters=24)
