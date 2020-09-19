r"""
From previous experiments, we saw that ephemeral pseudo-labelling helped boost accuracy
despite starting with only 20 points. We could kick-start BALD with 85% accuracy with 24 iterations
but it seems like using 80% accuracy at 10 iterations is a good trade-off. It's harder to gain more
accuracy as the number of iteration increases.

This experiment kick-starts BALD10 acquisition by warming the model to 80% accuracy (with 10 iterations
of ephemeral pseudo-labelling). However, the acquisition loop will NOT run ephemeral P.L. as we've seen
a decrease in performance when doing so. There are two possibilities: (1) warm-starting the model
has caused it to lower its entropy on the pool dataset, hence causing it to actually perform worse.
(2) warm-starting it actually helped! my bet is (unfortunately) on the former, given previous observations
(i.e. ephemeral bald10 performs worse than bald10 -- but i'm hopeful, notwithstanding.).
"""
from collections import defaultdict

from alr.utils import manual_seed, eval_fwd_exp, timeop
from alr.acquisition import BALD
from alr import MCDropout
from alr.data.datasets import Dataset
from alr.training.samplers import RandomFixedLengthSampler
from alr.data import UnlabelledDataset, DataManager
from alr.training import Trainer
from alr.training.repeated_acquisition_utils import get_confident_indices, RelabelledDataset

import torch
import torch.utils.data as torchdata
import pickle
from torch.nn import functional as F
from pathlib import Path


def main(b, threshold, warm_start_iters, log_every):
    manual_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    kwargs = dict(num_workers=4, pin_memory=True)

    # --- constants ---
    BATCH_SIZE = 64
    EPOCHS = 200
    REPS = 6
    ITERS = 23
    # +1 because of the structure of our loop
    warm_start_iters += 1
    VAL_SIZE = 5_000
    MIN_TRAIN_LEN = 12_500

    # --- setup ---
    train, pool, test = Dataset.MNIST.get_fixed()
    val, pool = torchdata.random_split(pool, (VAL_SIZE, len(pool) - VAL_SIZE))
    pool = UnlabelledDataset(pool, debug=True)
    model = MCDropout(Dataset.MNIST.model, forward=20, fast=True).to(device)
    bald = BALD(eval_fwd_exp(model), device=device, batch_size=1024, **kwargs)
    dm = DataManager(train, pool, bald)
    val_loader = torchdata.DataLoader(
        val, batch_size=1024, shuffle=False,
        **kwargs,
    )
    test_loader = torchdata.DataLoader(
        test, batch_size=1024, shuffle=False,
        **kwargs,
    )

    warm_start_accs = []
    accs = defaultdict(list)
    template = f"wsi={warm_start_iters}_b={b}_thresh={threshold}"
    pl_metrics = Path("pl_metrics") / template
    metrics = Path("metrics") / template
    saved_models = Path("saved_models") / template
    metrics.mkdir(parents=True)
    saved_models.mkdir(parents=True)

    for r in range(1, REPS + 1):
        print(f"- Repeat {r} of {REPS} -")
        dm.reset()
        ws_accs_r = {}
        # store temporarily labelled points (will be union-ed with the training dataset)
        pseudo_labelled_points = None
        for i in range(1, warm_start_iters + 1):
            if pseudo_labelled_points is not None:
                full_train_dataset = torchdata.ConcatDataset((dm.labelled, pseudo_labelled_points))
            else:
                full_train_dataset = dm.labelled
            train_length = len(full_train_dataset)
            print(f"=== Warm start iteration {i} of {warm_start_iters} ({i / warm_start_iters:.2%}) ===")
            print(f"\ttrain: {train_length}; "
                  f"pool: {dm.n_unlabelled}; "
                  f"val: {len(val)}; "
                  f"test: {len(test)}")
            model.reset_weights()

            # -- stage 1: train --
            trainer = Trainer(model, F.nll_loss, 'Adam', patience=3, reload_best=True, device=device)
            train_loader = torchdata.DataLoader(
                full_train_dataset, batch_size=BATCH_SIZE,
                sampler=RandomFixedLengthSampler(full_train_dataset, MIN_TRAIN_LEN, shuffle=True),
                **kwargs,
            )
            with timeop() as t:
                history = trainer.fit(train_loader, val_loader, epochs=EPOCHS)

            test_metrics = trainer.evaluate(test_loader)
            ws_accs_r[train_length] = test_metrics['acc']

            print(f"\t[test] loss, acc: ({test_metrics['loss']:.4f}, {test_metrics['acc']:.4f}); time: {t}")

            with open(metrics / f"repeat_{r}_dsize_{train_length}_metrics.pkl", "wb") as fp:
                payload = {
                    'history': history,
                    'test_metrics': test_metrics,
                }
                pickle.dump(payload, fp)

            if (i - 1) % log_every == 0:
                torch.save(model.state_dict(), saved_models / f"repeat_{r}_dsize_{train_length}_weights.pth")

            # skip if this is the last iteration
            if i == warm_start_iters:
                accs[dm.n_labelled].append(test_metrics['acc'])
                continue

            # -- stage 2: acquire more data into the training set --

            # -- acquire using pseudo-labels --
            dm.unlabelled.debug = True
            idxs, plabs = get_confident_indices(
                model=model, dataset=dm.unlabelled,
                threshold=threshold, root=((pl_metrics / f"repeat_{r}") if r == 1 else None),
                step=i, device=device, **kwargs
            )

            if idxs.shape[0]:
                truth = torchdata.Subset(dm.unlabelled, idxs)

                # replace true labels with pseudo-labels
                pseudo_labelled_points = RelabelledDataset(truth, plabs)
                assert len(pseudo_labelled_points) == idxs.shape[0]
            else:
                print(f"\tSelf-labelling didn't happen because none of the pseudo-labels are confident enough.")
        warm_start_accs.append(ws_accs_r)

        dm.unlabelled.debug = False

        print(f"Warm-started with {warm_start_iters} iterations. Beginning AL acquisitions")

        for i in range(1, ITERS + 1):
            dm.acquire(b=b)
            print(f"=== Iteration {i} of {ITERS} ({i / ITERS:.2%}) ===")
            print(f"\ttrain: {dm.n_labelled}; val: {len(val)}; "
                  f"pool: {dm.n_unlabelled}; test: {len(test)}")
            # model.reset_weights()   # leverage p.l. from before, DON'T reset!
            trainer = Trainer(
                model, F.nll_loss, optimiser='Adam',
                patience=3, reload_best=True, device=device
            )
            train_loader = torchdata.DataLoader(
                dm.labelled, batch_size=BATCH_SIZE,
                sampler=RandomFixedLengthSampler(dm.labelled, MIN_TRAIN_LEN, shuffle=True),
                **kwargs
            )
            with timeop() as t:
                trainer.fit(train_loader, val_loader, epochs=EPOCHS)
            test_metric = trainer.evaluate(test_loader)
            print(f"\t[test] acc: {test_metric['acc']}, time: {t}")
            accs[dm.n_labelled].append(test_metric['acc'])

    with open(f"{template}_warm_start_accs.pkl", "wb") as fp:
        pickle.dump(warm_start_accs, fp)

    with open(f"{template}_accs.pkl", "wb") as fp:
        pickle.dump(accs, fp)


if __name__ == '__main__':
    main(b=10, threshold=0.9, warm_start_iters=10, log_every=2)

