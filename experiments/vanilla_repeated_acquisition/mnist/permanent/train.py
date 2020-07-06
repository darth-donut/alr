r"""

Repeatedly acquire and self-label confident points -- permanently added into the training set.

2 variants: 1) in conjunction with active learning
            2) w/o active learning
"""

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


def main(use_al, b, threshold, log_every):
    manual_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    kwargs = dict(num_workers=4, pin_memory=True)

    # --- constants ---
    BATCH_SIZE = 64
    EPOCHS = 200
    REPS = 6
    ITERS = 24
    VAL_SIZE = 5_000
    MIN_TRAIN_LEN = 12_500

    # --- setup ---
    train, pool, test = Dataset.MNIST.get_fixed()
    val, pool = torchdata.random_split(pool, (VAL_SIZE, len(pool) - VAL_SIZE))
    pool = UnlabelledDataset(pool)
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

    accs = []
    template = f"{'al' if use_al else 'no_al'}_b={b}_thresh={threshold}"
    pl_metrics = Path("pl_metrics") / template
    metrics = Path("metrics") / template
    saved_models = Path("saved_models") / template
    metrics.mkdir(parents=True)
    saved_models.mkdir(parents=True)

    for r in range(1, REPS + 1):
        print(f"- Repeat {r} of {REPS} -")
        dm.reset()
        accs_r = {}
        for i in range(1, ITERS + 1):
            print(f"=== Iteration {i} of {ITERS} ({i / ITERS:.2%}) ===")
            print(f"\ttrain: {dm.n_labelled}; "
                  f"pool: {dm.n_unlabelled}; "
                  f"val: {len(val)}; "
                  f"test: {len(test)}")
            model.reset_weights()

            # -- stage 1: train --
            trainer = Trainer(model, F.nll_loss, 'Adam', patience=3, reload_best=True, device=device)
            train_loader = torchdata.DataLoader(
                dm.labelled, batch_size=BATCH_SIZE,
                sampler=RandomFixedLengthSampler(dm.labelled, MIN_TRAIN_LEN, shuffle=True),
                **kwargs,
            )
            with timeop() as t:
                history = trainer.fit(train_loader, val_loader, epochs=EPOCHS)

            test_metrics = trainer.evaluate(test_loader)
            accs_r[dm.n_labelled] = test_metrics['acc']

            print(f"\t[test] loss, acc: ({test_metrics['loss']:.4f}, {test_metrics['acc']:.4f}); time: {t}")

            with open(metrics / f"repeat_{r}_dsize_{dm.n_labelled}_metrics.pkl", "wb") as fp:
                payload = {
                    'history': history,
                    'test_metrics': test_metrics,
                }
                pickle.dump(payload, fp)

            if (i - 1) % log_every == 0:
                torch.save(model.state_dict(), saved_models / f"repeat_{r}_dsize_{dm.n_labelled}_weights.pth")

            # skip if this is the last iteration
            if i == ITERS:
                continue

            # -- stage 2: acquire more data into the training set --

            # -- stage 2.1: acquire using AL acquisition function --
            if use_al:
                dm.acquire(b)

            # -- stage 2.2: acquire using pseudo-labels --
            pool.debug = True  # to get the true labels from the unlabelled pool (for evaluation purposes)
            idxs, plabs = get_confident_indices(
                model=model, dataset=dm.unlabelled,
                threshold=threshold, root=(pl_metrics / f"repeat_{r}"),
                step=i, device=device, **kwargs
            )
            pool.debug = False

            if idxs.shape[0]:
                n_unlabelled_before = dm.n_unlabelled
                # remove these from the unlabelled pool
                truth = pool.label(idxs)
                # sanity check
                assert dm.n_unlabelled == (n_unlabelled_before - idxs.shape[0])

                # replace true labels with pseudo-labels
                relabelled_dataset = RelabelledDataset(truth, plabs)
                assert len(relabelled_dataset) == idxs.shape[0]

                # add to the labelled pool
                n_labelled_before = dm.n_labelled
                dm.append_to_labelled(relabelled_dataset)
                assert dm.n_labelled == (n_labelled_before + idxs.shape[0])
            else:
                print(f"\tSelf-labelling didn't happen because none of the pseudo-labels are confident enough.")
        accs.append(accs_r)
    with open(f"{template}_accs.pkl", "wb") as fp:
        pickle.dump(accs, fp)


if __name__ == '__main__':
    main(use_al=True, b=10, threshold=0.9, log_every=2)
    main(use_al=False, b=10, threshold=0.9, log_every=2)
