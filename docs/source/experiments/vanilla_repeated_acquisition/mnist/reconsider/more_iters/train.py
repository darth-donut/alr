r"""
Repeatedly acquire and self-label confident points without adding them into the training set.
These points remain in the unlabelled pool and will be reconsidered in the next iteration.

2 variants: 1) in conjunction with active learning
            2) w/o active learning

warning: bug in code: when storing accuracy (pkl) file, we assume that no future iterations
will have the same number of training points (incl. pseudo-labelled points). However, this is
not always true. Thus, the array may have less than ITER values because the value has been overridden
when the key (# training points) is identical. Solution: just look into the metrics folder
where payload is saved.
"""

from alr.utils import manual_seed, eval_fwd_exp, timeop
from alr.acquisition import BALD
from alr import MCDropout
from alr.data.datasets import Dataset
from alr.training.samplers import RandomFixedLengthSampler
from alr.data import UnlabelledDataset, DataManager
from alr.training import Trainer
from alr.training.repeated_acquisition_utils import (
    get_confident_indices,
    RelabelledDataset,
)

import torch
import torch.utils.data as torchdata
import pickle
from torch.nn import functional as F
from pathlib import Path


def main(use_al, b, threshold, log_every):
    manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kwargs = dict(num_workers=4, pin_memory=True)

    # --- constants ---
    BATCH_SIZE = 64
    EPOCHS = 200
    REPS = 6
    ITERS = 50
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
        accs_r = []
        # store temporarily labelled points (will be union-ed with the training dataset)
        pseudo_labelled_points = None
        for i in range(1, ITERS + 1):
            if pseudo_labelled_points is not None:
                full_train_dataset = torchdata.ConcatDataset(
                    (dm.labelled, pseudo_labelled_points)
                )
            else:
                full_train_dataset = dm.labelled
            train_length = len(full_train_dataset)
            print(f"=== Iteration {i} of {ITERS} ({i / ITERS:.2%}) ===")
            print(
                f"\ttrain: {train_length}; "
                f"pool: {dm.n_unlabelled}; "
                f"val: {len(val)}; "
                f"test: {len(test)}"
            )
            model.reset_weights()

            # -- stage 1: train --
            trainer = Trainer(
                model, F.nll_loss, "Adam", patience=3, reload_best=True, device=device
            )
            train_loader = torchdata.DataLoader(
                full_train_dataset,
                batch_size=BATCH_SIZE,
                sampler=RandomFixedLengthSampler(
                    full_train_dataset, MIN_TRAIN_LEN, shuffle=True
                ),
                **kwargs,
            )
            with timeop() as t:
                history = trainer.fit(train_loader, val_loader, epochs=EPOCHS)

            test_metrics = trainer.evaluate(test_loader)
            accs_r.append(test_metrics["acc"])

            print(
                f"\t[test] loss, acc: ({test_metrics['loss']:.4f}, {test_metrics['acc']:.4f}); time: {t}"
            )

            with open(
                metrics / f"repeat_{r}_dsize_{train_length}_metrics.pkl", "wb"
            ) as fp:
                payload = {
                    "history": history,
                    "test_metrics": test_metrics,
                }
                pickle.dump(payload, fp)

            if (i - 1) % log_every == 0:
                torch.save(
                    model.state_dict(),
                    saved_models / f"repeat_{r}_dsize_{train_length}_weights.pth",
                )

            # skip if this is the last iteration
            if i == ITERS:
                continue

            # -- stage 2: acquire more data into the training set --

            # -- stage 2.1: acquire using AL acquisition function --
            pool.debug = False
            if use_al:
                dm.acquire(b)
            # stage 2.2's Subset dataset (RelabelledDataset) expects (x, y) pair
            pool.debug = True

            # -- stage 2.2: acquire using pseudo-labels --
            idxs, plabs = get_confident_indices(
                model=model,
                dataset=dm.unlabelled,
                threshold=threshold,
                root=((pl_metrics / f"repeat_{r}") if r == 1 else None),
                step=i,
                device=device,
                **kwargs,
            )

            if idxs.shape[0]:
                truth = torchdata.Subset(dm.unlabelled, idxs)

                # replace true labels with pseudo-labels
                pseudo_labelled_points = RelabelledDataset(truth, plabs)
                assert len(pseudo_labelled_points) == idxs.shape[0]
            else:
                print(
                    f"\tSelf-labelling didn't happen because none of the pseudo-labels are confident enough."
                )
        accs.append(accs_r)

    with open(f"{template}_accs.pkl", "wb") as fp:
        pickle.dump(accs, fp)


if __name__ == "__main__":
    main(use_al=False, b=10, threshold=0.9, log_every=2)
