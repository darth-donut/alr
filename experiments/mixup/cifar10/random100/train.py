import numpy as np
from alr.utils import manual_seed, eval_fwd_exp, timeop
from alr.data.datasets import Dataset
from alr.data import DataManager, UnlabelledDataset
from alr.acquisition import BALD, RandomAcquisition, _bald_score
from alr.training.samplers import RandomFixedLengthSampler
from alr.training.utils import PLPredictionSaver, EarlyStopper
from alr import MCDropout

from alr.utils import _map_device
from ignite.engine import Engine, create_supervised_evaluator, Events
from ignite.metrics import Loss, Accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import pickle
import torch.utils.data as torchdata
from torch.nn import functional as F
from collections import defaultdict
from pathlib import Path
from torch import nn


def calc_calib_metrics(loader, model: nn.Module, log_dir, device):
    evaluator = create_supervised_evaluator(model, metrics=None, device=device)
    pds = PLPredictionSaver(log_dir)
    pds.attach(evaluator)
    evaluator.run(loader)


def mixup_data(x, y, alpha, device):
    # from https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# from https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def create_mixup_trainer(model: nn.Module, loss_fn, optimiser, device, alpha):
    def _step(_, batch):
        model.train()
        x, y = batch
        x, y = _map_device([x, y], device)
        xp, y1, y2, lamb = mixup_data(x, y, alpha=alpha, device=device)
        preds = model(xp)
        loss = mixup_criterion(loss_fn, preds, y1, y2, lamb)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        return loss.item()

    return Engine(_step)


def train_model(
    model: nn.Module,
    loss_fn,
    optimiser,
    device,
    alpha,
    train_loader: torchdata.DataLoader,
    val_loader: torchdata.DataLoader,
    patience: int,
    epochs: int,
):
    scheduler = ReduceLROnPlateau(
        optimiser,
        mode="max",
        factor=0.1,
        patience=10,
        verbose=True,
    )

    val_eval = create_supervised_evaluator(
        model, metrics={"acc": Accuracy(), "nll": Loss(F.nll_loss)}, device=device
    )
    history = {"acc": [], "loss": []}

    trainer = create_mixup_trainer(model, loss_fn, optimiser, device, alpha)

    @trainer.on(Events.EPOCH_COMPLETED)
    def _log(e: Engine):
        metrics = val_eval.run(val_loader).metrics
        acc, loss = metrics["acc"], metrics["nll"]
        print(f"\tepoch {e.state.epoch:3}: " f"[val] acc, loss = {acc:.4f}, {loss:.4f}")
        history["acc"].append(acc)
        history["loss"].append(loss)
        scheduler.step(acc)

    es = EarlyStopper(model, patience, trainer, key="acc", mode="max")
    es.attach(val_eval)

    trainer.run(
        train_loader,
        max_epochs=epochs,
    )
    es.reload_best()
    return history


def evaluate_model(loader, model: nn.Module, loss_fn, device):
    evaluator = create_supervised_evaluator(
        model, metrics={"acc": Accuracy(), "loss": Loss(loss_fn)}, device=device
    )
    return evaluator.run(loader).metrics


def main(acq_name, alpha, b, iters, repeats):
    manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kwargs = dict(num_workers=4, pin_memory=True)

    # ========= CONSTANTS ===========
    BATCH_SIZE = 128
    # with early stopping, this'll probably be lesser
    EPOCHS = 200
    # at least have this much points in one epoch (see RandomFixedLengthSampler)
    MIN_TRAIN_LENGTH = 20_000
    VAL_SIZE = 5_000
    LR = 0.1
    MOMENTUM = 0.9
    DECAY = 1e-4
    PATIENCE = 20

    REPEATS = repeats
    ITERS = iters

    # ========= SETUP ===========
    train, pool, test = Dataset.CIFAR10.get_fixed()
    pool, val = torchdata.random_split(pool, (len(pool) - VAL_SIZE, VAL_SIZE))
    pool = UnlabelledDataset(pool)
    val_loader = torchdata.DataLoader(
        val,
        batch_size=512,
        shuffle=False,
        **kwargs,
    )
    test_loader = torchdata.DataLoader(
        test,
        batch_size=512,
        shuffle=False,
        **kwargs,
    )
    accs = defaultdict(list)

    template = f"{acq_name}_{b}_alpha_{alpha}"
    metrics = Path("metrics") / template
    calib_metrics = Path("calib_metrics") / template
    saved_models = Path("saved_models") / template
    metrics.mkdir(parents=True)
    calib_metrics.mkdir(parents=True)
    saved_models.mkdir(parents=True)
    bald_scores = None

    # since we need to know which points were taken for val dataset
    with open(metrics / "pool_idxs.pkl", "wb") as fp:
        pickle.dump(pool._dataset.indices, fp)

    for r in range(1, REPEATS + 1):
        print(f"- [{acq_name} (b={b})] repeat #{r} of {REPEATS}-")
        model = MCDropout(Dataset.CIFAR10.model, forward=20, fast=False).to(device)
        if acq_name == "bald":
            acq_fn = BALD(eval_fwd_exp(model), device=device, batch_size=512, **kwargs)
        elif acq_name == "random":
            acq_fn = RandomAcquisition()
        else:
            raise Exception("Done goofed.")

        dm = DataManager(train, pool, acq_fn)
        dm.reset()  # this resets pool

        for i in range(1, ITERS + 1):
            model.reset_weights()
            # because in each iteration, we modify the learning rate.
            optimiser = torch.optim.SGD(
                model.parameters(),
                lr=LR,
                momentum=MOMENTUM,
                weight_decay=DECAY,
            )
            train_loader = torchdata.DataLoader(
                dm.labelled,
                batch_size=BATCH_SIZE,
                sampler=RandomFixedLengthSampler(
                    dm.labelled, MIN_TRAIN_LENGTH, shuffle=True
                ),
                **kwargs,
            )
            with timeop() as t:
                history = train_model(
                    model,
                    F.nll_loss,
                    optimiser,
                    device,
                    alpha,
                    train_loader,
                    val_loader,
                    patience=PATIENCE,
                    epochs=EPOCHS,
                )

            # eval
            test_metrics = evaluate_model(test_loader, model, F.nll_loss, device=device)

            print(f"=== Iteration {i} of {ITERS} ({i/ITERS:.2%}) ===")
            print(
                f"\ttrain: {dm.n_labelled}; val: {len(val)}; "
                f"pool: {dm.n_unlabelled}; test: {len(test)}"
            )
            print(f"\t[test] acc: {test_metrics['acc']:.4f}, time: {t}")
            accs[dm.n_labelled].append(test_metrics["acc"])

            # save stuff

            # pool calib
            with dm.unlabelled.tmp_debug():
                pool_loader = torchdata.DataLoader(
                    dm.unlabelled,
                    batch_size=512,
                    shuffle=False,
                    **kwargs,
                )
                calc_calib_metrics(
                    pool_loader,
                    model,
                    calib_metrics / "pool" / f"rep_{r}" / f"iter_{i}",
                    device=device,
                )
            calc_calib_metrics(
                test_loader,
                model,
                calib_metrics / "test" / f"rep_{r}" / f"iter_{i}",
                device=device,
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
            torch.save(model.state_dict(), saved_models / f"rep_{r}_iter_{i}.pt")
            # flush results frequently for the impatient
            with open(template + "_accs.pkl", "wb") as fp:
                pickle.dump(accs, fp)

            # finally, acquire points
            acquired_idxs, acquired_ds = dm.acquire(b=b)
            # if bald, store ALL bald scores and the acquired idx so we can map the top b scores
            # to the b acquired_idxs
            if acq_name == "bald":
                # acquired_idxs has the top b scores from recent_score
                bald_scores = (acquired_idxs, acq_fn.recent_score)
            # if RA, then store the acquired indices and their associated bald scores
            else:
                # compute bald scores of Random Acq. points
                bald_scores = _bald_score(
                    pred_fn=eval_fwd_exp(model),
                    dataloader=torchdata.DataLoader(
                        acquired_ds,
                        batch_size=512,
                        shuffle=False,  # don't shuffle to get 1-1 pairing with acquired_idxs
                        **kwargs,
                    ),
                    device=device,
                )
                assert acquired_idxs.shape[0] == bald_scores.shape[0], (
                    f"Acquired idx length {acquired_idxs.shape[0]} does not"
                    f" match bald scores length {bald_scores.shape[0]}"
                )
                bald_scores = list(zip(acquired_idxs, bald_scores))


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--acq", choices=["bald", "random"], default="bald")
    args.add_argument("--alpha", default=0.4, type=float)
    args.add_argument("--b", default=10, type=int, help="Batch acq size (default = 10)")
    args.add_argument("--iters", default=199, type=int)
    args.add_argument("--reps", default=1, type=int)
    args = args.parse_args()

    main(args.acq, alpha=args.alpha, b=args.b, iters=args.iters, repeats=args.reps)
