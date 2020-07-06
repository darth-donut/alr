# BALD vs BALD and mixup training
import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from ignite.engine import Engine, create_supervised_evaluator, Events
from ignite.metrics import Loss, Accuracy
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data as torchdata

from alr import MCDropout
from alr.acquisition import BALD
from alr.data import UnlabelledDataset, DataManager
from alr.data.datasets import Dataset
from alr.training.samplers import RandomFixedLengthSampler
from alr.training.utils import EarlyStopper, PLPredictionSaver
from alr.utils import _map_device
from alr.utils import timeop, eval_fwd_exp, manual_seed
from alr.utils._type_aliases import _DeviceType


# # from https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py#L119
# def mixup(x: torch.Tensor,
#           y: torch.Tensor,
#           alpha: float = 1.0,
#           device: _DeviceType = None):
#     '''Returns mixed inputs, pairs of targets, and lambda'''
#     if alpha > 0:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1
#     batch_size = x.size()[0]
#     index = torch.randperm(batch_size)
#     if device:
#         index = index.to(device)
#     mixed_x = lam * x + (1 - lam) * x[index, :]
#     y_a, y_b = y, y[index]
#     return mixed_x, y_a, y_b, lam


# def reg_mixup_loss(coef: Optional[Tuple[float, float]] = (.8, .4)):
#     def _reg_mixup_loss(pred: torch.Tensor,
#                         y1: torch.Tensor,
#                         y2: torch.Tensor,
#                         lamb: int):
#         """
#         pred is log_softmax,
#         y1 and y2 are softmax probabilities
#         """
#         C = y1.size(-1)
#         assert y2.size(-1) == C
#         # NxC
#         prob = pred.exp()
#         # C
#         prob_avg = prob.mean(dim=0)
#         prior = y2.new_ones(C) / C
#
#         # term1, term2, [1,]
#         term1 = -torch.mean(torch.sum(y1 * pred, dim=1))
#         term2 = -torch.mean(torch.sum(y2 * pred, dim=1))
#         mixup_loss = lamb * term1 + (1 - lamb) * term2
#
#         prior_loss = -torch.sum(
#             prior * torch.log(prob_avg)
#         )
#         entropy_loss = -torch.mean(
#             torch.sum(prob * pred, dim=1)
#         )
#
#         return mixup_loss + coef[0] * prior_loss + coef[1] * entropy_loss
#
#     return _reg_mixup_loss


# def onehot_transform(n):
#     def _onehot_transform(x):
#         return torch.eye(n)[x]
#
#     return _onehot_transform


def calib_metrics(loader, model: nn.Module, log_dir, device):
    evaluator = create_supervised_evaluator(
        model, metrics=None, device=device
    )
    pds = PLPredictionSaver(log_dir)
    pds.attach(evaluator)
    evaluator.run(loader)


def mixup_data(x, y, alpha, device):
    # from https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
    '''Returns mixed inputs, pairs of targets, and lambda'''
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


def create_mixup_trainer(model: nn.Module,
                         loss_fn, optimiser,
                         device, alpha):
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


def train_model(model: nn.Module,
                loss_fn, optimiser, device, alpha,
                train_loader: torchdata.DataLoader,
                val_loader: torchdata.DataLoader,
                patience: int,
                epochs: int):
    # scheduler = ReduceLROnPlateau(
    #     optimiser, mode='max',
    #     factor=.1, patience=10,
    #     verbose=True,
    # )

    val_eval = create_supervised_evaluator(
        model, metrics={'acc': Accuracy(), 'nll': Loss(F.nll_loss)},
        device=device
    )
    history = {'acc': [], 'loss': []}

    trainer = create_mixup_trainer(
        model, loss_fn, optimiser, device, alpha
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def _log(e: Engine):
        metrics = val_eval.run(val_loader).metrics
        acc, loss = metrics['acc'], metrics['nll']
        print(f"\tepoch {e.state.epoch:3}: "
              f"[val] acc, loss = {acc:.4f}, {loss:.4f}")
        history['acc'].append(acc)
        history['loss'].append(loss)
        # scheduler.step(acc)

    es = EarlyStopper(model, patience, trainer, key='acc', mode='max')
    es.attach(val_eval)

    trainer.run(
        train_loader, max_epochs=epochs,
    )
    es.reload_best()
    return history


def evaluate_model(loader, model: nn.Module, loss_fn, device):
    evaluator = create_supervised_evaluator(
        model, metrics={'acc': Accuracy(),
                        'nll': Loss(loss_fn)},
        device=device
    )
    metrics = evaluator.run(loader).metrics
    return metrics['acc'], metrics['nll']


def main(reps, iters, b, alpha):
    manual_seed(42)
    EPOCHS = 200
    # ReduceLROnPlateau's patience = 10 => decrease LR at least 2 times before stopping
    PATIENCE = 5
    VAL_LEN = 5_000
    BATCH_SIZE = 128
    DECAY = 1e-4
    LR = 0.1
    RFLS = 10_500
    MOMENTUM = 0.9

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        kwargs = dict(num_workers=4, pin_memory=True)
    else:
        device = torch.device('cpu')
        kwargs = {}
    accs = defaultdict(list)
    template = f"b_{b}_alpha_{alpha}"
    metrics = Path("metrics") / template
    saved_models = Path("saved_models") / template
    calib = Path("calib_metrics") / template
    metrics.mkdir(exist_ok=False, parents=True)
    saved_models.mkdir(exist_ok=False, parents=True)

    for r in range(1, reps + 1):
        print(f"- repeat {r} of {reps} - ")
        train, pool, test = Dataset.MNIST.get_fixed()
        pool, val = torchdata.random_split(pool, (len(pool) - VAL_LEN, VAL_LEN))
        pool = UnlabelledDataset(pool)
        model = MCDropout(Dataset.MNIST.model, forward=20, fast=True).to(device)
        bald = BALD(eval_fwd_exp(model), device=device, batch_size=1024, **kwargs)
        dm = DataManager(train, pool, bald)
        val_loader = torchdata.DataLoader(
            val, batch_size=1024, shuffle=False, **kwargs
        )
        test_loader = torchdata.DataLoader(
            test, batch_size=1024, shuffle=False, **kwargs,
        )

        for i in range(1, iters + 1):
            print(f"=== iteration {i} of {iters} ===")
            model.reset_weights()
            # because in each iteratin, we modify the learning rate.
            # optimiser = torch.optim.SGD(
            #     model.parameters(), lr=LR,
            #     momentum=MOMENTUM,
            #     weight_decay=DECAY,
            # )
            optimiser = torch.optim.Adam(model.parameters())
            train_loader = torchdata.DataLoader(
                dm.labelled, batch_size=BATCH_SIZE,
                sampler=RandomFixedLengthSampler(dm.labelled, RFLS, shuffle=True),
                **kwargs,
            )
            with timeop() as t:
                history = train_model(
                    model, F.nll_loss, optimiser,
                    device, alpha, train_loader, val_loader,
                    patience=PATIENCE, epochs=EPOCHS
                )
            test_acc, test_loss = evaluate_model(
                test_loader, model, F.nll_loss, device=device
            )
            accs[dm.n_labelled].append(test_acc)
            print(f"\ttrain: {dm.n_labelled}; pool: {dm.n_unlabelled}\n"
                  f"\t[test] acc, loss = {test_acc}, {test_loss}; time taken: {t}")
            payload = {
                'test_metrics': (test_acc, test_loss),
                'history': history,
            }
            with open(metrics / f"rep_{r}_iter_{i}.pkl", "wb") as fp:
                pickle.dump(payload, fp)
            torch.save(model.state_dict(), saved_models / f"rep_{r}_iter_{i}.pt")
            calib_metrics(test_loader, model, calib / f"rep_{r}" / f"iter_{i}", device=device)
            dm.acquire(b)


if __name__ == '__main__':
    # don't regularise the loss (at least, not yet.)
    # use argparse for alpha in: 0.1, 0.2, 0.3, 0.4, 1.0
    args = argparse.ArgumentParser()
    args.add_argument(
        "--reps", type=int,
        help="# of repetitions [default = 6]", default=6
    )
    args.add_argument(
        "--iters", type=int,
        help="# of iterations [default = 24]", default=24,
    )
    args.add_argument(
        "--mixup_alpha", type=float,
        help="mixup's alpha parameter",
    )
    args.add_argument(
        "--acquisition_size", type=int,
        help="batch acquisition size", default=10,
    )
    args = args.parse_args()
    main(reps=args.reps,
         iters=args.iters,
         b=args.acquisition_size,
         alpha=args.mixup_alpha)

