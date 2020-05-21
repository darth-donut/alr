from collections import defaultdict
from typing import Optional, Dict, Sequence, Union, Callable

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as torchdata
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events, \
    create_supervised_evaluator
from ignite.metrics import Loss, Accuracy
from torch import nn

from alr.data import UnlabelledDataset
from alr.training import Trainer
from alr.training.utils import EarlyStopper
from alr.utils import _map_device
from alr.utils.math import cross_entropy
from alr.utils._type_aliases import _DeviceType, _Loss_fn

r"""
todo(harry):
    0. Acc PLTracker for val/test dataset
    1. thresholding capabilities
"""


class WraparoundLoader:
    def __init__(self, ds: torchdata.DataLoader):
        self._ds = ds
        self._iter = iter(ds)

    def __next__(self) -> torch.Tensor:
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self._ds)
            return next(self._iter)

    def __iter__(self):
        return self


class Annealer:
    def __init__(self, step: Optional[int] = 0,
                 T1: Optional[int] = 100, T2: Optional[int] = 700,
                 alpha: Optional[float] = 3.0, step_interval: Optional[int] = 50):
        self._step = step
        self._T1 = T1
        self._T2 = T2
        self._alpha = alpha
        self._step_interval = step_interval

    def step(self, _):
        self._step += 1

    @property
    def weight(self):
        if self._step < self._T1:
            return 0
        elif self._step > self._T2:
            return self._alpha
        else:
            return ((self._step - self._T1) / (self._T2 - self._T1)) * self._alpha

    def attach(self, engine: Engine):
        engine.add_event_handler(
            Events.ITERATION_COMPLETED(every=self._step_interval),
            self.step
        )


class PLTracker:
    def __init__(self,
                 entropy_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 device: _DeviceType = None):
        self._device = device
        self._last_y = None
        self._correct = self._total = 0
        self._confidence = []
        self._entropy = []
        self._track = entropy_fn is not None
        self._entropy_fn = entropy_fn

    def process_batch(self, batch):
        if self._track:
            x, y = _map_device(batch, self._device)
            assert self._last_y is None
            self._last_y = y
            assert x.size(0) == y.size(0)
            self._total += y.size(0)
            return x
        # batch = x
        return batch.to(self._device)

    def record_predictions(self, probs: torch.Tensor):
        if self._track:
            # record accuracy
            assert probs.ndim == 2  # [N x C]
            conf, preds = torch.max(probs, dim=1)
            assert self._last_y is not None
            self._correct += torch.eq(preds, self._last_y).sum().item()
            self._last_y = None

            # record confidence
            self._confidence.extend(conf.tolist())
            self._entropy.extend(self._entropy_fn(probs).sum(dim=-1).tolist())

    def attach(self, engine: Engine):
        if self._track:
            engine.add_event_handler(Events.EPOCH_STARTED, self._reset)
            engine.add_event_handler(Events.EPOCH_COMPLETED, self._flush)

    def _flush(self, engine):
        engine.state.pl_tracker['pl_acc'] = self._correct / self._total
        engine.state.pl_tracker['confidence'] = self._confidence
        engine.state.pl_tracker['entropy'] = self._entropy

    def _reset(self, engine):
        self._correct = self._total = 0
        self._confidence = []
        self._entropy = []
        engine.state.pl_tracker = {}


def soft_nll_loss(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    r"""
    Calculates the soft negative log-likelihood loss

    Args:
        preds (torch.Tensor): predictions. This is expected to be log-softmax scores.
        target (torch.Tensor): target. This is expected to be log-softmax scores.

    Returns:
        torch.Tensor: a singleton tensor with the loss value
    """
    # -1/N * sum_y p(y)log[p(y)]
    res = cross_entropy(target, preds, mode='logsoftmax').sum(dim=1).mean()
    assert torch.isfinite(res)
    return res


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    r"""
    Calculates the soft cross entropy loss. This combines log-softmax with `soft_nll_loss`.

    Args:
        logits (torch.Tensor): predictions. This is expected to be logits.
        target (torch.Tensor): target. This is expected to be logits.

    Returns:
        torch.Tensor: a singleton tensor with the loss value
    """
    # -1/N * sum_y p(y)log[p(y)]
    res = cross_entropy(target, logits, mode='logits').sum(dim=1).mean()
    assert torch.isfinite(res)
    return res


def create_semisupervised_trainer(model: nn.Module, optimiser,
                                  lloss_fn: _Loss_fn, uloss_fn: _Loss_fn,
                                  annealer: Annealer,
                                  train_iterable: WraparoundLoader,
                                  tracker: PLTracker,
                                  use_soft_labels: bool = False,
                                  device: _DeviceType = None):

    def _step(_, batch):
        x = tracker.process_batch(batch)
        # get pseudo-labels for this batch using eval mode
        with torch.no_grad():
            model.eval()
            preds = model(x)
            tracker.record_predictions(preds)
            if not use_soft_labels:
                preds = torch.argmax(preds, dim=1)

        # normal forward pass on pseudo_labels
        model.train()
        u_loss = uloss_fn(model(x), preds)

        # normal forward pass on training data
        model.train()
        x, y = _map_device(next(train_iterable), device)
        l_loss = lloss_fn(model(x), y)

        # total loss
        loss = l_loss + annealer.weight * u_loss
        assert torch.isfinite(loss)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        return loss.item()
    e = Engine(_step)
    tracker.attach(e)
    annealer.attach(e)
    return e


class VanillaPLTrainer:
    def __init__(self, model: nn.Module,
                 labelled_loss: _Loss_fn,
                 unlabelled_loss: _Loss_fn,
                 optimiser: str,
                 use_soft_labels: Optional[bool] = False,
                 patience: Optional[int] = None,
                 reload_best: Optional[bool] = False,
                 track_pl_metrics: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 T1: Optional[int] = 0,
                 T2: Optional[int] = 40,
                 step_interval: Optional[int] = 50,
                 device: _DeviceType = None,
                 *args, **kwargs):
        r"""
        A vanilla pseudo-label training object.

        Args:
            model (torch.nn.Module): module object
            labelled_loss (Callable): should be a function `fn` that takes `preds` and `targets` and returns
                a singleton tensor with the loss value: `loss = fn(preds, targets)`. E.g. F.nll_loss.
            unlabelled_loss (Callable): similar to `labelled_loss`, but will be used on the pool dataset instead.
            optimiser (str): a string that corresponds to the type of optimiser to use. Must be an optimiser from
                `torch.optim` (case sensitive). E.g. 'Adam'.
            use_soft_labels (bool, optional): if `True`, then `unlabelled_loss` is presumed to
                be a function that calculates the loss of soft-labels instead. Examples of such loss functions are
                :func:`~alr.training.pseudo_label_trainer.soft_nll_loss` or
                :func:`~alr.training.pseudo-Label_trainer.soft_cross_entropy`.
            patience (int, optional): if not `None`, then validation accuracy will be used to determine when to stop.
            reload_best (bool, optional): patience must be non-`None` if this is set to `True`: reloads the best model
                according to validation accuracy at the end of training.
            track_pl_metrics (Callable, optional): tracks the quality , uncertainty, and entropy of pseudo-labels
                throughout the training epochs. :meth:`fit` will also return additional keys representing
                these aforementioned metrics. This callable is expected to take in a tensor (output of model)
                and return the (unreduced) entropy. E.g. :func:`~alr.utils.math.entropy`. If None, no tracking will
                be done.
            T1 (int, optional): when the weight coefficient starts kicking in. 0 implies it immediately starts
                taking effect since; this is probably what you want -- the model is already warm-started.
            T2 (int, optional): when the weight coefficient starts plateauing. For example,
                if `step_interval` is 50 and the number of iterations per epoch is 200,
                then there will be a total of 4 steps per epoch. If T2 is 40, then on the 10th epoch onwards,
                the coefficient is maxed out and plateaus at 3.
            step_interval (int, optional): how often should the annealer increment a step (counted in number of
                iterations, *not* epochs); this value is related to `T1` and `T2`.
            device (str, None, torch.device): device type.
            *args (Any, optional): arguments to be passed into the optimiser.
            **kwargs (Any, optional): keyword arguments to be passed into the optimiser.
        """
        # essentials
        assert hasattr(torch.optim, optimiser)
        assert patience is None or patience > 0
        assert not reload_best or patience is not None
        self._model = model
        self._lloss = labelled_loss
        self._uloss = unlabelled_loss
        self._optim = optimiser

        # hparams
        self._use_soft_labels = use_soft_labels
        self._patience = patience
        self._reload_best = reload_best
        self._track_pl_metrics = track_pl_metrics
        self._T1 = T1
        self._T2 = T2
        self._step_interval = step_interval

        self._device = device

        # optimiser args
        self._args = args
        self._kwargs = kwargs

    def fit(self,
            train_loader: torchdata.DataLoader,
            pool_loader: torchdata.DataLoader,
            val_loader: Optional[torchdata.DataLoader] = None,
            epochs: Union[int, Sequence[int]] = 1) -> Dict[str, list]:
        if self._track_pl_metrics is not None and \
                (not isinstance(pool_loader.dataset, UnlabelledDataset)
                 or not pool_loader.dataset.debug):
            raise ValueError(
                f"If track_pl_metrics is provided, then the dataset in pool_loader "
                f"must be of the type UnlabelledDataset with debug on."
            )

        if self._patience and val_loader is None:
            raise ValueError("If patience is specified, then val_loader must be provided in .fit().")

        if isinstance(epochs, int):
            epochs = (epochs, epochs)
        assert len(epochs) == 2
        epoch1, epoch2 = epochs[0], epochs[1]
        # stage 1
        supervised_trainer = Trainer(
            self._model, self._lloss, self._optim,
            patience=self._patience,
            reload_best=self._reload_best,
            device=self._device, *self._args, **self._kwargs
        )
        # until convergence
        supervised_history = supervised_trainer.fit(
            train_loader, val_loader, epochs=epoch1,
        )

        if self._track_pl_metrics is not None:
            for k, v in _get_pl_metrics(
                    self._model,
                    PLTracker(entropy_fn=self._track_pl_metrics, device=self._device),
                    train_loader).items():
                assert k not in supervised_history
                supervised_history[k] = v

        # stage 2
        pbar = ProgressBar()
        history = defaultdict(list)

        train_evaluator = create_supervised_evaluator(
            self._model, metrics={'acc': Accuracy(), 'loss': Loss(self._lloss)},
            device=self._device
        )
        val_evaluator = create_supervised_evaluator(
            self._model, metrics={'acc': Accuracy(), 'loss': Loss(self._lloss)},
            device=self._device
        )

        def _log_metrics(engine: Engine):
            # engine = ssl engine with `pl_tracker`

            # train loader - save to history and print metrics
            metrics = train_evaluator.run(train_loader).metrics
            history[f"train_acc"].append(metrics['acc'])
            history[f"train_loss"].append(metrics['loss'])
            pbar.log_message(
                f"epoch {engine.state.epoch}/{engine.state.max_epochs}\n"
                f"\ttrain acc = {metrics['acc']}, train loss = {metrics['loss']}"
            )

            # log PL tracking metrics
            if self._track_pl_metrics is not None:
                history["pl_acc"].append(engine.state.pl_tracker['pl_acc'])
                history["confidence"].append(engine.state.pl_tracker['confidence'])
                history["entropy"].append(engine.state.pl_tracker['entropy'])

            if val_loader is None:
                return  # job done

            # val loader - save to history and print metrics. Also, add handlers to
            # evaluator (e.g. early stopping, model checkpointing that depend on val_acc)
            metrics = val_evaluator.run(val_loader).metrics

            history[f"val_acc"].append(metrics['acc'])
            history[f"val_loss"].append(metrics['loss'])
            pbar.log_message(
                f"\tval acc = {metrics['acc']}, val loss = {metrics['loss']}"
            )

        ssl_trainer = create_semisupervised_trainer(
            model=self._model,
            optimiser=getattr(torch.optim, self._optim)(
                self._model.parameters(), *self._args, **self._kwargs
            ), lloss_fn=self._lloss, uloss_fn=self._uloss,
            annealer=Annealer(step=1, T1=self._T1, T2=self._T2, step_interval=self._step_interval),
            train_iterable=WraparoundLoader(train_loader),
            tracker=PLTracker(entropy_fn=self._track_pl_metrics, device=self._device),
            use_soft_labels=self._use_soft_labels,
            device=self._device
        )
        if val_loader is not None and self._patience:
            es = EarlyStopper(self._model, patience=self._patience, trainer=ssl_trainer, key='acc', mode='max')
            es.attach(val_evaluator)
        ssl_trainer.add_event_handler(Events.EPOCH_COMPLETED, _log_metrics)
        pbar.attach(ssl_trainer, output_transform=lambda x: {'loss': x})

        ssl_trainer.run(
            pool_loader, max_epochs=epoch2,
            seed=np.random.randint(0, 1e6)
        )
        if val_loader is not None and self._patience and self._reload_best:
            es.reload_best()

        for k, v in supervised_history.items():
            v.extend(history[k])

        # return combined history
        return supervised_history

    def evaluate(self, data_loader: torchdata.DataLoader) -> dict:
        evaluator = create_supervised_evaluator(
            self._model, metrics={'acc': Accuracy(), 'loss': Loss(self._lloss)},
            device=self._device
        )
        return evaluator.run(data_loader).metrics


def _get_pl_metrics(model: nn.Module,
                    tracker: PLTracker,
                    loader: torchdata.DataLoader):
    with torch.no_grad():
        model.eval()
        for batch in loader:
            x = tracker.process_batch(batch)
            tracker.record_predictions(model(x))
    metrics = {
        'pl_acc': [tracker._correct / tracker._total],
        'confidence': [tracker._confidence],
        'entropy': [tracker._entropy],
    }
    tracker._reset()
    return metrics
