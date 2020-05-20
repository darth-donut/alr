import torch
from ignite.engine import Engine, Events
from torch import nn
import torch.nn.functional as F
from typing import Optional
from ignite.engine import Engine, Events, \
    create_supervised_evaluator
from ignite.metrics import Loss, Accuracy
import torch.utils.data as torchdata
from ignite.contrib.handlers import ProgressBar
import numpy as np

from collections import defaultdict
from typing import Optional, Dict, Sequence, Union

from alr.data import UnlabelledDataset
from alr.training import Trainer
from alr.utils._type_aliases import _DeviceType, _Loss_fn
from alr.utils import _map_device
from alr.training.utils import EarlyStopper

r"""
todo(harry):
    1. acc/quality BEFORE training with pseudo labels: currently only after first epoch.
    2. thresholding capabilities
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
        # todo(harry): remove me
        print("stepped annealer")
        self._step += 1

    @property
    def weight(self):
        if self._step < self._T1:
            return 0
        elif self._step > self._T2:
            # todo(harry): remove me
            print("Passed threshold")
            return self._alpha
        else:
            return ((self._step - self._T1) / (self._T2 - self._T1)) * self._alpha

    def attach(self, engine: Engine):
        engine.add_event_handler(
            Events.ITERATION_COMPLETED(every=self._step_interval),
            self.step
        )


class PLTracker:
    # todo(harry): test this class
    def __init__(self, active: bool, device: _DeviceType = None):
        self._device = device
        self._last_y = None
        self._correct = self._total = 0
        self._confidence = []
        self._track = active

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

    def attach(self, engine: Engine):
        if self._track:
            engine.add_event_handler(Events.EPOCH_STARTED, self.reset)
            engine.add_event_handler(Events.EPOCH_COMPLETED, self.write_accuracy)
            engine.add_event_handler(Events.EPOCH_COMPLETED, self.write_confidence)

    def write_accuracy(self, engine):
        engine.state.pl_tracker['acc'] = self._correct / self._total

    def write_confidence(self, engine):
        engine.state.pl_tracker['confidence'] = self._confidence

    def reset(self, engine):
        self._correct = self._total = 0
        self._confidence = []
        engine.state.pl_tracker = {}


def soft_nll_loss(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # -1/N * sum_y p(y)log[p(y)]
    res = -(target.exp() * preds).sum(dim=1).mean()
    assert torch.isfinite(res)
    return res


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # -1/N * sum_y p(y)log[p(y)]
    res = -(F.softmax(target, dim=-1) * F.log_softmax(logits, dim=-1)).sum(dim=1).mean()
    assert torch.isfinite(res)
    return res


def create_semisupervised_trainer(model: nn.Module, optimiser,
                                  lloss_fn: _Loss_fn, uloss_fn: _Loss_fn,
                                  annealer: Annealer,
                                  train_iterable: WraparoundLoader,
                                  track: bool,
                                  use_soft_labels: bool, device: _DeviceType):
    tracker = PLTracker(active=track, device=device)

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
                 track_pl_metrics: Optional[bool] = False,
                 T1: Optional[int] = 0,
                 T2: Optional[int] = 200,
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
            track_pl_metrics (bool, optional): if `True`, then the quality of pseudo-labels and uncertainty will be
                tracked throughout the training epochs. :meth:`fit` will also return additional keys representing
                these aforementioned metrics.
            T1 (int, optional): when the weight coefficient starts kicking in.
            T2 (int, optional): when the weight coefficient starts plateauing.
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
        if self._track_pl_metrics and \
                (not isinstance(pool_loader.dataset, UnlabelledDataset)
                 or not pool_loader.dataset.debug):
            raise ValueError(
                f"If track_pl_metrics is True, then the dataset in pool_loader "
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

        # todo(harry): record into supervised_history, the accuracy and uncertainty of pseudo-labels at this stage
        #   because PLTracker only does it after the model has been trained for at least one epoch.

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
            if self._track_pl_metrics:
                history["pl_acc"].append(engine.state.pl_tracker['acc'])
                history["confidence"].append(engine.state.pl_tracker['confidence'])

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
            track=self._track_pl_metrics, use_soft_labels=self._use_soft_labels,
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

        if self._track_pl_metrics:
            supervised_history["pl_acc"] = history["pl_acc"]
            supervised_history["confidence"] = history["confidence"]

        # return combined history
        return supervised_history

    def evaluate(self, data_loader: torchdata.DataLoader) -> dict:
        evaluator = create_supervised_evaluator(
            self._model, metrics={'acc': Accuracy(), 'loss': Loss(self._lloss)},
            device=self._device
        )
        return evaluator.run(data_loader).metrics
