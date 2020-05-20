import torch
from torch import nn
from ignite.engine import Engine, Events, \
    create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Loss, Accuracy
import torch.utils.data as torchdata
from ignite.contrib.handlers import ProgressBar
import numpy as np

from collections import defaultdict
from typing import Optional, Dict

from alr.utils._type_aliases import _DeviceType, _Loss_fn
from alr.training.utils import EarlyStopper


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 loss: _Loss_fn,
                 optimiser: str,
                 patience: Optional[int] = None,
                 reload_best: Optional[bool] = False,
                 device: _DeviceType = None,
                 *args,
                 **kwargs):
        r"""

        Args:
            model (torch.nn.Module): module object
            loss (Callable): should be a function `fn` that takes `preds` and `targets` and returns
                a singleton tensor with the loss value: `loss = fn(preds, targets)`. E.g. F.nll_loss.
            optimiser (str): a string that corresponds to the type of optimiser to use. Must be an optimiser from
                `torch.optim` (case sensitive). E.g. 'Adam'.
            patience (int, optional): if not `None`, then validation accuracy will be used to determine when to stop.
            reload_best (bool, optional): patience must be non-`None` if this is set to `True`: reloads the best model
                according to validation accuracy at the end of training.
            device (str, None, torch.device): device type.
            *args (Any, optional): arguments to be passed into the optimiser.
            **kwargs (Any, optional): keyword arguments to be passed into the optimiser.
        """
        self._loss = loss
        self._optim = getattr(torch.optim, optimiser)(model.parameters(), *args, **kwargs)
        self._patience = patience
        self._reload_best = reload_best
        assert patience is None or patience > 0
        assert not reload_best or patience is not None
        self._device = device
        self._model = model

    def fit(self,
            train_loader: torchdata.DataLoader,
            val_loader: Optional[torchdata.DataLoader] = None,
            epochs: Optional[int] = 1) -> Dict[str, list]:
        if self._patience and val_loader is None:
            raise ValueError("If patience is specified, then val_loader must be provided in .fit().")

        pbar = ProgressBar()
        history = defaultdict(list)

        train_evaluator = create_supervised_evaluator(
            self._model, metrics={'acc': Accuracy(), 'loss': Loss(self._loss)},
            device=self._device
        )
        val_evaluator = create_supervised_evaluator(
            self._model, metrics={'acc': Accuracy(), 'loss': Loss(self._loss)},
            device=self._device
        )

        def _log_metrics(engine: Engine):
            # train loader - save to history and print metrics
            metrics = train_evaluator.run(train_loader).metrics
            history[f"train_acc"].append(metrics['acc'])
            history[f"train_loss"].append(metrics['loss'])
            pbar.log_message(
                f"epoch {engine.state.epoch}/{engine.state.max_epochs}\n"
                f"\ttrain acc = {metrics['acc']}, train loss = {metrics['loss']}"
            )

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

        trainer = create_supervised_trainer(
            self._model, optimizer=self._optim,
            loss_fn=self._loss, device=self._device
        )
        pbar.attach(trainer, output_transform=lambda x: {'loss': x})
        if val_loader is not None and self._patience:
            es = EarlyStopper(self._model, self._patience, trainer, key='acc', mode='max')
            es.attach(val_evaluator)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, _log_metrics)
        # pytorch-ignite v0.3.0's explicit seed parameter
        trainer.run(
            train_loader, max_epochs=epochs,
            seed=np.random.randint(0, 1e6)
        )
        if val_loader is not None and self._patience and self._reload_best:
            es.reload_best()
        return history

    def evaluate(self, data_loader: torchdata.DataLoader) -> dict:
        evaluator = create_supervised_evaluator(
            self._model, metrics={'acc': Accuracy(), 'loss': Loss(self._loss)},
            device=self._device
        )
        return evaluator.run(data_loader).metrics
