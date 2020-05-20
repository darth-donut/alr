import torch
from ignite.engine import Engine, Events, \
    create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Loss, Accuracy
import torch.utils.data as torchdata
from ignite.contrib.handlers import ProgressBar
import numpy as np

from collections import defaultdict
from typing import Optional, Dict

from alr.utils import _DeviceType
from alr.training.utils import EarlyStopper


class Trainer:
    def __init__(self, model, loss, optimiser: str, device: _DeviceType, *args, **kwargs):
        self._loss = loss
        self._optim = getattr(torch.optim, optimiser)(model.parameters(), *args, **kwargs)
        self._device = device
        self._model = model

    def fit(self,
            train_loader: torchdata.DataLoader,
            val_loader: Optional[torchdata.DataLoader] = None,
            epochs: Optional[int] = 1,
            patience: Optional[int] = None,
            reload_best: Optional[bool] = False) -> Dict[str, list]:
        assert patience is None or patience > 0
        assert patience is None or val_loader is not None
        assert not reload_best or val_loader is not None

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
        if val_loader is not None and patience:
            es = EarlyStopper(self._model, patience, trainer, key='acc', mode='max')
            es.attach(val_evaluator)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, _log_metrics)
        # pytorch-ignite v0.3.0's explicit seed parameter
        trainer.run(
            train_loader, max_epochs=epochs,
            seed=np.random.randint(0, 1e6)
        )
        if val_loader is not None and patience and reload_best:
            es.reload_best()
        return history

    def evaluate(self, data_loader: torchdata.DataLoader) -> dict:
        evaluator = create_supervised_evaluator(
            self._model, metrics={'acc': Accuracy(), 'loss': Loss(self._loss)},
            device=self._device
        )
        return evaluator.run(data_loader).metrics
