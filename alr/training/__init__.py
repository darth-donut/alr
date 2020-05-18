import torch
from torch import nn
from ignite.engine import Engine, Events,\
    create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Loss, Accuracy
import torch.utils.data as torchdata
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers import ProgressBar
import numpy as np

from collections import defaultdict
from typing import Optional, Callable, Dict, Sequence


r"""
Checklist:
1. basic trainer + evaluator function DONE
2. early stopping + validation set DONE
3. return best model after early stopping has kicked in (or not, just return best)
4. Pseudo-labelling trainer
"""


class Trainer:
    def __init__(self, loss, optimiser, device):
        self._loss = loss
        self._optim = optimiser
        self._device = device

    def fit(self,
            model: nn.Module,
            train_loader: torchdata.DataLoader,
            val_loader: Optional[torchdata.DataLoader],
            epochs: Optional[int] = 1,
            early_stopping: Optional[int] = None) -> Dict[str, list]:
        assert early_stopping is None or early_stopping > 0

        trainer = create_supervised_trainer(
            model, optimizer=self._optim,
            loss_fn=self._loss, device=self._device
        )

        pbar = ProgressBar()
        history = defaultdict(list)
        pbar.attach(trainer, output_transform=lambda x: {'loss': x})

        def log_metrics(engine: Engine, data_loader: torchdata.DataLoader, dtype: str,
                        save_to: Optional[Dict[str, list]] = None,
                        handlers: Optional[Dict[Events, Callable]] = {}):
            metrics = self.evaluate(model, data_loader, handlers)
            if save_to is not None:
                save_to[f"{dtype}_acc"].append(metrics['acc'])
                save_to[f"{dtype}_loss"].append(metrics['loss'])
            pbar.log_message(
                # f"Iter {engine.state.iteration}, "
                f"epoch {engine.state.epoch}/{engine.state.max_epochs}, "
                f"{dtype} acc = {metrics['acc']}, {dtype} loss = {metrics['loss']}"
            )

        # trainer.add_event_handler(
        #     Events.ITERATION_COMPLETED(every=200),
        #     log_metrics, data_loader=train_loader, dtype="train"
        # )
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            log_metrics, data_loader=train_loader,
            dtype="train", save_to=history
        )

        if val_loader is not None:
            handler = {}
            if early_stopping:
                def get_val_accuracy(engine):
                    return engine.state.metrics['acc']
                es_handler = EarlyStopping(
                    patience=early_stopping,
                    score_function=get_val_accuracy,
                    trainer=trainer
                )
                handler[Events.COMPLETED] = es_handler

            # trainer.add_event_handler(
            #     Events.ITERATION_COMPLETED(every=200),
            #     log_metrics, data_loader=val_loader, dtype="val"
            # )
            trainer.add_event_handler(
                Events.EPOCH_COMPLETED,
                log_metrics, data_loader=val_loader,
                dtype="val", save_to=history, handlers=handler,
            )

        # pytorch-ignite v0.3.0's explicit seed parameter
        trainer.run(
            train_loader, max_epochs=epochs,
            seed=np.random.randint(0, 1e6)
        )
        return history

    def evaluate(self, model: nn.Module, data_loader: torchdata.DataLoader,
                 handlers: Optional[Dict[Events, Callable]] = {}) -> dict:
        evaluator = create_supervised_evaluator(
            model, metrics={'acc': Accuracy(), 'loss': Loss(self._loss)},
            device=self._device
        )
        for e, h in handlers.items():
            evaluator.add_event_handler(e, h)
        return evaluator.run(data_loader).metrics
