import torch
import tempfile
from pathlib import Path
from torch import nn
from ignite.engine import Engine, Events,\
    create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Loss, Accuracy
import torch.utils.data as torchdata
from ignite.handlers import EarlyStopping, Checkpoint, global_step_from_engine, DiskSaver
from ignite.contrib.handlers import ProgressBar
import numpy as np

from collections import defaultdict
from typing import Optional, Callable, Dict, Sequence


r"""
Checklist:
1. Pseudo-labelling trainer
"""


class Trainer:
    def __init__(self, loss, optimiser, device):
        self._loss = loss
        self._optim = optimiser
        self._device = device

    def fit(self,
            model: nn.Module,
            train_loader: torchdata.DataLoader,
            val_loader: Optional[torchdata.DataLoader] = None,
            epochs: Optional[int] = 1,
            early_stopping: Optional[int] = None,
            reload_best: Optional[bool] = False) -> Dict[str, list]:
        assert early_stopping is None or early_stopping > 0
        assert early_stopping is None or val_loader is not None
        assert not reload_best or val_loader is not None

        pbar = ProgressBar()
        history = defaultdict(list)

        def _epoch_end_callback(engine: Engine, val_handlers={}):
            # train loader - save to history and print metrics
            metrics = self.evaluate(model, train_loader)
            history[f"train_acc"].append(metrics['acc'])
            history[f"train_loss"].append(metrics['loss'])
            pbar.log_message(
                f"epoch {engine.state.epoch}/{engine.state.max_epochs}, "
                f"train acc = {metrics['acc']}, train loss = {metrics['loss']}"
            )

            if val_loader is None:
                return  # job done

            # val loader - save to history and print metrics. Also, add handlers to
            # evaluator (e.g. early stopping, model checkpointing that depend on val_acc)
            evaluator = create_supervised_evaluator(
                model, metrics={'acc': Accuracy(), 'loss': Loss(self._loss)},
                device=self._device
            )
            for e, hs in val_handlers.items():
                for h in hs:
                    evaluator.add_event_handler(e, h)

            metrics = evaluator.run(val_loader).metrics

            history[f"val_acc"].append(metrics['acc'])
            history[f"val_loss"].append(metrics['loss'])
            pbar.log_message(
                f"epoch {engine.state.epoch}/{engine.state.max_epochs}, "
                f"val acc = {metrics['acc']}, val loss = {metrics['loss']}"
            )

        chpt_handler = None
        trainer = create_supervised_trainer(
            model, optimizer=self._optim,
            loss_fn=self._loss, device=self._device
        )
        pbar.attach(trainer, output_transform=lambda x: {'loss': x})

        with tempfile.TemporaryDirectory() as tmpdir:
            handlers = {}
            if val_loader is not None and early_stopping:
                def get_val_accuracy(engine):
                    return engine.state.metrics['acc']
                es_handler = EarlyStopping(
                    patience=early_stopping,
                    score_function=get_val_accuracy,
                    trainer=trainer
                )
                chpt_handler = Checkpoint(
                    {'model': model}, DiskSaver(str(tmpdir), create_dir=False),
                    n_saved=1, filename_prefix='best', score_function=get_val_accuracy,
                    score_name="val_acc", global_step_transform=global_step_from_engine(trainer)
                )
                handlers[Events.COMPLETED] = [es_handler, chpt_handler]

            trainer.add_event_handler(Events.EPOCH_COMPLETED, _epoch_end_callback, handlers)

            # pytorch-ignite v0.3.0's explicit seed parameter
            trainer.run(
                train_loader, max_epochs=epochs,
                seed=np.random.randint(0, 1e6)
            )

            if reload_best and chpt_handler is not None:
                model.load_state_dict(
                    torch.load(
                        Path(str(tmpdir)) / str(chpt_handler.last_checkpoint)
                    ),
                    strict=True
                )
            return history

    def evaluate(self, model: nn.Module, data_loader: torchdata.DataLoader) -> dict:
        evaluator = create_supervised_evaluator(
            model, metrics={'acc': Accuracy(), 'loss': Loss(self._loss)},
            device=self._device
        )
        return evaluator.run(data_loader).metrics
