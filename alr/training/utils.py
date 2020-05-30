import pickle

import numpy as np
import torch
from torch import nn
from ignite.handlers import EarlyStopping, ModelCheckpoint, global_step_from_engine
import tempfile
from pathlib import Path
from ignite.engine import Engine, Events

from typing import Optional, Callable


class EarlyStopper:
    def __init__(self, model: nn.Module,
                 patience: int, trainer: Engine,
                 key: Optional[str] = 'acc',
                 mode: Optional[str] = 'max'):
        self._model = model
        self._patience = patience
        self._trainer = trainer
        self._key = key
        self._tmpdir = tempfile.TemporaryDirectory()

        mode = mode.lower()
        assert mode in {"min", "max"}
        self._mode = -1 if mode == "min" else 1

        assert Path(str(self._tmpdir.name)).is_dir()
        assert not list(Path(str(self._tmpdir.name)).rglob("*.pth"))
        self._chkpt_handler = None

    def attach(self, engine: Engine):
        r"""
        Attach an early stopper to engine that will terminate the provided trainer
        when the predetermined metric does not improve for `patience` epochs.

        Args:
            engine (ignite.engine.Engine): this is expected to be a validation
                evaluator. The `key` metric will be extracted and the best will
                be used.

        Returns:
            NoneType: None
        """
        es_handler = EarlyStopping(
            patience=self._patience,
            score_function=self._score_function,
            trainer=self._trainer
        )
        self._chkpt_handler = ModelCheckpoint(
            str(self._tmpdir.name), filename_prefix='best', n_saved=1, create_dir=False,
            score_function=self._score_function, score_name=f'val_{self._key}',
            global_step_transform=global_step_from_engine(self._trainer)
        )
        engine.add_event_handler(Events.COMPLETED, es_handler)
        engine.add_event_handler(Events.COMPLETED, self._chkpt_handler, {'model': self._model})

    def _score_function(self, engine):
        return engine.state.metrics[self._key] * self._mode

    def reload_best(self):
        if self._chkpt_handler is None or self._chkpt_handler.last_checkpoint is None:
            raise RuntimeError("Cannot reload model until it has been trained for at least one epoch.")
        self._model.load_state_dict(
            torch.load(str(self._chkpt_handler.last_checkpoint)),
            strict=True
        )


class PLPredictionSaver:
    def __init__(self,
                 log_dir: Optional[str] = None):
        self._output_transform = lambda x: x
        self._preds = []
        self._targets = []
        if log_dir:
            self._log_dir = Path(log_dir)
            self._log_dir.mkdir(parents=True, exist_ok=True)
        self._other_engine = None

    def attach(self,
               engine: Engine,
               output_transform: Callable[..., tuple] = lambda x: x):
        self._output_transform = output_transform
        self._other_engine = engine
        engine.add_event_handler(Events.EPOCH_STARTED, self._reset)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self._flush)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._parse)

    def _parse(self, engine: Engine):
        pred, target = self._output_transform(engine.state.output)
        self._preds.append(pred.cpu().numpy())
        self._targets.append(target.cpu().numpy())

    def _flush(self, _):
        payload = {
            'preds': np.concatenate(self._preds, axis=0),
            'targets': np.concatenate(self._targets, axis=0),
        }
        epoch = self._other_engine.state.epoch
        fname = self._log_dir / f"{str(epoch)}_pl_predictions.pkl"
        assert not fname.exists(), "You've done goofed"
        with open(fname, "wb") as fp:
            pickle.dump(payload, fp)

    def _reset(self, _):
        self._preds = []
        self._targets = []

    def global_step_from_engine(self, engine: Engine):
        self._other_engine = engine

