import torch
from torch import nn
from ignite.handlers import EarlyStopping, ModelCheckpoint, global_step_from_engine
import tempfile
from pathlib import Path
from ignite.engine import Engine, Events

from typing import Optional


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
