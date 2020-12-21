import pickle

import numpy as np
import torch
from torch import nn
from ignite.handlers import EarlyStopping, ModelCheckpoint, global_step_from_engine
import tempfile
from pathlib import Path
from ignite.engine import Engine, Events

from typing import Optional, Callable

from alr.utils.math import entropy


class EarlyStopper:
    def __init__(
        self,
        model: nn.Module,
        patience: int,
        trainer: Engine,
        key: Optional[str] = "acc",
        mode: Optional[str] = "max",
    ):
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

        self._reload_called = False

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
            trainer=self._trainer,
        )
        self._chkpt_handler = ModelCheckpoint(
            str(self._tmpdir.name),
            filename_prefix="best",
            n_saved=1,
            create_dir=False,
            score_function=self._score_function,
            score_name=f"val_{self._key}",
            global_step_transform=global_step_from_engine(self._trainer),
        )
        engine.add_event_handler(Events.COMPLETED, es_handler)
        engine.add_event_handler(
            Events.COMPLETED, self._chkpt_handler, {"model": self._model}
        )

    def _score_function(self, engine):
        return engine.state.metrics[self._key] * self._mode

    def reload_best(self):
        if self._reload_called:
            raise RuntimeError("Cannot reload more than once.")
        if self._chkpt_handler is None or self._chkpt_handler.last_checkpoint is None:
            raise RuntimeError(
                "Cannot reload model until it has been trained for at least one epoch."
            )
        self._model.load_state_dict(
            torch.load(str(self._chkpt_handler.last_checkpoint)), strict=True
        )
        self._tmpdir.cleanup()
        self._reload_called = True


class PLPredictionSaver:
    def __init__(
        self,
        log_dir: str,
        compact: Optional[bool] = True,
        pred_transform: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = lambda x: x.exp(),
        onehot_target: Optional[bool] = False,
    ):
        r"""

        Args:
            log_dir (): duh
            compact (): save what you need (compact) instead of saving all predictions (huge files)
            pred_transform (): typically used to exponentiate model's output predictions
            onehot_target (): set to True if the target label is a distribution (i.e.
                argmax should be called on it to get the class); leave as false if targets are
                ints.
        """
        self._output_transform = lambda x: x
        self._preds = []
        self._targets = []
        if log_dir:
            self._log_dir = Path(log_dir)
            self._log_dir.mkdir(parents=True, exist_ok=True)
        self._other_engine = None
        self._compact = compact
        self._pred_transform = pred_transform
        self._onehot_target = onehot_target

    def attach(
        self, engine: Engine, output_transform: Callable[..., tuple] = lambda x: x
    ):
        self._output_transform = output_transform
        self._other_engine = engine
        engine.add_event_handler(Events.EPOCH_STARTED, self._reset)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self._flush)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._parse)

    def _parse(self, engine: Engine):
        pred, target = self._output_transform(engine.state.output)
        self._preds.append(pred.detach().cpu())
        if self._onehot_target:
            self._targets.append(target.detach().cpu().argmax(dim=-1))
        else:
            self._targets.append(target.detach().cpu())

    def _flush(self, _):
        preds_N_C = self._pred_transform(torch.cat(self._preds, dim=0)).numpy()
        assert preds_N_C.ndim == 2
        targets_N = torch.cat(self._targets, dim=0).numpy()
        assert targets_N.ndim == 1 and targets_N.shape[0] == preds_N_C.shape[0]
        if self._compact:
            acc = _accuracy(preds_N_C, targets_N)
            payload = {
                "ece": _expected_calibration_error(preds_N_C, targets_N),
                "conf-thresh": _confidence_threshold(preds_N_C),
                "entropy": _entropy(preds_N_C),
                "accuracy": acc.mean(),
                "per-instance-accuracy": acc,
            }
        else:
            payload = {
                "preds": preds_N_C,
                "targets": targets_N,
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


def _confidence_threshold(preds_N_C: np.ndarray):
    x = np.linspace(0, 1, num=100)
    y = np.empty(shape=x.shape[0])
    for idx, thresh in enumerate(x):
        y[idx] = np.mean(np.max(preds_N_C, axis=-1) >= thresh)
    return x, y


def _entropy(preds_N_C: np.ndarray):
    ent_N = entropy(torch.from_numpy(preds_N_C), mode="softmax").numpy().sum(axis=-1)
    return ent_N


def _accuracy(preds_N_C, targets_N):
    return np.equal(preds_N_C.argmax(axis=-1), targets_N)


def _expected_calibration_error(preds_N_C: np.ndarray, targets_N: np.ndarray):
    # https://arxiv.org/pdf/1706.04599.pdf
    width = 0.1

    N = preds_N_C.shape[0]
    bins = np.arange(0, 1 + width, width)
    acc = np.zeros(shape=(len(bins) - 1))
    counts = np.zeros_like(acc)
    conf = np.zeros_like(acc)

    class_N = preds_N_C.argmax(axis=-1)
    probs_N = np.max(preds_N_C, axis=-1)

    for idx, b in enumerate(bins[1:]):
        low, high = bins[idx], b
        mask = (low < probs_N) & (probs_N <= high)
        if mask.any():
            acc[idx] = np.equal(class_N[mask], targets_N[mask]).mean()
            counts[idx] = mask.sum()
            # average confidence in bin (low, high]
            conf[idx] = np.mean(probs_N[mask])

    res = np.abs(acc - conf) * counts
    assert res.shape == (len(bins) - 1,)
    assert np.isfinite(res).all()

    return bins, acc, counts, conf, np.sum(res) / N


class PerformanceTracker:
    def __init__(self, model: nn.Module, patience: int):
        self.model = model
        self.patience = patience
        self._original_patience = patience
        self.last_acc = None
        self._temp_dir = tempfile.TemporaryDirectory()
        self._model_filename = (
            Path(str(self._temp_dir.name)).absolute() / f"{id(self)}.pt"
        )
        self._reloaded = False

    def reset(self):
        self.patience = self._original_patience

    def step(self, acc):
        if self.last_acc is None or acc > self.last_acc:
            self.reset()
            if self._model_filename.exists():
                # 2 am paranoia: make sure old model weight is overridden
                self._model_filename.unlink()
            torch.save(self.model.state_dict(), str(self._model_filename))
            self.last_acc = acc
        else:
            self.patience -= 1

    @property
    def done(self) -> bool:
        return self.patience <= 0

    @property
    def reloaded(self) -> bool:
        return self._reloaded

    def reload_best(self):
        if self.last_acc is None:
            raise RuntimeError(
                "Cannot reload model until step is called at least once."
            )
        self.model.load_state_dict(torch.load(self._model_filename), strict=True)
        self._temp_dir.cleanup()
        self._reloaded = True
