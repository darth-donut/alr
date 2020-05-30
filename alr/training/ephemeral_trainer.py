from collections import defaultdict

from alr import ALRModel
from alr.data import RelabelDataset, PseudoLabelDataset, UnlabelledDataset
from alr.training.utils import EarlyStopper, PLPredictionSaver
from alr.utils._type_aliases import _DeviceType, _Loss_fn
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.utils.data as torchdata
from ignite.engine import create_supervised_evaluator, Events, Engine
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers import ProgressBar
from alr.training import Trainer
from alr.training.samplers import RandomFixedLengthSampler


class PseudoLabelManager:
    def __init__(self,
                 pool: UnlabelledDataset,
                 model: nn.Module,
                 threshold: float,
                 log_dir: Optional[str] = None,
                 device: _DeviceType = None,
                 **kwargs):
        bs = kwargs.pop('batch_size', 1024)
        shuffle = kwargs.pop('shuffle', False)
        assert not shuffle
        self._pool = pool
        self._loader = torchdata.DataLoader(pool, batch_size=bs, shuffle=shuffle, **kwargs)
        self._model = model
        self._log_dir = log_dir
        self._device = device
        self._threshold = threshold
        self.acquired_sizes = []

    def attach(self, engine: Engine):
        engine.add_event_handler(Events.STARTED, self._initialise)
        # could also be EPOCH_COMPLETED since there's only one iteration in each epoch
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._load_labels)

    def _load_labels(self, engine: Engine):
        evaluator = create_supervised_evaluator(self._model, metrics=None, device=self._device)
        plc = PseudoLabelCollector(
            self._threshold, log_dir=self._log_dir,
        )
        plc.attach(evaluator, batch_size=self._loader.batch_size)
        plc.global_step_from_engine(engine)
        evaluator.run(self._loader)
        indices, pseudo_labels = \
            evaluator.state.pl_indices.cpu().numpy(), \
            evaluator.state.pl_plabs.cpu().numpy()
        self.acquired_sizes.append(indices.shape[0])
        if indices.shape[0]:
            confident_points = torchdata.Subset(self._pool, indices)
            if self._pool.debug:
                # pool returns target labels too
                engine.state.pseudo_labelled_dataset = RelabelDataset(confident_points, pseudo_labels)
            else:
                engine.state.pseudo_labelled_dataset = PseudoLabelDataset(confident_points, pseudo_labels)
        else:
            engine.state.pseudo_labelled_dataset = None

    @staticmethod
    def _initialise(engine: Engine):
        engine.state.pseudo_labelled_dataset = None


class PseudoLabelCollector:
    def __init__(self,
                 threshold: float,
                 log_dir: Optional[str] = None,
                 pred_transform=lambda x: x.exp()):
        self._indices = []
        self._plabs = []
        self._pred_transform = pred_transform
        self._thresh = threshold
        self._targets = []
        self._preds = []
        if log_dir:
            self._saver = PLPredictionSaver(log_dir)
        else:
            self._saver = None
        self._batch_size = None

    def _parse(self, engine: Engine):
        preds, targets = engine.state.output
        # state.iteration starts with 1
        iteration = engine.state.iteration - 1
        offset = iteration * self._batch_size
        with torch.no_grad():
            preds = self._pred_transform(preds)
            preds_max, plabs = torch.max(preds, dim=-1)
            mask = torch.nonzero(preds_max >= self._thresh).flatten()
            if mask.shape[0]:
                # plabs = [N,]
                self._plabs.append(plabs[mask])
                self._indices.append(mask + offset)

    def _flush(self, engine: Engine):
        if self._indices and self._plabs:
            engine.state.pl_indices = torch.cat(self._indices)
            engine.state.pl_plabs = torch.cat(self._plabs)
        else:
            engine.state.pl_indices = torch.Tensor([])
            engine.state.pl_plabs = torch.Tensor([])
        self._indices = []
        self._plabs = []

    def attach(self, engine: Engine, batch_size: int, output_transform=lambda x: x):
        r"""

        Args:
            engine (Engine): ignite engine object
            batch_size (int): engine's batch size
            output_transform (Callable): if engine.state.output is not (preds, target),
                then output_transform should return aforementioned tuple.

        Returns:
            NoneType: None
        """
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._parse)
        engine.add_event_handler(Events.COMPLETED, self._flush)
        self._batch_size = batch_size
        if self._saver:
            self._saver.attach(engine, output_transform=output_transform)

    def global_step_from_engine(self, engine: Engine):
        if self._saver:
            self._saver.global_step_from_engine(engine)


def _update_dataloader(loader: torchdata.DataLoader,
                       dataset: torchdata.Dataset,
                       sampler: Optional[torchdata.Sampler] = None):
    # attributes that usually go in dataloader's constructor
    attrs = [k for k in loader.__dict__.keys() if not k.startswith('_')]
    drop = ['dataset', 'sampler', 'batch_sampler', 'dataset_kind']
    kwargs = {k: getattr(loader, k) for k in attrs if k not in drop}
    if not isinstance(loader.sampler, (torchdata.SequentialSampler, torchdata.RandomSampler, RandomFixedLengthSampler)):
        raise ValueError(f"Only sequential, random, and random fixed length samplers "
                         f"are supported in _update_dataloader")
    kwargs['dataset'] = dataset
    # Sequential and Random will be automatically determined if sampler is None (depending on shuffle)
    kwargs['sampler'] = sampler
    return torchdata.DataLoader(**kwargs)


def create_pseudo_label_trainer(model: ALRModel, loss: _Loss_fn, optimiser: str,
                                train_loader: torchdata.DataLoader, val_loader: torchdata.DataLoader,
                                pseudo_label_manager: PseudoLabelManager,
                                rfls_len: Optional[int] = None,
                                patience: Optional[int] = None, reload_best: Optional[bool] = None,
                                epochs: Optional[int] = 1, device: _DeviceType = None,
                                *args, **kwargs):
    def _step(engine: Engine, _):
        # always reset weights
        model.reset_weights()

        # update loader accordingly: if pld is not none, concatenate them
        new_loader = train_loader
        pld = engine.state.pseudo_labelled_dataset
        if pld is not None:
            train_ds = torchdata.ConcatDataset((train_loader.dataset, pld))
            # update dataloader's dataset attribute
            if rfls_len:
                new_loader = _update_dataloader(
                    train_loader, train_ds,
                    RandomFixedLengthSampler(train_ds, length=rfls_len, shuffle=True)
                )
            else:
                new_loader = _update_dataloader(train_loader, train_ds)

        # begin supervised training
        trainer = Trainer(model, loss, optimiser, patience, reload_best, device=device, *args, **kwargs)
        history = trainer.fit(
            new_loader, val_loader=val_loader,
            epochs=epochs,
        )

        # if early stopping was applied w/ patience, then the actual train acc and loss should be
        # -patience from the final loss/acc UNLESS we reached the maximum number of epochs.
        if patience and len(history['train_loss']) != epochs:
            return history['train_loss'][-patience], history['train_acc'][-patience]
        return history["train_loss"][-1], history["train_acc"][-1]
    e = Engine(_step)
    pseudo_label_manager.attach(e)
    return e


class EphemeralTrainer:
    def __init__(self,
                 model: ALRModel,
                 pool: UnlabelledDataset,
                 loss: _Loss_fn, optimiser: str,
                 threshold: float,
                 random_fixed_length_sampler_length: Optional[int] = None,
                 root: Optional[str] = None,
                 patience: Optional[int] = None,
                 reload_best: Optional[bool] = False,
                 device: _DeviceType = None,
                 pool_loader_kwargs: Optional[dict] = {},
                 *args, **kwargs):
        self._pool = pool
        self._model = model
        self._loss = loss
        self._optimiser = optimiser
        self._patience = patience
        self._reload_best = reload_best
        self._device = device
        self._args = args
        self._kwargs = kwargs
        self._threshold = threshold
        self._root = root
        self._pool_loader_kwargs = pool_loader_kwargs
        self._rfls_len = random_fixed_length_sampler_length

    def fit(self, train_loader: torchdata.DataLoader,
            val_loader: Optional[torchdata.DataLoader] = None,
            iterations: Optional[int] = 1,
            epochs: Optional[int] = 1):
        if self._patience and val_loader is None:
            raise ValueError("If patience is specified, then val_loader must be provided in .fit().")

        val_evaluator = create_supervised_evaluator(
            self._model, metrics={'acc': Accuracy(), 'loss': Loss(self._loss)},
            device=self._device,
        )

        history = defaultdict(list)
        pbar = ProgressBar()

        def _log_metrics(engine: Engine):
            # train_loss and train_acc are moving averages of the last epoch
            # in the supervised training loop
            train_loss, train_acc = engine.state.output
            history[f"train_loss"].append(train_loss)
            history[f"train_acc"].append(train_acc)
            pbar.log_message(
                f"Eph. iteration {engine.state.epoch}/{engine.state.max_epochs}\n"
                f"\ttrain acc = {train_acc}, train loss = {train_loss}"
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

        pseudo_label_manager = PseudoLabelManager(
            pool=self._pool, model=self._model,
            threshold=self._threshold, log_dir=self._root,
            device=self._device, **self._pool_loader_kwargs
        )
        trainer = create_pseudo_label_trainer(
            model=self._model, loss=self._loss, optimiser=self._optimiser,
            train_loader=train_loader, val_loader=val_loader,
            pseudo_label_manager=pseudo_label_manager, rfls_len=self._rfls_len,
            patience=self._patience, reload_best=self._reload_best,
            epochs=epochs, device=self._device, *self._args, **self._kwargs,
        )
        # output of trainer are running averages of train_loss and train_acc (from the
        # last epoch of the supervised trainer)
        pbar.attach(trainer, output_transform=lambda x: {'loss': x[0], 'acc': x[1]})
        if val_loader is not None and self._patience:
            es = EarlyStopper(self._model, self._patience, trainer, key='acc', mode='max')
            es.attach(val_evaluator)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, _log_metrics)
        trainer.run(
            range(iterations), max_epochs=iterations, epoch_length=1,
        )
        if val_loader is not None and self._patience and self._reload_best:
            es.reload_best()

        history['train_size'] = np.array(pseudo_label_manager.acquired_sizes) + len(train_loader.dataset)
        return history

    def evaluate(self, data_loader: torchdata.DataLoader) -> dict:
        evaluator = create_supervised_evaluator(
            self._model, metrics={'acc': Accuracy(), 'loss': Loss(self._loss)},
            device=self._device
        )
        return evaluator.run(data_loader).metrics
