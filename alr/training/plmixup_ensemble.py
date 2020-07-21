import numpy as np
import tempfile
from typing import Optional, Tuple, Callable, Union, List
import torch.utils.data as torchdata
from torch.optim.lr_scheduler import ReduceLROnPlateau

from alr.training.pl_mixup import mixup, reg_mixup_loss, PDS, IndexMarker, onehot_transform, create_warmup_trainer
from alr.training.utils import EarlyStopper
from alr.utils._type_aliases import _DeviceType
from alr.training.samplers import RandomFixedLengthSampler, MinLabelledSampler
from alr.utils import _map_device
from torch import nn
from torch.nn import functional as F
import torch

from ignite.engine import create_supervised_evaluator, Events
from ignite.metrics import Accuracy, Loss

from pathlib import Path


class PLMixupEnsembleTrainer:
    def __init__(self,
                 models: List[nn.Module],
                 optimiser: str,
                 train_transform: Callable,
                 test_transform: Callable,
                 optimiser_kwargs,
                 loader_kwargs,
                 rfls_length: int,
                 log_dir: Optional[str] = None,
                 alpha: Optional[float] = 1.0,
                 min_labelled: Optional[Union[int, float]] = 16,
                 num_classes: Optional[int] = 10,
                 data_augmentation: Optional[Callable] = None,
                 batch_size: Optional[int] = 100,
                 patience: Optional[Union[Tuple[int, int], int]] = (5, 25),
                 lr_patience: Optional[int] = 10,
                 device: _DeviceType = None):
        self._models = models
        self._train_transform = train_transform
        self._test_transform = test_transform
        self._data_augmentation = data_augmentation
        self._optim_kwargs = optimiser_kwargs
        self._optimiser = optimiser
        self._device = device
        self._batch_size = batch_size
        self._patience = patience
        self._loader_kwargs = loader_kwargs
        self._rfls_length = rfls_length
        self._min_labelled = min_labelled
        self._num_classes = num_classes
        self._alpha = alpha
        self._lr_patience = lr_patience
        self._log_dir = log_dir
        self.soft_label_history = None

    def _instantiate_optimiser(self, model: nn.Module):
        return getattr(
            torch.optim, self._optimiser
        )(model.parameters(), **self._optim_kwargs)

    def fit(self,
            train: torchdata.Dataset,
            val: torchdata.Dataset,
            pool: torchdata.Dataset,
            epochs: Optional[Tuple[int, int]] = (50, 400)):
        # stage 1
        if isinstance(self._patience, int):
            pat1, pat2 = self._patience
        else:
            pat1, pat2 = self._patience[0], self._patience[1]
        train = PDS(
            IndexMarker(
                train, mark=IndexMarker.LABELLED
            ),
            transform=self._train_transform,
            augmentation=self._data_augmentation,
            target_transform=onehot_transform(self._num_classes),
        )
        pool = PDS(
            IndexMarker(pool, mark=IndexMarker.PSEUDO_LABELLED),
            transform=self._train_transform,
            augmentation=self._data_augmentation,
        )
        val = PDS(
            IndexMarker(val, mark=None),
            transform=self._test_transform,
        )
        val._with_metadata = False
        train_loader = torchdata.DataLoader(
            train, batch_size=self._batch_size,
            sampler=RandomFixedLengthSampler(
                train, self._rfls_length, shuffle=True
            ),
            **self._loader_kwargs
        )
        pool_loader = torchdata.DataLoader(
            pool, batch_size=512,
            shuffle=False, **self._loader_kwargs
        )
        val_loader = torchdata.DataLoader(
            val, batch_size=512,
            shuffle=False, **self._loader_kwargs
        )

        models = self._models
        optimisers = [self._instantiate_optimiser(m) for m in models]
        history = {
            'val_loss': [[] for _ in range(len(models))],
            'val_acc': [[] for _ in range(len(models))],
            'override_acc': [],
        }

        print("Commencing stage 1 ...")
        with train.no_fluff():
            for idx, (m, o) in enumerate(zip(models, optimisers)):
                print(f"\tTraining model {idx + 1} of {len(models)}")
                val_eval = create_supervised_evaluator(
                    m, metrics={
                        'acc': Accuracy(),
                        'loss': Loss(F.nll_loss)
                    }, device=self._device
                )
                trainer = create_warmup_trainer(
                    m, optimiser=o,
                    device=self._device,
                )
                es = EarlyStopper(
                    m, patience=pat1,
                    trainer=trainer, key='acc', mode='max'
                )
                es.attach(val_eval)

                @trainer.on(Events.EPOCH_COMPLETED)
                def _log(_):
                    metrics = val_eval.run(val_loader).metrics
                    acc, loss = metrics['acc'], metrics['loss']
                    history['val_acc'][idx].append(acc)
                    history['val_loss'][idx].append(loss)
                trainer.run(train_loader, max_epochs=epochs[0])
                es.reload_best()
                print(f"\tModel {idx + 1} of {len(models)} done, "
                      f"acc = {max(history['val_acc'][idx])}")

        # pseudo-label points
        plab_acc = self._override_pool_labels(
            pool, pool_loader
        )
        print(f"End of stage 1: overridden labels' acc: {plab_acc}")
        history['override_acc'].append(plab_acc)

        # stage 2
        full_dataset = torchdata.ConcatDataset((train, pool))
        fds_loader = torchdata.DataLoader(
            full_dataset, batch_sampler=MinLabelledSampler(
                train, pool, batch_size=self._batch_size,
                min_labelled=self._min_labelled
            ), **self._loader_kwargs
        )
        # reset optimiser
        optimisers = [self._instantiate_optimiser(m) for m in models]
        schedulers = [
            ReduceLROnPlateau(
                op, mode='max',
                factor=.1, patience=self._lr_patience,
                verbose=True, min_lr=1e-3,
            ) for op in optimisers]
        model_tracker = [PerformanceTracker(m, pat2) for m in models]
        current_epoch = 0

        print("Commencing stage 2 ...")
        while any(not mt.done for mt in model_tracker) and current_epoch < epochs[1]:
            current_epoch += 1
            for idx, (m, o, mt, scheduler) in enumerate(zip(
                    models, optimisers, model_tracker,
                    schedulers
            )):
                if mt.done:
                    continue
                # train model m for one epoch
                plmixup_train(fds_loader, m, o, alpha=self._alpha, device=self._device)

                # get val acc for model m
                metrics = create_supervised_evaluator(
                    m, metrics={
                        'acc': Accuracy(),
                        'loss': Loss(F.nll_loss)
                    }, device=self._device
                ).run(val_loader).metrics
                acc, loss = metrics['acc'], metrics['loss']
                mt.step(acc)
                scheduler.step(acc)
                history['val_acc'][idx].append(acc)
                history['val_loss'][idx].append(loss)
                print(f"\tModel {idx + 1} val acc at epoch {current_epoch} = {acc:.4f}")

            # reload best weights if haven't done so
            for idx, mt in enumerate(model_tracker):
                if mt.done and not mt.reloaded:
                    print(f"\tModel {idx + 1} converged, reloading weights")
                    mt.reload_best()

            plab_acc = self._override_pool_labels(
                pool, pool_loader
            )
            history['override_acc'].append(plab_acc)
            print(f"\tEpoch {current_epoch}/{epochs[1]}: "
                  f"mean val acc = {np.mean([h[-1] for h in history['val_acc']]):.4f}; "
                  f"pseudo-label acc = {plab_acc:.4f}")

        # the last element in pool.label_history is the most accurate one to-date:
        #  all the individual models have (converged and) reloaded their weights
        self.soft_label_history = torch.stack(pool.label_history, dim=0)
        return history

    def _override_pool_labels(self, pool, pool_loader):
        ensemble = Ensemble(self._models)
        with pool.no_augmentation():
            with pool.no_fluff():
                pseudo_labels = []
                with torch.no_grad():
                    for x, _ in pool_loader:
                        x = x.to(self._device)
                        # NOTE: ensemble's forward call returns softmax probabilities
                        pseudo_labels.append(ensemble(x).detach().cpu())
        pool.override_targets(torch.cat(pseudo_labels))
        return pool.override_accuracy

    def evaluate(self, data_loader: torchdata.DataLoader) -> dict:
        ensemble = Ensemble(self._models, return_log=True)  # return_log=True for the loss function
        evaluator = create_supervised_evaluator(
            ensemble, metrics={'acc': Accuracy(), 'loss': Loss(F.nll_loss)},
            device=self._device
        )
        return evaluator.run(data_loader).metrics


class PerformanceTracker:
    def __init__(self, model: nn.Module, patience: int):
        self.model = model
        self.patience = patience
        self._original_patience = patience
        self.last_acc = None
        self._temp_dir = tempfile.TemporaryDirectory()
        self._model_filename = Path(str(self._temp_dir.name)).absolute() / f"{id(self)}.pt"
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
            raise RuntimeError("Cannot reload model until step is called at least once.")
        self.model.load_state_dict(torch.load(self._model_filename), strict=True)
        self._temp_dir.cleanup()
        self._reloaded = True


class Ensemble:
    def __init__(self, models: list, return_log: bool = False):
        # assumes models return log-softmax probabilities
        self.models = models
        self.return_log = return_log

    def forward(self, x):
        if self.return_log:
            return torch.log(self.get_preds(x).mean(dim=0) + 1e-5)
        return self.get_preds(x).mean(dim=0)

    def get_preds(self, x):
        preds = []
        with torch.no_grad():
            for m in self.models:
                m.eval()
                preds.append(m(x).exp())
        return torch.stack(preds)

    def evaluate(self, loader, device):
        with torch.no_grad():
            correct = 0
            total = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                correct += (self.forward(x).argmax(dim=1) == y).sum().item()
                total += y.size(0)
        return correct / total

    def __call__(self, x):
        return self.forward(x)

    def eval(self):
        self.train(mode=False)

    def train(self, mode=True):
        # should never ever be in training mode
        assert not mode

    def save_weights(self, prefix: str):
        for mi, m in enumerate(self.models, 1):
            torch.save(m.state_dict(), f"{prefix}_model_{mi}.pt")


def plmixup_train(loader, model, optimiser, alpha, device):
    model.train()
    for _, img_aug, target, _, _ in loader:
        img_aug, target = _map_device(
            [img_aug, target], device
        )
        xp, y1, y2, lamb = mixup(img_aug, target, alpha=alpha)
        preds = model(xp)
        loss = reg_mixup_loss()(preds, y1, y2, lamb)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
