from torch.optim.lr_scheduler import ReduceLROnPlateau

from alr.training.progress_bar.ignite_progress_bar import ProgressBar
from alr.training.samplers import RandomFixedLengthSampler, MinLabelledSampler
import torch.utils.data as torchdata
import torch
from typing import Optional, Tuple, Callable, Union
from torch import nn
from alr.utils._type_aliases import _DeviceType
from alr.training.utils import EarlyStopper, PLPredictionSaver
from alr.utils import _map_device
import numpy as np
from contextlib import contextmanager
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from torch.nn import functional as F


class IndexMarker(torchdata.Dataset):
    PSEUDO_LABELLED = True
    LABELLED = False

    def __init__(self, dataset: torchdata.Dataset, mark):
        self.dataset = dataset
        self.mark = mark

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # returns (x, y), idx, mark
        return self.dataset[idx], idx, self.mark


class PDS(torchdata.Dataset):
    def __init__(self,
                 dataset: IndexMarker,
                 transform: Callable[[torch.Tensor], torch.Tensor],
                 augmentation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 target_transform: Callable = lambda x: x):
        self.dataset = dataset
        self._augmentation = augmentation
        self._transform = transform
        self._with_metadata = True
        self._new_targets = None
        self._target_transform = target_transform
        self._original_labels = False
        self.label_history = []

    def __getitem__(self, idx):
        (img_raw, target), idx, mark = self.dataset[idx]

        # override target
        if self._new_targets is not None and (not self._original_labels):
            target = self._new_targets[idx]

        if self._augmentation:
            img_aug = self._augmentation(img_raw)
            img_raw, img_aug = map(self._transform, [img_raw, img_aug])
        else:
            img_raw = self._transform(img_raw)
            img_aug = img_raw
        if self._with_metadata:
            return img_raw, img_aug, self._target_transform(target), idx, mark
        return img_aug, self._target_transform(target)

    def __len__(self):
        return len(self.dataset)

    @contextmanager
    def original_labels(self):
        if not self._original_labels:
            self._original_labels = True
            yield
            self._original_labels = False
        else:
            yield self

    @contextmanager
    def no_fluff(self):
        if self._with_metadata:
            self._with_metadata = False
            yield self
            self._with_metadata = True
        else:
            yield self

    @contextmanager
    def no_augmentation(self):
        if self._augmentation:
            store = self._augmentation
            self._augmentation = None
            yield self
            self._augmentation = store
        else:
            yield self

    def override_targets(self, new_targets: torch.Tensor):
        assert new_targets.size(0) == len(self.dataset)
        # new_targets = [N x C]
        self.label_history.append(new_targets)
        self._new_targets = new_targets

    @property
    def override_accuracy(self):
        assert self._new_targets is not None
        correct = 0
        for i in range(len(self)):
            overridden_target = self._new_targets[i]
            original_target = self.dataset[i][0][-1]
            correct += (overridden_target.argmax(dim=-1).item() == original_target)
        return correct / len(self)


# from https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py#L119
def mixup(x: torch.Tensor,
          y: torch.Tensor,
          alpha: float = 1.0,
          device: _DeviceType = None):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    if device:
        index = index.to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def reg_nll_loss(coef: Optional[Tuple[float, float]] = (.8, .4)):
    def _reg_nll_loss(pred: torch.Tensor,
                      target: torch.Tensor):
        C = target.size(-1)
        prob = pred.exp()
        # heuristic: empirical mean of mini-batch
        prob_avg = prob.mean(dim=0)
        # uniform prior
        prior = target.new_ones(C) / C

        # standard cross entropy loss: H[target, pred]
        ce_loss = -torch.mean(
            torch.sum(target * pred, dim=1)
        )
        # prior loss: KL(prior || empirical mean) = sum c=1..C of prior * log[prior/emp. mean]
        # note, this is simplified, the full prior loss is:
        #  sum(prior * log[prior] - prior * log[prob_avg])
        # but since the first term is a constant, we drop it.
        prior_loss = -torch.sum(
            prior * torch.log(prob_avg)
        )
        # entropy loss: neg. mean of sum c=1..C of p(y=c|x)log[p(y=c|x)]
        entropy_loss = -torch.mean(
            torch.sum(prob * pred, dim=1)
        )
        return ce_loss + coef[0] * prior_loss + coef[1] * entropy_loss

    return _reg_nll_loss


def reg_mixup_loss(coef: Optional[Tuple[float, float]] = (.8, .4)):
    def _reg_mixup_loss(pred: torch.Tensor,
                        y1: torch.Tensor,
                        y2: torch.Tensor,
                        lamb: int):
        """
        pred is log_softmax,
        y1 and y2 are softmax probabilities
        """
        C = y1.size(-1)
        assert y2.size(-1) == C
        # NxC
        prob = pred.exp()
        # C
        prob_avg = prob.mean(dim=0)
        prior = y2.new_ones(C) / C

        # term1, term2, [1,]
        term1 = -torch.mean(torch.sum(y1 * pred, dim=1))
        term2 = -torch.mean(torch.sum(y2 * pred, dim=1))
        mixup_loss = lamb * term1 + (1 - lamb) * term2

        prior_loss = -torch.sum(
            prior * torch.log(prob_avg)
        )
        entropy_loss = -torch.mean(
            torch.sum(prob * pred, dim=1)
        )

        return mixup_loss + coef[0] * prior_loss + coef[1] * entropy_loss

    return _reg_mixup_loss


def onehot_transform(n):
    def _onehot_transform(x: int):
        # return torch.eye(n)[x.long()]
        res = torch.zeros(size=(n,))
        res[x] = 1
        return res
    return _onehot_transform


def create_warmup_trainer(
        model: nn.Module,
        optimiser,
        device: _DeviceType = None
):

    def _step(engine: Engine, batch):
        model.train()
        # prepare batch
        x, y = batch
        x, y = _map_device([x, y], device)

        # predict, loss, optimise
        pred = model(x)
        loss = reg_nll_loss()(pred, y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        return loss.item()

    return Engine(_step)


class PLMixupTrainer:
    def __init__(self,
                 model: nn.Module,
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
        # for now, assume model returns logsoftmax - ceebs.
        self._model = model
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

    def _instantiate_optimiser(self):
        return getattr(
            torch.optim, self._optimiser
        )(self._model.parameters(), **self._optim_kwargs)

    def fit(self,
            train: torchdata.Dataset,
            val: torchdata.Dataset,
            pool: torchdata.Dataset,
            epochs: Optional[Tuple[int, int]] = (50, 400)):
        if isinstance(self._patience, int):
            pat1, pat2 = self._patience
        else:
            pat1, pat2 = self._patience[0], self._patience[1]
        history = {
            'val_loss': [],
            'val_acc': [],
            'override_acc': [],
        }
        optimiser = self._instantiate_optimiser()
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
        pbar = ProgressBar(desc=lambda _: "Stage 1")

        # warm up
        with train.no_fluff():
            val_eval = create_supervised_evaluator(
                self._model, metrics={
                    'acc': Accuracy(),
                    'loss': Loss(F.nll_loss)
                }, device=self._device
            )
            trainer = create_warmup_trainer(
                self._model, optimiser=optimiser,
                device=self._device,
            )
            es = EarlyStopper(
                self._model, patience=pat1,
                trainer=trainer, key='acc', mode='max'
            )
            es.attach(val_eval)

            @trainer.on(Events.EPOCH_COMPLETED)
            def _log(e: Engine):
                metrics = val_eval.run(val_loader).metrics
                acc, loss = metrics['acc'], metrics['loss']
                pbar.log_message(f"\tStage 1 epoch {e.state.epoch}/{e.state.max_epochs} "
                                 f"[val] acc, loss = "
                                 f"{acc:.4f}, {loss:.4f}")
                history['val_acc'].append(acc)
                history['val_loss'].append(loss)
            pbar.attach(trainer)
            trainer.run(train_loader, max_epochs=epochs[0])
            es.reload_best()

        # pseudo-label points
        with pool.no_augmentation():
            with pool.no_fluff():
                pseudo_labels = []
                with torch.no_grad():
                    self._model.eval()
                    for x, _ in pool_loader:
                        x = x.to(self._device)
                        # add (softmax) probability, hence .exp()
                        pseudo_labels.append(self._model(x).exp().detach().cpu())
        pool.override_targets(torch.cat(pseudo_labels))
        plab_acc = pool.override_accuracy
        pbar.log_message(f"\t*End of stage 1*: overridden labels' acc: {plab_acc}")
        history['override_acc'].append(plab_acc)

        # start training with PL
        full_dataset = torchdata.ConcatDataset((train, pool))
        fds_loader = torchdata.DataLoader(
            full_dataset, batch_sampler=MinLabelledSampler(
                train, pool, batch_size=self._batch_size,
                min_labelled=self._min_labelled
            ), **self._loader_kwargs
        )
        val_eval = create_supervised_evaluator(
            self._model, metrics={
                'acc': Accuracy(),
                'loss': Loss(F.nll_loss)
            }, device=self._device
        )
        optimiser = self._instantiate_optimiser()
        scheduler = ReduceLROnPlateau(
            optimiser, mode='max',
            factor=.1, patience=self._lr_patience,
            verbose=True, min_lr=1e-3,
        )
        trainer = create_plmixup_trainer(
            self._model, optimiser,
            pool, alpha=self._alpha,
            num_classes=self._num_classes,
            log_dir=self._log_dir,
            device=self._device
        )
        es = EarlyStopper(
            self._model, patience=pat2,
            trainer=trainer, key='acc', mode='max'
        )
        es.attach(val_eval)

        pbar = ProgressBar(desc=lambda _: "Stage 2")

        @trainer.on(Events.EPOCH_COMPLETED)
        def _log(e: Engine):
            metrics = val_eval.run(val_loader).metrics
            acc, loss = metrics['acc'], metrics['loss']
            pbar.log_message(f"\tEpoch {e.state.epoch}/{e.state.max_epochs} "
                             f"[val] acc, loss = "
                             f"{acc:.4f}, {loss:.4f}")
            history['val_acc'].append(acc)
            history['val_loss'].append(loss)
            history['override_acc'].append(pool.override_accuracy)
            scheduler.step(acc)

        pbar.attach(trainer)
        trainer.run(fds_loader, max_epochs=epochs[1])
        es.reload_best()
        soft_label_history = pool.label_history
        if trainer.state.epoch != trainer.state.max_epochs:
            soft_label_history = soft_label_history[:-pat2]
        self.soft_label_history = torch.stack(soft_label_history, dim=0)
        return history

    def evaluate(self, data_loader: torchdata.DataLoader) -> dict:
        evaluator = create_supervised_evaluator(
            self._model, metrics={'acc': Accuracy(), 'loss': Loss(F.nll_loss)},
            device=self._device
        )
        return evaluator.run(data_loader).metrics


def create_plmixup_trainer(model, optimiser, pool, alpha, num_classes, log_dir, device):
    def _step(engine: Engine, batch):
        model.train()
        img_raw, img_aug, target, idx, mark = batch
        img_raw, img_aug, target, idx, mark = _map_device(
            [img_raw, img_aug, target, idx, mark], device
        )
        xp, y1, y2, lamb = mixup(img_aug, target, alpha=alpha)
        preds = model(xp)
        loss = reg_mixup_loss()(preds, y1, y2, lamb)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        return loss.item(), img_raw, img_aug, target, idx, mark
    e = Engine(_step)
    PLUpdater(
        model, pool, log_dir=log_dir,
        num_class=num_classes, device=device
    ).attach(e)
    return e


class PLUpdater:
    def __init__(self, model: nn.Module, pool: PDS,
                 log_dir: str, num_class: int, device=None):
        self._pseudo_labels = torch.empty(size=(len(pool), num_class))
        self._model = model
        self._pool = pool
        self._log_dir = log_dir
        self._device = device

    def attach(self, engine: Engine):
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._on_iteration_end)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self._on_epoch_end)

    def _on_iteration_end(self, engine: Engine):
        # after iteration ended
        _, img_raw, img_aug, target, idx, mark = engine.state.output
        with torch.no_grad():
            self._model.eval()
            pld_mask = mark == IndexMarker.PSEUDO_LABELLED
            # unaugmented, raw, pseudo-labelled images
            pld_img = img_raw[pld_mask]
            # get *softmax* predictions -- exponentiate the output!
            new_pld = self._model(pld_img).exp().detach().cpu()
            self._pseudo_labels[idx[pld_mask]] = new_pld

    def _on_epoch_end(self, engine: Engine):
        self._pool.override_targets(self._pseudo_labels)

        if self._log_dir is not None:
            # original pool labels w/o augmentation and metadata from PDS
            with self._pool.no_augmentation():
                with self._pool.no_fluff():
                    with self._pool.original_labels():
                        _calib_metrics(
                            self._model, self._pool, self._log_dir,
                            other_engine=engine, device=self._device
                        )


def _calib_metrics(model, ds, log_dir,
                   other_engine=None, device=None,
                   pred_transform=lambda x: x.exp()):
    kwargs = {} if not torch.cuda.is_available() else dict(
        num_workers=4, pin_memory=True
    )
    loader = torchdata.DataLoader(
        ds, shuffle=False, batch_size=512, **kwargs
    )
    save_pl_metrics = create_supervised_evaluator(
        model, metrics=None, device=device
    )
    pps = PLPredictionSaver(
        log_dir=log_dir,
        pred_transform=pred_transform,
    )
    pps.attach(save_pl_metrics)
    if other_engine is not None:
        pps.global_step_from_engine(other_engine)
    save_pl_metrics.run(loader)


class _WithTransform(torchdata.Dataset):
    def __init__(self, dataset: torchdata.Dataset, transform, with_targets):
        super(_WithTransform, self).__init__()
        self._dataset = dataset
        self._transform = transform
        self._with_targets = with_targets

    def __getitem__(self, idx):
        if self._with_targets:
            x, y = self._dataset[idx]
            return self._transform(x), y
        # (x,) only
        return self._transform(self._dataset[idx])

    def __len__(self):
        return len(self._dataset)


def temp_ds_transform(transform, with_targets=False):
    def _trans(dataset: torchdata.Dataset) -> torchdata.Dataset:
        return _WithTransform(dataset, transform, with_targets)
    return _trans
