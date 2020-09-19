import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, Callable, Sequence

import torch
import torch.utils.data as torchdata
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.engine import Engine, Events, \
    create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Loss, Accuracy, RunningAverage

from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from alr import MCDropout
from alr.acquisition import RandomAcquisition
from alr.data import DataManager, UnlabelledDataset
from alr.data.datasets import Dataset
from alr.training.progress_bar.ignite_progress_bar import ProgressBar
from alr.training.samplers import RandomFixedLengthSampler
from alr.training.utils import EarlyStopper
from alr.training.utils import PLPredictionSaver
from alr.utils import manual_seed, timeop, stratified_partition
from alr.utils._type_aliases import _DeviceType, _Loss_fn
import torchvision as tv
import numpy as np


class Noise(torchdata.Dataset):
    def __init__(self, length: int, channels=3, img_shape=(32, 32)):
        self.length = length
        black = torch.zeros(size=(1, channels, img_shape[0], img_shape[1]))
        white = torch.ones(size=(1, channels, img_shape[0], img_shape[1]))
        # noise
        std = 0.15
        n = length - 2
        weak = torch.randn(size=(n // 2, channels, img_shape[0], img_shape[1])) * std
        strong = torch.randn(size=(n // 2 + (n % 2), channels, img_shape[0], img_shape[1])) * std * 2
        self.data = torch.cat([weak, strong, black, white])
        assert self.data.shape == (length, channels, *img_shape)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx], 0


def xlogy(x, y):
    res = x * torch.log(y)
    res[y == 0] = .0
    assert torch.isfinite(res).all()
    return res


def get_scores(mc_preds):
    # K N C
    mc_preds = mc_preds.double()
    # N C
    mean_mc_preds = mc_preds.mean(dim=0)

    H = -(mean_mc_preds * torch.log(mean_mc_preds + 1e-5)).sum(dim=1).cpu().numpy()
    H_1 = -(xlogy(mean_mc_preds, mean_mc_preds)).sum(dim=1).cpu().numpy()

    E = (mc_preds * torch.log(mc_preds + 1e-5)).sum(dim=2).mean(dim=0).cpu().numpy()
    E_1 = (xlogy(mc_preds, mc_preds)).sum(dim=2).mean(dim=0).cpu().numpy()

    I = H + E
    I_1 = H_1 + E_1

    assert np.isfinite(I).all()
    assert np.isfinite(I_1).all()

    confidence, argmax = mean_mc_preds.max(dim=1)
    confidence, argmax = confidence.cpu().numpy(), argmax.cpu().numpy()

    assert E.shape == H.shape == I.shape == confidence.shape

    return {
        'average_entropy': -E,
        'predictive_entropy': H,
        'average_entropy2': -E_1,
        'predictive_entropy2': H_1,
        'bald_score': I,
        'bald_score2': I_1,
        'confidence': confidence,
        'class': argmax,
    }

def get_preds(models, loader, device):
    preds = []
    with torch.no_grad():
        for m in models:
            model_preds = []
            m.eval()
            for x, _ in loader:
                model_preds.append(m(x.to(device)).exp())
            preds.append(torch.cat(model_preds))
    return torch.stack(preds)

def evaluate(preds, loader, device):
    ys = torch.cat([y for _, y in loader]).to(device)
    assert ys.shape[0] == preds.shape[0]
    return (preds.argmax(dim=1) == ys).float().mean().cpu().item()


class TDataset(torchdata.Dataset):
    def __init__(self, dataset, augmentation):
        self.dataset = dataset
        self.augmentation = tv.transforms.Compose(augmentation)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        res = self.dataset[idx]
        if isinstance(res, (list, tuple)):
            x, y = res
            return self.augmentation(x), y
        else:
            return self.augmentation(res)

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 loss: _Loss_fn,
                 optimiser: str,
                 patience: Optional[int] = None,
                 lr_patience: Optional[int] = None,
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
        self._lr_patience = lr_patience
        assert patience is None or patience > 0
        assert not reload_best or patience is not None
        self._device = device
        self._model = model

    def fit(self,
            train_loader: torchdata.DataLoader,
            val_loader: Optional[torchdata.DataLoader] = None,
            epochs: Optional[int] = 1,
            callbacks: Optional[Sequence[Callable]] = None) -> Dict[str, list]:
        if self._patience and val_loader is None:
            raise ValueError("If patience is specified, then val_loader must be provided in .fit().")

        pbar = ProgressBar(desc=lambda _: "Training")
        history = defaultdict(list)

        val_evaluator = create_supervised_evaluator(
            self._model, metrics={'acc': Accuracy(), 'loss': Loss(self._loss)},
            device=self._device
        )

        if self._lr_patience:
            scheduler = ReduceLROnPlateau(
                optimizer=self._optim, mode='max',
                factor=.1, patience=self._lr_patience,
                verbose=True, min_lr=1e-3,
            )

        def _log_metrics(engine: Engine):
            # moving averages
            train_acc, train_loss = engine.state.metrics['train_acc'], engine.state.metrics['train_loss']
            history[f"train_acc"].append(train_acc)
            history[f"train_loss"].append(train_loss)
            pbar.log_message(
                f"epoch {engine.state.epoch}/{engine.state.max_epochs}\n"
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
            scheduler.step(metrics['acc'])

        trainer = create_supervised_trainer(
            self._model, optimizer=self._optim,
            loss_fn=self._loss, device=self._device,
            output_transform=lambda x, y, y_pred, loss: (loss.item(), y_pred, y),
        )
        pbar.attach(trainer)

        RunningAverage(Accuracy(output_transform=lambda x: (x[1], x[2]))).attach(trainer, 'train_acc')
        RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'train_loss')

        if val_loader is not None and self._patience:
            es = EarlyStopper(self._model, self._patience, trainer, key='acc', mode='max')
            es.attach(val_evaluator)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, _log_metrics)
        if callbacks is not None:
            for c in callbacks:
                trainer.add_event_handler(Events.EPOCH_COMPLETED, c)

        trainer.run(
            train_loader, max_epochs=epochs,
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


def calc_calib_metrics(loader, model: nn.Module, log_dir, device):
    evaluator = create_supervised_evaluator(
        model, metrics=None, device=device
    )
    pds = PLPredictionSaver(log_dir)
    pds.attach(evaluator)
    evaluator.run(loader)


def main(model_name, aug, dataset, iters, repeats, result):
    manual_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    kwargs = dict(num_workers=4, pin_memory=True)

    # ========= CONSTANTS ===========
    BATCH_SIZE = 64
    # with early stopping, this'll probably be lesser
    EPOCHS = 400
    # at least have this much points in one epoch (see RandomFixedLengthSampler)
    MIN_TRAIN_LENGTH = 20_000
    VAL_SIZE = 5_000

    REPEATS = repeats
    ITERS = iters

    template = f"{model_name}_{dataset}" + ("_aug" if aug else "")
    metrics = Path("metrics") / template
    calib_metrics = Path("calib_metrics") / template
    saved_models = Path("saved_models") / template
    result = Path(result)
    result.mkdir(parents=True)
    metrics.mkdir(parents=True)
    calib_metrics.mkdir(parents=True)
    saved_models.mkdir(parents=True)

    # ========= SETUP ===========
    if aug:
        standard_augmentation = [
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomHorizontalFlip(),
        ]
    else:
        standard_augmentation = []
    if dataset == "cifar":
        train, test = Dataset.CIFAR10.get(raw=True)
        regular_transform = [
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(*Dataset.CIFAR10.normalisation_params)
        ]
    elif dataset == "cinic":
        train, test = Dataset.CINIC10.get(raw=True)
        regular_transform = [
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(*Dataset.CINIC10.normalisation_params)
        ]
    else:
        raise ValueError("dataset only accepts two arguments: cinic or cifar")
    train, pool = stratified_partition(train, classes=10, size=20)
    pool_idxs = pool.indices[:]
    pool, val = torchdata.random_split(pool, (len(pool) - VAL_SIZE, VAL_SIZE))
    # since we need to know which points were taken for val dataset
    with open(metrics / "pool_idxs.pkl", "wb") as fp:
        pickle.dump((pool_idxs, pool.indices), fp)

    # apply transformations to pool, train, and val
    pool = UnlabelledDataset(TDataset(pool, standard_augmentation + regular_transform))
    train = TDataset(train, standard_augmentation + regular_transform)
    val = TDataset(val, regular_transform)

    val_loader = torchdata.DataLoader(
        val, batch_size=512, shuffle=False, **kwargs,
    )
    test_loader = torchdata.DataLoader(
        test, batch_size=512, shuffle=False, **kwargs,
    )
    accs = defaultdict(list)
    dm = DataManager(train, pool, RandomAcquisition())

    manual_seed(42) # reseed
    _, cifar_test = Dataset.CIFAR10.get()
    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])
    svhn_test = tv.datasets.SVHN("data", split="test", transform=transform, download=True)
    subset, _ = stratified_partition(svhn_test, classes=10, size=10_000)

    with open("subset_idxs.pkl", "wb") as fp:
        pickle.dump(subset.indices, fp)

    noise = Noise(length=20)
    score_test = torchdata.ConcatDataset((cifar_test, svhn_test, noise))
    score_test_loader = torchdata.DataLoader(
        score_test, shuffle=False, batch_size=512, **kwargs,
    )

    for r in range(1, REPEATS + 1):
        print(f"- [{model_name}] repeat #{r} of {REPEATS}-")
        dm.reset()  # this resets pool
        for i in range(1, ITERS + 1):
            models = []
            for m in range(5):
                print(f"\tTraining model {m + 1} of {m}")
                model = Dataset.CIFAR10.model.to(device)
                trainer = Trainer(
                    model, F.nll_loss, optimiser='SGD',
                    patience=20, lr_patience=16,
                    reload_best=True, device=device,
                    # SGD parameters
                    lr=0.1, weight_decay=1e-4, momentum=0.9,
                )
                train_loader = torchdata.DataLoader(
                    dm.labelled, batch_size=BATCH_SIZE,
                    sampler=RandomFixedLengthSampler(dm.labelled, MIN_TRAIN_LENGTH, shuffle=True),
                    **kwargs
                )
                trainer.fit(train_loader, val_loader, epochs=EPOCHS)
                models.append(model)
            # eval
            preds = get_preds(models, test_loader, device).mean(dim=0)
            test_acc = evaluate(preds, test_loader, device)
            print(f"=== Iteration {i} of {ITERS} ({i / ITERS:.2%}) ===")
            print(f"\ttrain: {dm.n_labelled}; val: {len(val)}; "
                  f"pool: {dm.n_unlabelled}; test: {len(test)}")
            print(f"\t[test] acc: {test_acc}")
            accs[dm.n_labelled].append(test_acc)
            with open(template + "_accs.pkl", "wb") as fp:
                pickle.dump(accs, fp)
            for mi, m in enumerate(models, 1):
                torch.save(m.state_dict(), saved_models / f"rep_{r}_iter_{i}_model_{mi}.pt")

            preds = get_preds(models, score_test_loader, device)
            scores = get_scores(preds)
            with open(result / f"rep_{r}_iter_{i}.pkl", "wb") as fp:
                pickle.dump(scores, fp)
            # next
            dm.acquire(b=400)


if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--data", choices=['cinic', 'cifar'])
    args.add_argument("--iters", default=11, type=int)
    args.add_argument("--reps", default=1, type=int)
    args.add_argument("--aug", action='store_true')
    args = args.parse_args()
    main("13cnn", args.aug, args.data, args.iters, args.reps, result="scores/13cnn")
