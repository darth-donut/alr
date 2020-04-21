"""
Main alr module
"""

import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.data as torchdata
import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.nn import functional as F

from alr.acquisition import AcquisitionFunction
from alr.modules.dropout import replace_dropout
from alr.utils import _DeviceType

__version__ = '0.0.0b2'


class ALRModel(nn.Module, ABC):
    def __init__(self):
        """
        A :class:`ALRModel` provides generic methods required for common
        operations in active learning experiments.
        """
        super(ALRModel, self).__init__()
        self._models = []

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Regular forward pass. Usually reserved for training.

        :param x: input tensor
        :type x: `torch.Tensor`
        :return: output tensor
        :rtype: `torch.Tensor`
        """
        pass

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Override this function if predict has a different behaviour from :meth:`forward`.
        For example, if :meth:`forward` returns logits, this function could augment the
        output with softmax. Another example would be if the base model is a :class:`MCDropout` model,
        in which case, this function should return multiple stochastic forward passes.

        :param x: input tensor
        :type x: `torch.Tensor`
        :return: output tensor
        :rtype: `torch.Tensor`
        """
        return self.forward(x)

    def reset_weights(self) -> None:
        """
        Resets the model's weights.

        :return: None
        :rtype: NoneType
        """
        for m, state in self._models:
            # reload initial states
            m.load_state_dict(state)

    def __setattr__(self, key, value):
        super(ALRModel, self).__setattr__(key, value)
        if isinstance(value, nn.Module):
            # register nn.Module
            self._models.append((value, value.state_dict()))


class MCDropout(ALRModel):
    def __init__(self, model: nn.Module, forward: Optional[int] = 100, clone: Optional[bool] = False):
        """
        Implements `Monte Carlo Dropout <https://arxiv.org/abs/1506.02142>`_ (MCD). The difference between
        :meth:`forward` and :meth:`predict`
        is that the former returns a single forward pass (logits) while the
        latter returns the mean of `forward` number of passes (softmax scores).

        :param model: base `torch.nn.Module` object
        :type model: `nn.Module`
        :param forward: number of stochastic forward passes
        :type forward: int, optional
        :param clone: If `False`, the `model` is modified *in-place* such that the
                        dropout layers are replaced with :mod:`~alr.modules.dropout` layers.
                        If `True`, `model` is not modified and a new model is cloned.
        :type clone: `bool`, optional
        """
        super(MCDropout, self).__init__()
        self.base_model = replace_dropout(model, clone=clone)
        self.n_forward = forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Regular forward pass. Raises exception if `self.training` is `False`.

        :param x: input tensor
        :type x: `torch.Tensor`
        :return: output tensor
        :rtype: `torch.Tensor`
        :raises RuntimeError: raises `RuntimeError` if this method is used during evaluation;
            :meth:`predict` should be used instead.
        """
        if self.training:
            assert self.base_model.training
            return self.base_model(x)
        raise RuntimeError('Use model.predict(x) during evaluation.')

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the mean softmax score of :meth:`stochastic_forward` passes.

        Equivalent to:

        .. code:: python

            def predict(x):
                return torch.mean(stochastic_forward(x), dim=0)

        :param x: input tensor
        :type x: `torch.Tensor`
        :return: output tensor
        :rtype: `torch.Tensor`
        """
        return torch.mean(self.stochastic_forward(x), dim=0)

    def stochastic_forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Returns a :math:`m \times N \times C` `torch.Tensor` where:

            1. :math:`m` is equal to `self.n_forward`
            2. :math:`N` is the batch size, equal to `x.size(0)`
            3. :math:`C` is the number of units in the final layer (e.g. number of classes in a classification model)

        :param x: input tensor
        :type x: `torch.Tensor`
        :return: output tensor of shape :math:`m \times N \times C`
        :rtype: `torch.Tensor`

        .. warning::
            It is the *user's own responsibility* to make sure that the model
            is in training mode. For example, unless dropout is hard-coded
            to always drop units in the base model, stochastic_forward will
            not have any stochasticity!
        """
        preds = torch.stack(
            [F.softmax(self.base_model(x), dim=1) for _ in range(self.n_forward)]
        )
        assert preds.size(0) == self.n_forward
        return preds


class DataManager:
    def __init__(self, acquirer: AcquisitionFunction, X_train: torch.Tensor, y_train: torch.Tensor,
                 X_pool: torch.Tensor, y_pool: torch.Tensor, **data_loader_params):
        """
        A stateful data manager class

        Training data and labels will be updated according to newly acquired samples
        as dictated by the provided `acquirer`.
        :py:attr:`training_data` returns said data
        as a :class:`~torch.utils.data.DataLoader` object with the specified `batch_size` in `data_loader_params`.

        :param acquirer: acquisition object
        :type acquirer: :class:`AcquisitionFunction`
        :param X_train: tensor object
        :type X_train: `torch.Tensor`
        :param y_train: tensor object
        :type y_train: `torch.Tensor`
        :param X_pool: tensor object
        :type X_pool: `torch.Tensor`
        :param y_pool: tensor object
        :type y_pool: `torch.Tensor`
        :param data_loader_params: keyword parameters to be passed into `DataLoader` when calling
            :py:attr:`training_data`
        """
        # TODO: accept as y_pool as None for actual use-cases
        self._acquirer = acquirer
        self._X_train = X_train
        self._y_train = y_train
        self._X_pool = X_pool
        self._y_pool = y_pool
        if not data_loader_params:
            self._data_loader_params = dict(shuffle=True, num_workers=2,
                                            pin_memory=True, batch_size=32)
        else:
            self._data_loader_params = data_loader_params

    def acquire(self, b: int) -> None:
        """
        Acquires `b` points from the provided `X_pool` according to `acquirer`.

        :param b: number of points to acquire
        :type b: int
        :return: None
        :rtype: NoneType
        """
        assert b <= self._X_pool.size(0)
        idxs = self._acquirer(self._X_pool, b)
        assert idxs.shape == (b,)
        self._X_train = torch.cat((self._X_train, self._X_pool[idxs]), dim=0)
        self._y_train = torch.cat((self._y_train, self._y_pool[idxs]), dim=0)
        mask = torch.ones(self._X_pool.size(0), dtype=torch.bool)
        mask[idxs] = 0
        self._X_pool = self._X_pool[mask]
        self._y_pool = self._y_pool[mask]

    @property
    def training_data(self) -> torch.utils.data.DataLoader:
        """
        Returns current training data after being updated by :meth:`acquire`.

        :return: A `DataLoader` object than represents the latest updated training pool.
        :rtype: `DataLoader`
        """
        return torch.utils.data.DataLoader(torchdata.TensorDataset(self._X_train, self._y_train),
                                           **self._data_loader_params)

    @property
    def train_size(self) -> int:
        """
        Current number of points in `X_train`.

        :return: `X_train.size(0)`
        :rtype: int
        """
        return self._X_train.size(0)


def _train(model: nn.Module, dataloader: torch.utils.data.DataLoader, optimiser: torch.optim.Optimizer,
           epochs: int = 50, device: _DeviceType = None) -> List[float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    tepochs = tqdm.trange(epochs, file=sys.stdout)
    for _ in tepochs:
        epoch_losses = []
        for x, y in dataloader:
            if device:
                x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            epoch_losses.append(loss.item())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        losses.append(np.mean(epoch_losses))
        tepochs.set_postfix(loss=losses[-1])
    return losses


def _evaluate(model: ALRModel,
              dataloader: torch.utils.data.DataLoader,
              device: _DeviceType = None) -> float:
    model.eval()
    score = total = 0
    with torch.no_grad():
        for x, y in dataloader:
            if device:
                x, y = x.to(device), y.to(device)
            _, preds = torch.max(model.predict(x), dim=1)
            score += (preds == y).sum().item()
            total += y.size(0)

    return score / total


def stratified_partition(X_train: np.array, y_train: np.array, train_size: Optional[int] = 20) ->\
        Tuple[np.array, np.array, np.array, np.array]:
    """
    Returns (`X_train`, `y_train`, `X_pool`, `y_pool`) where `X_train.size(0) == train_size` and
    `y_train`'s classes are as balanced as possible.

    :param X_train: training input
    :type X_train: `np.array`
    :param y_train: training targets
    :type y_train: `np.array`
    :param train_size: `X_train`'s output size
    :type train_size: int, optional
    :return: (`X_train`, `y_train`, `X_pool`, `y_pool`) where
             `X_train.size(0) == train_size` and
             `y_train`'s classes are as balanced as possible.
    :rtype: 4-tuple consisting of `np.array`
    """
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size)
    train_idxs, pool_idxs = next(sss.split(X_train, y_train))
    return (X_train[train_idxs], y_train[train_idxs],
            X_train[pool_idxs], y_train[pool_idxs])


def run_experiment(model: ALRModel, acquisition_function: AcquisitionFunction, X_train: torch.Tensor,
                   y_train: torch.Tensor, X_pool: torch.Tensor, y_pool: torch.Tensor,
                   val_data_loader: torch.utils.data.DataLoader, optimiser: torch.optim.Optimizer,
                   *, b: Optional[int] = 10, iters: Optional[int] = 20,
                   init_epochs: Optional[int] = 75, epochs: Optional[int] = 50,
                   device: _DeviceType = None, pin_memory: Optional[bool] = True,
                   num_workers: Optional[int] = 2, batch_size: Optional[int] = 32) -> Dict[(int, float)]:
    r"""
    A helper function useful for running a simple train-evaluate-acquire active learning loop.

    :param model: an :class:`ALRModel` object
    :type model: :class:`ALRModel`
    :param acquisition_function: an :class:`AcquisitionFunction` object
    :type acquisition_function: :class:`AcquisitionFunction`
    :param X_train: A tensor with shape :math:`N \times C \times H \times W`
    :type X_train: `torch.Tensor`
    :param y_train: A tensor with shape :math:`N`
    :type y_train: `torch.Tensor`
    :param X_pool: A tensor with shape :math:`N' \times C \times H \times W`
    :type X_pool: `torch.Tensor`
    :param y_pool: A tensor with shape :math:`N'`
    :type y_pool: `torch.Tensor`
    :param val_data_loader: data loader used to calculate test/validation loss/acc
    :type val_data_loader: `torch.utils.data.DataLoader`
    :param optimiser: PyTorch optimiser object
    :type optimiser: `torch.optim.Optimiser`
    :param b: number of samples to acquire in each iteration
    :type b: int, optional
    :param iters: number of iterations to repeat acquisition from X_pool
    :type iters: int, optional
    :param init_epochs: number of initial epochs to train
    :type init_epochs: int, optional
    :param epochs: number of epochs for each subsequent training iterations
    :type epochs: int, optional
    :param device: Device object. Used for training and evaluation.
    :type device: `None`, `str`, `torch.device`, optional
    :param pin_memory: `pin_memory` argument passed to `DataLoader` object when training model
    :type pin_memory: bool, optional
    :param num_workers: `num_workers` argument passed to `DataLoader` object when training model
    :type num_workers: int, optional
    :param batch_size: `batch_size` argument passed to `DataLoader` object when training model
    :type batch_size: int, optional
    :return: a mapping of training pool size to accuracy.
    :rtype: `dict`

    :Example:

    .. code:: python

        # load training data
        X_train, y_train = np.random.normal(0, 1, size=(100, 10)), np.random.randint(0, 5, size=(100,))
        X_test, y_test = np.random.normal(0, 1, size=(10, 10)), np.random.randint(0, 5, size=(10,))

        # partition data using stratified_partition
        X_train, y_train, X_pool, y_pool = stratified_partition(X_train, y_train, train_size=20)

        # create test DataLoader object
        test_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test),
                                                   batch_size=2048, pin_memory=True,
                                                   shuffle=False, num_workers=2)

        # instantiate model and optimiser.
        class Net(nn.Module):
            pass
        model = MCDropout(Net()).to('cpu')
        optimiser = torch.optim.Adam(model.parameters())

        # optional
        model.reset_weights()

        afunction = RandomAcquisition()
        history = run_experiment(model, afunction, X_train, y_train, X_pool, y_pool,
                                 test_dataset, optimiser, b=30, iters=10, init_epochs=100,
                                 epochs=100, device='cpu', batch_size=64,
                                 pin_memory=True, num_workers=2)

    """
    assert X_train.dim() == X_pool.dim() == 4
    assert y_train.size(0) == X_train.size(0)
    assert y_pool.size(0) == X_pool.size(0)
    X_train = X_train.clone()
    y_train = y_train.clone()
    X_pool = X_pool.clone()
    y_pool = y_pool.clone()
    dm = DataManager(acquisition_function, X_train, y_train, X_pool, y_pool,
                     shuffle=True,
                     num_workers=num_workers,
                     pin_memory=pin_memory,
                     batch_size=batch_size)
    print(f"Commencing initial training with {dm.train_size} points")
    _train(model, dataloader=dm.training_data, optimiser=optimiser, epochs=init_epochs,
           device=device)
    accs = {dm.train_size: _evaluate(model, val_data_loader, device=device)}
    print(f"Accuracy = {accs[dm.train_size]}\n=====")
    for i in range(iters):
        dm.acquire(b=b)
        print(f"Acquisition iteration {i + 1} ({(i + 1) / iters:.2%}), training size: {dm.train_size}")
        model.reset_weights()
        _train(model, dataloader=dm.training_data,
               optimiser=optimiser, epochs=epochs, device=device)
        accs[dm.train_size] = _evaluate(model, val_data_loader, device=device)
        print(f"Accuracy = {accs[dm.train_size]}\n=====")
    return accs
