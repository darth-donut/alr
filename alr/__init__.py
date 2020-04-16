"""
Main alr module
"""

import numpy as np
import torch
import tqdm
import sys
import warnings
import torch.distributions as dist

from abc import ABC, abstractmethod
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.nn import functional as F
from typing import Dict, List, Optional, Tuple, Any, Union, Callable,\
    Sequence

__version__ = '0.0.0b2'
_DeviceType = Optional[Union[str, torch.device]]


class ALRDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.array, y: Optional[np.array] = None):
        """
        Wrapper class to convert numpy arrays into `torch.utils.data.Dataset`

        :param X: input features
        :type X: `np.array`
        :param y: Optional targets
        :type X: `np.array`, optional
        """
        super(ALRDataset, self).__init__()
        self._X = X
        self._y = y

    def __getitem__(self, i: int) -> Union[Any, Tuple[Any, Any]]:
        if self._y is not None:
            return self._X[i], self._y[i]
        return self._X[i]

    def __len__(self) -> int:
        return self._X.shape[0]


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


class AcquisitionFunction(ABC):
    """
    A base class for all acquisition functions. All subclasses should
    override the `__call__` method.
    """
    @abstractmethod
    def __call__(self, X_pool: torch.Tensor, b: int) -> np.array:
        """
        Given unlabelled data pool `X_pool`, return the best `b`
        points for labelling by an oracle, where the best points
        are determined by this acquisition function and its parameters.

        :param X_pool: Unlabelled dataset
        :type X_pool: `torch.Tensor`
        :param b: number of points to acquire
        :type b: int
        :return: array of indices to `X_pool`.
        :rtype: `np.array`
        """
        pass


class MCDropout(ALRModel):
    def __init__(self, model: nn.Module, forward: Optional[int] = 100):
        """
        Implements `Monte Carlo Dropout <https://arxiv.org/abs/1506.02142>`_ (MCD). The difference between
        :meth:`forward` and :meth:`predict`
        is that the former returns a single forward pass (logits) while the
        latter returns the mean of `forward` number of passes (softmax scores).

        :param model: base `torch.nn.Module` object
        :type model: `nn.Module`
        :param forward: number of stochastic forward passes
        :type forward: int, optional
        """
        super(MCDropout, self).__init__()
        self.base_model = model
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
        return torch.utils.data.DataLoader(ALRDataset(self._X_train, self._y_train),
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
        test_dataset = torch.utils.data.DataLoader(ALRDataset(X_test, y_test),
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


class RandomAcquisition(AcquisitionFunction):
    """
    Implements random acquisition. Uniformly sample `b` indices.
    """
    def __call__(self, X_pool: torch.Tensor, b: int) -> np.array:
        return np.random.choice((X_pool.size(0)), b, replace=False)


class BALD(AcquisitionFunction):
    def __init__(self, model: MCDropout,
                 subset: Optional[int] = -1,
                 device: _DeviceType = None,
                 **data_loader_params):
        r"""
        Implements `BALD <https://arxiv.org/abs/1112.5745>`_.

        .. math::

            \begin{align}
                -\sum_c\left(\frac{1}{T}\sum_t\hat{p}^t_c \right)
                 log \left( \frac{1}{T}\sum_t\hat{p}^t_c \right) +
                 \frac{1}{T}\sum_{c,t}\hat{p}^t_c log \hat{p}^t_c
            \end{align}

        where :math:`\hat{p}^t_c` is the softmax output of class :math:`c`
        on the :math:`t^{th}` stochastic iteration.

        .. code:: python

            model = MCDropout(...)
            bald = BALD(model, subset=-1, device=device,
                        batch_size=512, pin_memory=True,
                        num_workers=2)
            bald(X_pool, b=10)


        :param model: A :class:`MCDropout` model is required to calculate
                      the `m` different samples of :math:`p(y | \omega^{(m)})`.
        :type model: :class:`MCDropout`
        :param subset: Size of the subset of `X_pool`. Use -1 to denote the entire pool.
        :type subset: int, optional
        :param device: Device object used when making forward passes with the model.
        :type device: `None`, `str`, `torch.device`
        :param data_loader_params: params to be passed into `DataLoader` when
                                   iterating over `X_pool`.

        .. warning::
            Do not set `shuffle=True` in `data_loader_params`! The indices will be
            incorrect if the `DataLoader` object shuffles `X_pool`!
        """
        self._model = model
        self._device = device
        self._subset = subset
        self._dl_params = data_loader_params
        assert not self._dl_params.get('shuffle', False)

    def __call__(self, X_pool: torch.Tensor, b: int) -> np.array:
        mcmodel = self._model
        pool_size = X_pool.size(0)
        idxs = np.arange(pool_size)
        if self._subset != -1:
            r = min(self._subset, pool_size)
            assert b <= r, "Can't acquire more points that pool size"
            if b == r: return idxs
            idxs = np.random.choice(pool_size, r, replace=False)
            X_pool = X_pool[idxs]
        dl = torch.utils.data.DataLoader(ALRDataset(X_pool), **self._dl_params)
        with torch.no_grad():
            mc_preds = torch.cat(
                [mcmodel.stochastic_forward(x.to(self._device) if self._device else x)
                 for x in dl],
                dim=1
            )
            assert mc_preds.size()[:-1] == (mcmodel.n_forward, pool_size)
            mean_mc_preds = mc_preds.mean(dim=0)
            H = -(mean_mc_preds * torch.log2(mean_mc_preds)).sum(dim=1)
            E = (mc_preds * torch.log2(mc_preds)).sum(dim=2).mean(dim=0)
            I = (H + E).cpu()
            assert I.shape == (pool_size,)
            result = torch.argsort(I, descending=True).numpy()[:b]
            return idxs[result]


class ICAL(AcquisitionFunction):
    def __init__(self, model: MCDropout,
                 kernel_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 subset: Optional[int] = 200,
                 greedy_acquire: Optional[int] = 1,
                 use_one_hot: Optional[bool] = True,
                 sample_softmax: Optional[bool] = True,
                 device: _DeviceType = None,
                 **data_loader_params):
        r"""
        Implements 'normal' `ICAL <https://arxiv.org/abs/2002.07916>`_. :math:`R` points
        are randomly drawn from the pool and the average of the candidate batch's kernels
        is used instead. Thus, the dependency measure reduces to :math:`d = 2`.

        .. math::

            \frac{1}{|\mathcal{R}|} d\text{HSIC}(\displaystyle\sum_{x'\in\mathcal{R}} k^{x'},
            \frac{1}{B} \displaystyle\sum_{i = 1}^{B} k^{x_i})

        .. code:: python

            model = MCDropout(...)
            ical = ICAL(model, device=device,
                        batch_size=512,
                        pin_memory=True, num_workers=2)
            ical(X_pool, b=10)

        :param model: A :class:`MCDropout` model is required to calculate
                      the `m` different samples of :math:`p(y | \omega^{(m)})`.
        :type model: :class:`MCDropout`
        :param kernel_fn: Kernel function, see static methods of :class:`ICAL`. Defaults to
            weighted a rational quadratic kernel. This is the default kernel in the paper.
        :type kernel_fn: Callable[[torch.Tensor]. torch.Tensor]], optional
        :param subset: Normal ICAL uses a subset of `X_pool`. `subset` specifies the
                  size of this subset (:math:`|\mathcal{R}|` in the paper).
                  Use -1 to denote the entire pool.
        :type subset: int
        :param greedy_acquire: how many points to acquire at once in each acquisition step.
        :type greedy_acquire: int
        :param use_one_hot: use one_hot_encoding when calculating kernel matrix. This is the
            default behaviour in the paper.
        :type use_one_hot: bool
        :param sample_softmax: sample the softmax probabilities. If this is `True`, then
            `use_one_hot` is automatically overriden to be `True`. This is the default
            behaviour in the paper.
        :type sample_softmax: bool
        :param device: Device object. Used when making forward passes with the model.
        :type device: `None`, `str`, `torch.device`
        :param data_loader_params: params to be passed into `DataLoader` when
                                   iterating over `X_pool`.

        .. warning::
            Do not set `shuffle=True` in `data_loader_params`! The indices will be
            incorrect if the `DataLoader` object shuffles `X_pool`!
        """
        self._r = subset
        self._model = model
        self._dl_params = data_loader_params
        self._device = device
        self._l = greedy_acquire
        self._use_oh = True if sample_softmax else use_one_hot
        self._sample_softmax = sample_softmax
        if kernel_fn is None:
            self._kernel = ICAL.rational_quadratic()
        else:
            self._kernel = kernel_fn
        assert not self._dl_params.get('shuffle', False)
        assert subset != 0

    def __call__(self, X_pool: torch.Tensor, b: int) -> np.array:
        model = self._model
        l = self._l
        pool_size = X_pool.size(0)
        r = self._r if self._r != -1 else pool_size
        dl = torch.utils.data.DataLoader(ALRDataset(X_pool), **self._dl_params)
        with torch.no_grad():
            mc_preds = torch.cat([
                model.stochastic_forward(x.to(self._device) if self._device else x) for x in dl
            ], dim=1)
        mc_preds = mc_preds.detach_()
        assert mc_preds.size()[:-1] == (model.n_forward, pool_size)
        # num classes
        C = mc_preds.size(-1)
        if self._sample_softmax:
            assert self._use_oh
            cat_dist = dist.categorical.Categorical(mc_preds.view(model.n_forward * pool_size, -1))
            # mc_preds is now a vector of sampled class idx
            mc_preds = cat_dist.sample([1])[0]
            assert mc_preds.size() == (model.n_forward * pool_size,)
        if self._use_oh:
            if not self._sample_softmax:
                mc_preds = mc_preds.view(model.n_forward * pool_size, -1).argmax(dim=-1)
            assert mc_preds.size() == (model.n_forward * pool_size,)
            mc_preds = (torch.eye(C)[mc_preds]                      # shape [N * B x C]
                             .view(model.n_forward, pool_size, C))  # shape [N x B x C]
        assert mc_preds.size() == (model.n_forward, pool_size, C)
        kernel_matrices = self._kernel(mc_preds)
        assert kernel_matrices.size() == (model.n_forward, model.n_forward, pool_size)
        # [Pool_size x N x N]
        kernel_matrices = kernel_matrices.permute(2, 0, 1)
        # indices of points current in batch (a possible maximum of b by the
        # end of the iteration)
        batch_idxs = []

        while len(batch_idxs) < b:
            # always re-sample subset (what if we don't?)
            random_subset = np.random.choice(pool_size, size=r, replace=False)
            # a la theorem 2 - it suggested sum but we're using mean here - shouldn't make a difference
            pool_kernel = kernel_matrices[random_subset].mean(0)  # [N x N]
            # normal ICAL uses average batch kernels
            batch_kernels = (kernel_matrices + kernel_matrices[batch_idxs].sum(0, keepdim=True))\
                            / (len(batch_idxs) + 1)  # [Pool_size x N x N]
            scores = self._dHSIC(
                torch.cat([
                    # TODO: can remove repeat?: potentially expensive!
                    pool_kernel.unsqueeze(0).repeat(batch_kernels.size(0), 1, 1).unsqueeze(-1),
                    batch_kernels.unsqueeze(-1)
                ], dim=-1)  # [Pool_size x N x N x 2]
            )
            assert scores.size() == (pool_size,)
            # mask chosen indices
            scores[batch_idxs] = -np.inf
            # greedily take top l scores
            idxs = torch.argsort(scores, descending=True)
            for idx in idxs[:l]: batch_idxs.append(idx.item())
        # greedily taking top l might sometimes acquire extra points if
        # b is not divisible by l, hence, truncate the output
        return np.array(batch_idxs[:b])

    @staticmethod
    def rational_quadratic(alphas: Optional[Sequence[float]] = (.2, .5, 1, 2, 5),
                           weights: Optional[Sequence[float]] = None) -> Callable:
        def _rational_quadratic(x: torch.Tensor) -> torch.Tensor:
            """
            :param x: tensor of shape [N x M x C]
            :return: tensor of shape [N x N x M]
            """
            N, M, _ = x.size()
            _alphas = x.new_tensor(alphas).view(-1, 1, 1, 1)
            if weights:
                _weights = x.new_tensor(weights)
            else:
                _weights = x.new_tensor(1.0 / _alphas.size(0)).repeat(_alphas.size(0))
            assert _weights.size(0) == _alphas.size(0)
            distances = (x.unsqueeze(0) - x.unsqueeze(1)).pow_(2).sum(-1)
            assert distances.size() == (N, N, M)

            distances = distances.unsqueeze_(0)   # 1 N N M
            # TODO: is logspace really necessary?
            log = torch.log1p(distances / (2 * _alphas))
            assert torch.isfinite(log).all()
            res = torch.einsum('w,wijk->ijk', _weights, torch.exp(-_alphas * log))
            assert torch.isfinite(res).all()
            return res
        return _rational_quadratic

    @staticmethod
    def _dHSIC(x: torch.Tensor) -> torch.Tensor:
        r"""
        Computes HSIC for d-variables in a batch of size :math:`K`.

        .. note::

            While the values are computed in logspace for numerical stability,
            the returned value is casted back to its original space.

        :param x: tensor of shape :math:`K \times N \times N \times D` where:

            * :math:`K` is the batch size
            * :math:`N` is the number of samples in each variable
            * :math:`D` is the number of variables
        :return: dHSIC scores, a tensor of shape :math:`K`.
        """
        K, N, N2, D = x.size()
        assert N == N2
        # trivial case, definition 2.6 https://arxiv.org/pdf/1603.00285.pdf
        if N < 2 * D:
            warnings.warn(f"The number of samples is lesser than twice "
                          f"the number of variables in dHISC. Trivial "
                          f"case of 0; this may or may not be intended.")
            return x.new_zeros(size=(K,))
        # https://github.com/NiklasPfister/dHSIC/blob/master/dHSIC/R/dhsic.R
        # logspace
        x = torch.log(x)
        logn = np.log(N)
        term1 = torch.sum(x, dim=-1).logsumexp(dim=(1, 2)) - 2 * logn
        term2 = torch.logsumexp(x, dim=(1, 2)).sum(dim=-1) - (2 * D * logn)
        term3 = (torch.logsumexp(x, dim=1)
                      .sum(dim=-1)
                      .logsumexp(dim=-1) + np.log(2) - (D + 1) * logn)
        assert term1.size() == term2.size() == term3.size() == (K,)
        # subtract max for numerical stabilisation
        term_max = torch.stack([term1, term2, term3], dim=0).max(dim=0)[0]
        assert term_max.size() == (K,)
        res = (term1 - term_max).exp_() + (term2 - term_max).exp_() - (term3 - term_max).exp_()
        res *= term_max.exp_()
        assert torch.isfinite(res).all()
        return res
