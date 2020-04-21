from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from timeit import default_timer
from typing import Optional, Callable, Union, Tuple

import torch
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# type aliases
_DeviceType = Optional[Union[str, torch.device]]


@contextmanager
def timeop():
    r"""
    Context manager for timing expressions in a `with` block:

    .. code:: python

        with timeop() as t:
            import time; time.sleep(2)
        assert abs(t.seconds - 2) < 1e-1

    :return: Context manager object

    .. note::

        if the expression in the `with` block raises an exception,
        `t.seconds is None`.
    """
    @dataclass
    class Elapsed:
        seconds: Optional[float] = None
    t = Elapsed()
    tick = default_timer()
    yield t
    tock = default_timer()
    t.seconds = tock - tick


def time_this(func: Callable):
    r"""
    A decorator to time functions. The result of `func` is returned
    in the first element of the tuple and the elapsed time in the second.

    .. code:: python

        @time_this
        def foo(x):
            return x
        x, elapsed = foo(42)
        assert x == 42 and elapsed.seconds >= 0

    :param func: any function
    :return: 2-tuple of (result, elapsed time)
    """
    @wraps(func)
    def dec(*args, **kwargs):
        with timeop() as t:
            res = func(*args, **kwargs)
        return res, t
    return dec


def stratified_partition(X_train: np.array, y_train: np.array, train_size: Optional[int] = 20) -> \
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

# def run_experiment(model: ALRModel, acquisition_function: AcquisitionFunction, X_train: torch.Tensor,
#                    y_train: torch.Tensor, X_pool: torch.Tensor, y_pool: torch.Tensor,
#                    val_data_loader: torch.utils.data.DataLoader, optimiser: torch.optim.Optimizer,
#                    *, b: Optional[int] = 10, iters: Optional[int] = 20,
#                    init_epochs: Optional[int] = 75, epochs: Optional[int] = 50,
#                    device: _DeviceType = None, pin_memory: Optional[bool] = True,
#                    num_workers: Optional[int] = 2, batch_size: Optional[int] = 32) -> Dict[(int, float)]:
#     r"""
#     A helper function useful for running a simple train-evaluate-acquire active learning loop.
#
#     :param model: an :class:`ALRModel` object
#     :type model: :class:`ALRModel`
#     :param acquisition_function: an :class:`AcquisitionFunction` object
#     :type acquisition_function: :class:`AcquisitionFunction`
#     :param X_train: A tensor with shape :math:`N \times C \times H \times W`
#     :type X_train: `torch.Tensor`
#     :param y_train: A tensor with shape :math:`N`
#     :type y_train: `torch.Tensor`
#     :param X_pool: A tensor with shape :math:`N' \times C \times H \times W`
#     :type X_pool: `torch.Tensor`
#     :param y_pool: A tensor with shape :math:`N'`
#     :type y_pool: `torch.Tensor`
#     :param val_data_loader: data loader used to calculate test/validation loss/acc
#     :type val_data_loader: `torch.utils.data.DataLoader`
#     :param optimiser: PyTorch optimiser object
#     :type optimiser: `torch.optim.Optimiser`
#     :param b: number of samples to acquire in each iteration
#     :type b: int, optional
#     :param iters: number of iterations to repeat acquisition from X_pool
#     :type iters: int, optional
#     :param init_epochs: number of initial epochs to train
#     :type init_epochs: int, optional
#     :param epochs: number of epochs for each subsequent training iterations
#     :type epochs: int, optional
#     :param device: Device object. Used for training and evaluation.
#     :type device: `None`, `str`, `torch.device`, optional
#     :param pin_memory: `pin_memory` argument passed to `DataLoader` object when training model
#     :type pin_memory: bool, optional
#     :param num_workers: `num_workers` argument passed to `DataLoader` object when training model
#     :type num_workers: int, optional
#     :param batch_size: `batch_size` argument passed to `DataLoader` object when training model
#     :type batch_size: int, optional
#     :return: a mapping of training pool size to accuracy.
#     :rtype: `dict`
#
#     :Example:
#
#     .. code:: python
#
#         # load training data
#         X_train, y_train = np.random.normal(0, 1, size=(100, 10)), np.random.randint(0, 5, size=(100,))
#         X_test, y_test = np.random.normal(0, 1, size=(10, 10)), np.random.randint(0, 5, size=(10,))
#
#         # partition data using stratified_partition
#         X_train, y_train, X_pool, y_pool = stratified_partition(X_train, y_train, train_size=20)
#
#         # create test DataLoader object
#         test_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test),
#                                                    batch_size=2048, pin_memory=True,
#                                                    shuffle=False, num_workers=2)
#
#         # instantiate model and optimiser.
#         class Net(nn.Module):
#             pass
#         model = MCDropout(Net()).to('cpu')
#         optimiser = torch.optim.Adam(model.parameters())
#
#         # optional
#         model.reset_weights()
#
#         afunction = RandomAcquisition()
#         history = run_experiment(model, afunction, X_train, y_train, X_pool, y_pool,
#                                  test_dataset, optimiser, b=30, iters=10, init_epochs=100,
#                                  epochs=100, device='cpu', batch_size=64,
#                                  pin_memory=True, num_workers=2)
#
#     """
#     assert X_train.dim() == X_pool.dim() == 4
#     assert y_train.size(0) == X_train.size(0)
#     assert y_pool.size(0) == X_pool.size(0)
#     X_train = X_train.clone()
#     y_train = y_train.clone()
#     X_pool = X_pool.clone()
#     y_pool = y_pool.clone()
#     dm = DataManager(acquisition_function, X_train, y_train, X_pool, y_pool,
#                      shuffle=True,
#                      num_workers=num_workers,
#                      pin_memory=pin_memory,
#                      batch_size=batch_size)
#     print(f"Commencing initial training with {dm.train_size} points")
#     _train(model, dataloader=dm.training_data, optimiser=optimiser, epochs=init_epochs,
#            device=device)
#     accs = {dm.train_size: _evaluate(model, val_data_loader, device=device)}
#     print(f"Accuracy = {accs[dm.train_size]}\n=====")
#     for i in range(iters):
#         dm.acquire(b=b)
#         print(f"Acquisition iteration {i + 1} ({(i + 1) / iters:.2%}), training size: {dm.train_size}")
#         model.reset_weights()
#         _train(model, dataloader=dm.training_data,
#                optimiser=optimiser, epochs=epochs, device=device)
#         accs[dm.train_size] = _evaluate(model, val_data_loader, device=device)
#         print(f"Accuracy = {accs[dm.train_size]}\n=====")
#     return accs
#
#
# def _train(model: nn.Module, dataloader: torch.utils.data.DataLoader, optimiser: torch.optim.Optimizer,
#            epochs: int = 50, device: _DeviceType = None) -> List[float]:
#     model.train()
#     criterion = nn.CrossEntropyLoss()
#     losses = []
#     tepochs = tqdm.trange(epochs, file=sys.stdout)
#     for _ in tepochs:
#         epoch_losses = []
#         for x, y in dataloader:
#             if device:
#                 x, y = x.to(device), y.to(device)
#             logits = model(x)
#             loss = criterion(logits, y)
#             epoch_losses.append(loss.item())
#             optimiser.zero_grad()
#             loss.backward()
#             optimiser.step()
#
#         losses.append(np.mean(epoch_losses))
#         tepochs.set_postfix(loss=losses[-1])
#     return losses
#
#
# def _evaluate(model: ALRModel,
#               dataloader: torch.utils.data.DataLoader,
#               device: _DeviceType = None) -> float:
#     model.eval()
#     score = total = 0
#     with torch.no_grad():
#         for x, y in dataloader:
#             if device:
#                 x, y = x.to(device), y.to(device)
#             _, preds = torch.max(model.predict(x), dim=1)
#             score += (preds == y).sum().item()
#             total += y.size(0)
#
#     return score / total
#
#
