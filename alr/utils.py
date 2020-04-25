from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from timeit import default_timer
from typing import Optional, Callable, Union, Tuple
from collections import namedtuple

import torch
import torch.utils.data as torchdata
import numpy as np

# type aliases
_DeviceType = Optional[Union[str, torch.device]]
_ActiveLearningDataset = namedtuple('ActiveLearningDataset', 'unlabelled training')

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


def stratified_partition(ds: torchdata.Dataset, classes: int, size: int) \
        -> Tuple[torchdata.Dataset, torchdata.Dataset]:
    r"""
    Partitions `ds` into training pool and a faux unlabelled pool. The "unlabelled"
    pool will contain `len(ds) - size` data points and the training pool will contain
    `size` data points where the target class is as balanced as possible. Note,
    the faux "unlabelled" pool will contain target labels since it's
    derived from the training pool.

    :param ds: dataset containing input features and target class
    :type ds: :class:`torch.utils.data.Dataset`
    :param classes: number of target classes contained in `ds`
    :type classes: int
    :param size: size of resulting training pool
    :type size: int
    :return: (unlabelled pool, training pool)
    :rtype: tuple
    """
    assert size < len(ds)
    c = size // classes
    extra = size % classes
    count = {cls: c for cls in range(classes)}
    original_idxs = set(range(len(ds)))
    sampled_idxs = []
    # the first `extra` classes gets the extra counts
    while extra:
        count[extra] += 1
        extra -= 1
    for idx in np.random.permutation(len(ds)):
        if all(i == 0 for i in count.values()):
            break
        y = ds[idx][1]
        if count[y]:
            count[y] -= 1
            sampled_idxs.append(idx)
    return _ActiveLearningDataset(unlabelled=torchdata.Subset(ds, list(original_idxs - set(sampled_idxs))),
                                  training=torchdata.Subset(ds, sampled_idxs))
