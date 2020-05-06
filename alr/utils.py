import inspect
import types
from collections import namedtuple
from contextlib import contextmanager, AbstractContextManager
from dataclasses import dataclass
from functools import wraps
from timeit import default_timer
from typing import Optional, Callable, Union, Tuple, List

import numpy as np
import torch
import torch.utils.data as torchdata
from tqdm.auto import tqdm, trange

# type aliases
_DeviceType = Optional[Union[str, torch.device]]
_ActiveLearningDataset = namedtuple('ActiveLearningDataset', 'training unlabelled')


def range_progress_bar(*args, **kwargs):
    return trange(*args, **kwargs)


def progress_bar(*args, **kwargs):
    return tqdm(*args, **kwargs)


@dataclass
class Elapsed:
    """Elapsed time in seconds"""
    seconds: Optional[float] = None

    @property
    def minutes(self):
        r"""Elapsed time in minutes"""
        if self.seconds is None:
            return None
        return self.seconds / 60

    @property
    def hours(self):
        r"""Elapsed time in hours"""
        if self.seconds is None:
            return None
        return self.minutes / 60

    @property
    def days(self):
        r"""Elapsed time in days"""
        if self.seconds is None:
            return None
        return self.hours / 24


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


class Time(AbstractContextManager):
    def __init__(self, fn: Callable):
        r"""
        Given a callable, keeps track of the execution time every time it is invoked.

        .. code:: python

            class Foo:
                def m(self):
                    time.sleep(1)
                    return 42
            f = Foo()

            # start tracking f.m
            with Time(f.m) as t:
                # do something with f.m
                f.m()
            # stop tracking f.m

            print(t.tape)

        :param fn: callable to track. Can be method or a function but not a nested or lambda function.
        :type fn: Callable

        .. warning::
            The callable cannot be a lambda or nested function!
        """
        self._tape: List[Elapsed] = []
        self._fn = fn
        self._namespace = None
        self._register()

    def _register(self):
        fn = self._fn
        if inspect.ismethod(fn):
            self._namespace = fn.__self__
            assert hasattr(self._namespace, fn.__name__)
            setattr(self._namespace, fn.__name__,
                    types.MethodType(self._timer(fn, is_method=True), fn.__self__))
        elif inspect.isfunction(fn):
            # parent module
            mod = inspect.getmodule(inspect.stack()[2][0])
            if hasattr(mod, fn.__name__):
                self._namespace = mod
            elif self._try_static(mod, fn):
                self._namespace = self._try_static(mod, fn)
            else:
                raise ValueError(f"Can't find {fn} in module level or class level.")
            setattr(self._namespace, fn.__name__, self._timer(fn, is_method=False))
        else:
            raise ValueError(f"{fn} is not a method or a function.")

    def _try_static(self, mod, fn: Callable):
        classname = fn.__qualname__.split(".")[0]
        if hasattr(mod, classname):
            return getattr(mod, classname)
        return None

    def deregister(self) -> None:
        r"""
        Stop tracking the registered function. This is automatically invoked if
        :class:`Time` was constructed in a `with` block.

        :return: None
        :rtype: NoneType
        """
        setattr(self._namespace, self._fn.__name__, self._fn)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.deregister()

    def _timer(self, fn: Callable, is_method: bool):
        @wraps(fn)
        def _t(*args, **kwargs):
            if is_method:
                # don't pass self to fn, since fn is already bound to self
                args = args[1:]
            with timeop() as t:
                res = fn(*args, **kwargs)
            self._tape.append(t)
            return res

        return _t

    def reset(self) -> None:
        r"""
        Empties the tape

        :return: None
        :rtype: NoneType
        """
        self._tape = []

    @property
    def tape(self) -> List[Elapsed]:
        r"""
        Returns a list of :class:`Elapsed` objects denoting the time each invocation took.
        The length of this list is equal to the number of times the function was invoked while
        registered.

        :return: list of elapsed time equal in length to the number
                    of times the function was invoked whilst registered.
        :rtype: List[Elapsed]
        """
        return self._tape


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
    :return: (training pool, unlabelled pool)
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
    return _ActiveLearningDataset(training=torchdata.Subset(ds, sampled_idxs),
                                  unlabelled=torchdata.Subset(ds, list(original_idxs - set(sampled_idxs))))
