from dataclasses import dataclass
from functools import wraps
from typing import Optional, Callable
from contextlib import contextmanager
from timeit import default_timer


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
