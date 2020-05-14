import time
import inspect
import pytest

from alr.utils import *


def test_timeop_normal():
    with timeop() as t:
        time.sleep(1)

    assert t.seconds >= 0


def test_timeop_exception():
    with pytest.raises(ValueError):
        with timeop() as t:
            try:
                raise ValueError
            finally:
                assert t.seconds is None


class Foo:
    def __init__(self, x):
        self.x = x

    def m(self, n):
        """
        test docs
        :param n: ok
        :type n: ok
        :return: ok
        :rtype: ok
        """
        time.sleep(0.5)
        return self.x * n

    @staticmethod
    def s(n):
        return n * 2

    @classmethod
    def c(cls, n):
        return cls(n)


def test_method_time():
    f = Foo(2)
    assert inspect.ismethod(f.m)
    doc = f.m.__doc__
    name = f.m.__name__
    assert f.m(32) == 32 * 2
    with Time(f.m) as t:
        assert inspect.ismethod(f.m)
        assert f.m.__doc__ == doc and f.m.__name__ == name
        f.x = 234
        assert f.m(32) == 32 * 234
        assert len(t.tape) == 1 and t.tape[0].seconds >= 0
        f.x = 123
        assert f.m(23) == 23 * 123
        assert len(t.tape) == 2 and t.tape[-1].seconds >= 0

    assert inspect.ismethod(f.m)
    assert(f.m(32) == 32 * 123)
    assert len(t.tape) == 2
    t.reset()
    assert len(t.tape) == 0


def test_static_time():
    t = Time(Foo.s)
    f = Foo.s(99)
    assert len(t.tape) == 1 and t.tape[0].seconds >= 0
    assert f == 99 * 2
    # stop tracking
    t.deregister()
    f = Foo.s(77)
    assert f == 77 * 2
    assert len(t.tape) == 1


def test_cls_time():
    t = Time(Foo.c)
    f = Foo.c(99)
    assert len(t.tape) == 1 and t.tape[0].seconds >= 0
    assert f.m(88) == 99 * 88
    # stop tracking
    t.deregister()
    f = Foo.c(77)
    assert f.m(66) == 77 * 66
    assert len(t.tape) == 1


def foobar():
    return


def test_func_time():
    with Time(foobar) as t:
        res = foobar()
        assert res is None
    assert len(t.tape) == 1
