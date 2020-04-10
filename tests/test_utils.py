import time
import pytest

from alr.utils import *


def test_timeop_normal():
    with timeop() as t:
        time.sleep(3)

    assert abs(t.seconds - 3) <= 1e-1


def test_timeop_exception():
    with pytest.raises(ValueError):
        with timeop() as t:
            try:
                raise ValueError
            finally:
                assert t.seconds is None
