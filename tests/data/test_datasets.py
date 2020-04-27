import pytest
from alr.data.datasets import Dataset


def test_MNIST():
    train, test = Dataset.MNIST.get()
    assert len(train) == 60_000 and len(test) == 10_000


def test_FashionMNIST():
    train, test = Dataset.FashionMNIST.get()
    assert len(train) == 60_000 and len(test) == 10_000


# todo(test)
@pytest.mark.skip(reason="HTTP 500")
def test_EMNISTMerge():
    train, test = Dataset.EMNISTMerge.get()
    assert len(train) == 697_932 and len(test) == 116_323


# todo(test)
@pytest.mark.skip(reason="HTTP 500")
def test_EMINISTBalanced():
    train, test = Dataset.EMNISTMerge.get()
    assert len(train) == 112_800 and len(test) == 18_800


def test_CIFAR10():
    train, test = Dataset.CIFAR10.get()
    assert len(train) == 50_000 and len(test) == 10_000


def test_CIFAR100():
    train, test = Dataset.CIFAR100.get()
    assert len(train) == 50_000 and len(test) == 10_000


def test_RepeatedMNIST():
    train, test = Dataset.RepeatedMNIST.get()
    assert len(train) == 60_000 * 3 and len(test) == 10_000
