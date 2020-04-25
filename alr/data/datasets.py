import torch.utils.data as torchdata
import torchvision
from dataclasses import dataclass

from enum import Enum
from typing import Optional, Tuple
from torchvision import transforms


@dataclass(frozen=True)
class DataDescription:
    """
    Describes the attributes of this dataset.
    """
    n_class: int
    width: int
    height: int
    channels: int


class Dataset(Enum):
    r"""
    An enum class that provides convenient data retrieval.

    Example:

    .. code:: python

        train, test = Dataset.MNIST.get()
        train_load = torch.utils.data.DataLoader(train, batch_size=32)
    """
    MNIST = "MNIST"
    FashionMNIST = "FashionMNIST"
    # https://arxiv.org/pdf/1702.05373v1.pdf (Cohen et. al 2017)
    # byMerge: a 47-class classification task.
    # The merged classes, as suggested by
    # the NIST, are for the letters C, I, J, K, L, M, O, P, S, U,
    # V, W, X, Y and Z
    EMNISTMerge = "EMNISTMerge"
    # subset of bymerge
    EMNISTBalanced = "EMNISTBalanced"
    CIFAR10 = "CIFAR10"
    CIFAR100 = "CIFAR100"
    RepeatedMNIST = "RepeatedMNIST"

    def get(self, root: Optional[str] = 'data') -> Tuple[torchdata.Dataset, torchdata.Dataset]:
        r"""
        Return (train, test) tuple of datasets.

        :param root: root path where data will be read from or downloaded to
        :type root: str, optional
        :return: a 2-tuple of (train, test) datasets
        :rtype: tuple
        """
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*self.normalisation_params)
        ])
        test_transform = train_transform
        train_params = dict(root=root, transform=train_transform, train=True, download=True)
        test_params = dict(root=root, transform=test_transform, train=False, download=True)
        if self in {Dataset.MNIST,
                    Dataset.FashionMNIST,
                    Dataset.CIFAR10,
                    Dataset.CIFAR100}:
            train = getattr(torchvision.datasets, self.value)(**train_params)
            test = getattr(torchvision.datasets, self.value)(**test_params)
        elif self in {Dataset.EMNISTBalanced, Dataset.EMNISTMerge}:
            split = 'balanced' if self is Dataset.EMNISTBalanced else 'bymerge'
            train = torchvision.datasets.EMNIST(**train_params, split=split)
            test = torchvision.datasets.EMNIST(**test_params, split=split)
        elif self is Dataset.RepeatedMNIST:
            train = torchvision.datasets.MNIST(**train_params)
            test = torchvision.datasets.MNIST(**test_params)
            train = torchdata.ConcatDataset([train] * 3)
        else:
            raise ValueError(f"{self} dataset hasn't been implemented.")
        return train, test

    @property
    def normalisation_params(self) -> Tuple[Tuple[float], Tuple[float]]:
        r"""
        Returns a tuple of channel mean and standard deviation of 0-1-scaled inputs.
        I.e. the input is assumed to be in the range of 0-1.

        :return: a 2-tuple of mean and standard deviation
        :rtype: `tuple`
        """
        params = {
            Dataset.MNIST: ((0.1307,), (0.3081,)),
            Dataset.FashionMNIST: ((0.2860,), (0.3530,)),
            Dataset.EMNISTMerge: ((0.1736,), (0.3317,)),
            Dataset.EMNISTBalanced: ((0.1751,), (0.3332,)),
            Dataset.CIFAR10: ((0.49139968, 0.48215841, 0.44653091),
                              (0.24703223, 0.24348513, 0.26158784)),
            Dataset.CIFAR100: ((0.50707516, 0.48654887, 0.44091784),
                               (0.26733429, 0.25643846, 0.27615047)),
        }
        params[Dataset.RepeatedMNIST] = params[Dataset.MNIST]
        return params[self]

    @property
    def about(self) -> DataDescription:
        r"""
        Returns information about this dataset including:
            * n_class
            * width
            * height
            * channels

        See :class:`DataDescription`.

        :return: information about this dataset
        :rtype: :class:`DataDescription`
        """
        params = {
            Dataset.MNIST: DataDescription(10, 28, 28, 1),
            Dataset.RepeatedMNIST: DataDescription(10, 28, 28, 1),
            Dataset.FashionMNIST: DataDescription(10, 28, 28, 1),
            Dataset.EMNISTBalanced: DataDescription(47, 28, 28, 1),
            Dataset.EMNISTMerge: DataDescription(47, 28, 28, 1),
            Dataset.CIFAR10: DataDescription(10, 32, 32, 3),
            Dataset.CIFAR100: DataDescription(100, 32, 32, 3),
        }
        return params[self]
