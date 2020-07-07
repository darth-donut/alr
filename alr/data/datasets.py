import torch.utils.data as torchdata
import torchvision as tv
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from torch.nn.utils import weight_norm

from enum import Enum
from typing import Optional, Tuple
from torchvision import transforms
from pathlib import Path


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
        >>> train, test = Dataset.MNIST.get()
        >>> train_load = torch.utils.data.DataLoader(train, batch_size=32)
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
    CINIC10 = "CINIC10"

    def get(self,
            root: Optional[str] = 'data',
            raw: Optional[bool] = False,
            augmentation: Optional[bool] = False) -> Tuple[torchdata.Dataset, torchdata.Dataset]:
        r"""
        Return (train, test) tuple of datasets.

        Args:
            root (str, optional): root path where data will be read from or downloaded to
            raw (bool, optional): if `True`, then training set will not be transformed (i.e.
                no normalisation, ToTensor, etc.); note, the test set *WILL* be transformed.
            augmentation (bool, optional): whether to add standard augmentation: horizontal flips and
                random cropping.

        Returns:
            tuple: a 2-tuple of (train, test) datasets
        """
        assert not raw or not augmentation, "Cannot enable augmentation on raw dataset!"

        regular_transform = [
            transforms.ToTensor(),
            transforms.Normalize(*self.normalisation_params)
        ]
        test_transform = transforms.Compose(regular_transform)
        standard_augmentation = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if augmentation:
            regular_transform = standard_augmentation + regular_transform
        train_transform = transforms.Compose(regular_transform)
        if raw:
            train_params = dict(root=root, train=True, download=True)
        else:
            train_params = dict(root=root, transform=train_transform, train=True, download=True)
        test_params = dict(root=root, transform=test_transform, train=False, download=True)
        if self in {Dataset.MNIST,
                    Dataset.FashionMNIST,
                    Dataset.CIFAR10,
                    Dataset.CIFAR100}:
            train = getattr(tv.datasets, self.value)(**train_params)
            test = getattr(tv.datasets, self.value)(**test_params)
        elif self in {Dataset.EMNISTBalanced, Dataset.EMNISTMerge}:
            split = 'balanced' if self is Dataset.EMNISTBalanced else 'bymerge'
            train = tv.datasets.EMNIST(**train_params, split=split)
            test = tv.datasets.EMNIST(**test_params, split=split)
        elif self is Dataset.RepeatedMNIST:
            train = tv.datasets.MNIST(**train_params)
            test = tv.datasets.MNIST(**test_params)
            train = torchdata.ConcatDataset([train] * 3)
        elif self is Dataset.CINIC10:
            from alr.data._cinic_indices import _cinic_test_indices, _cinic_train_indices
            if raw:
                train_transform = None
            cinic_root = Path().home() / "data" / "cinic-10"
            train = tv.datasets.ImageFolder(
                str(cinic_root / 'train'),
                transform=train_transform,
            )
            valid = tv.datasets.ImageFolder(
                str(cinic_root / 'valid'),
                transform=train_transform,
            )
            test = tv.datasets.ImageFolder(
                str(cinic_root / 'test'),
                transform=test_transform,
            )
            train = torchdata.ConcatDataset((train, valid))
            train = torchdata.Subset(train, _cinic_train_indices)
            test = torchdata.Subset(test, _cinic_test_indices)
        else:
            raise ValueError(f"{self} dataset hasn't been implemented.")
        return train, test

    @property
    def normalisation_params(self) -> Tuple[Tuple[float], Tuple[float]]:
        r"""
        Returns a tuple of channel mean and standard deviation of 0-1-scaled inputs.
        I.e. the input is assumed to be in the range of 0-1.

        Returns:
            tuple: a 2-tuple of mean and standard deviation
        """
        params = {
            Dataset.MNIST: ((0.1307,), (0.3081,)),
            Dataset.FashionMNIST: ((0.2860,), (0.3530,)),
            Dataset.EMNISTMerge: ((0.1736,), (0.3317,)),
            Dataset.EMNISTBalanced: ((0.1751,), (0.3332,)),
            Dataset.CIFAR10: ((0.49139968, 0.48215841, 0.44653091),
                              (0.2023, 0.1994, 0.2010)),
            Dataset.CIFAR100: ((0.50707516, 0.48654887, 0.44091784),
                               (0.26733429, 0.25643846, 0.27615047)),
            Dataset.CINIC10: ((0.47889522, 0.47227842, 0.43047404),
                              (0.24205776, 0.23828046, 0.25874835)),
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

        Returns:
            :class:`DataDescription`: information about this dataset
        """
        params = {
            Dataset.MNIST: DataDescription(10, 28, 28, 1),
            Dataset.RepeatedMNIST: DataDescription(10, 28, 28, 1),
            Dataset.FashionMNIST: DataDescription(10, 28, 28, 1),
            Dataset.EMNISTBalanced: DataDescription(47, 28, 28, 1),
            Dataset.EMNISTMerge: DataDescription(47, 28, 28, 1),
            Dataset.CIFAR10: DataDescription(10, 32, 32, 3),
            Dataset.CIFAR100: DataDescription(100, 32, 32, 3),
            Dataset.CINIC10: DataDescription(10, 32, 32, 3),
        }
        return params[self]

    def get_fixed(self, root: Optional[str] = 'data', which: Optional[int] = 0,
                  raw: Optional[bool] = False) -> Tuple[
        torchdata.Dataset, torchdata.Dataset, torchdata.Dataset]:
        r"""
        Returns a fixed train, pool, and test datasets. This is only used for experiments.

        Args:
            root (str, optional): root path where data will be read from or downloaded to.
            which (int, optional): there are multiple possible sets of fixed points for a given dataset.
                This argument specifies which of the multiple possible ones to choose from.
            raw (bool, optional): similar to :meth:`get`, train will not contain any
                transform whatsoever. (Test will still have ToTensor and Normalisation.)

        Returns:
            tuple: A tuple of train, pool, and test datasets.
        """
        if self is Dataset.MNIST:
            idx_set = _mnist_20
        elif self is Dataset.CIFAR10:
            idx_set = _cifar10_1
        else:
            raise NotImplementedError(f"Fixed points for {self} is not available yet.")

        train, test = self.get(root, raw=raw)
        assert which < len(idx_set), f"Only {len(idx_set)} sets are available for {self}."
        idxs = idx_set[which]
        cidxs = set(range(len(train))) - set(idxs)
        pool = torchdata.Subset(train, list(cidxs))
        train = torchdata.Subset(train, idxs)
        return train, pool, test

    @property
    def get_augmentation(self):
        if self is not Dataset.CIFAR10:
            raise NotImplementedError(f"get_augmentation not available for {self}")
        data_augmentation = tv.transforms.Compose([
            tv.transforms.Pad(2, padding_mode='reflect'),
            tv.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            tv.transforms.RandomCrop(32),
            tv.transforms.RandomHorizontalFlip(),
        ])
        return data_augmentation

    @property
    def model(self) -> nn.Module:
        r"""
        Returns a canonical model architecture for a given dataset.

        Returns:
            torch.nn.Module: a pytorch model
        """
        if self in {Dataset.MNIST, Dataset.RepeatedMNIST}:
            return MNISTNet()
        if self == Dataset.CIFAR10:
            return CIFAR10Net()
        raise NotImplementedError("No model defined for this dataset yet.")


# 24 sets of 20 points with +- 0.01 accuracy from median: 0.65205
_mnist_20 = [
    (33479, 3162, 19169, 11495, 15914, 20426, 21495, 52396, 16515, 19727, 52720, 12424, 30690, 58679, 44785,
     20682, 28513, 38932, 20408, 3801),
    (40920, 33826, 59132, 22914, 35401, 46080, 47822, 1925, 28716, 50867, 5126, 29159, 41470, 25508, 53476,
     27472, 44436, 44587, 56388, 19129),
    (19641, 4578, 29287, 37320, 21474, 31142, 20221, 49296, 35922, 32416, 28365, 38441, 26961, 22063, 23629,
     50925, 46201, 37567, 11815, 46561),
    (13506, 43404, 12223, 42233, 33552, 58117, 36527, 2797, 29225, 11150, 25582, 49413, 55733, 36569, 6531,
     50308, 9405, 34069, 16092, 41826),
    (46938, 45343, 10338, 6631, 24919, 32224, 14440, 52834, 21392, 40344, 31691, 43258, 25017, 56908, 41200,
     19552, 43623, 57612, 56061, 33019),
    (34380, 458, 27650, 18400, 36235, 34469, 31224, 52286, 22674, 49931, 5674, 18622, 2865, 30094, 37540,
     1887, 47609, 37123, 17887, 59419),
    (55333, 56236, 54764, 31705, 45729, 26454, 15116, 45512, 42629, 35753, 11879, 4471, 42841, 23479, 22760,
     1535, 30522, 32030, 6356, 31809),
    (654, 6458, 52794, 16987, 38175, 14182, 1679, 44590, 2654, 38630, 27540, 22150, 27289, 36348, 46448, 25692,
     2592, 43035, 11195, 19245),
    (38608, 28958, 49076, 55452, 43257, 38931, 28884, 52759, 41098, 28558, 46660, 59685, 34048, 51456, 19143,
     38580, 3850, 17198, 22749, 39503),
    (33674, 28825, 35042, 57790, 18797, 59202, 45838, 44119, 28229, 30357, 59087, 22074, 37914, 43639, 28235,
     59731, 2687, 1710, 16031, 37424),
    (37041, 32464, 2182, 5105, 25852, 42029, 15667, 53665, 52299, 19278, 29439, 21215, 12811, 20382, 50605,
     36311, 3196, 6964, 34140, 58381),
    (49580, 32583, 10860, 12029, 27952, 57306, 27114, 51904, 37445, 12358, 39175, 8044, 10086, 18826, 36491,
     27013, 53208, 49325, 55150, 50527),
    (34791, 43564, 16453, 18335, 19112, 18183, 17212, 473, 58744, 20163, 22546, 58391, 26952, 39042, 12006,
     48625, 26815, 49461, 6468, 6936),
    (47333, 32600, 7634, 15318, 3236, 43565, 34004, 47891, 52446, 5381, 27198, 56250, 44513, 57343, 6476, 27699,
     23440, 14554, 42538, 58241),
    (32861, 43028, 23876, 54561, 20624, 22584, 2156, 5675, 25557, 38187, 4675, 5643, 31785, 39365, 55789, 11507,
     50565, 14166, 46226, 2144),
    (52038, 47011, 35514, 36353, 13205, 26807, 37701, 24186, 22144, 8822, 39192, 30370, 42906, 19378, 9625,
     44845, 37137, 13356, 28077, 36932),
    (28931, 58414, 34981, 23698, 23096, 24403, 32018, 38366, 54223, 33457, 7647, 22917, 11600, 48807, 39192,
     47631, 16900, 15283, 14155, 55377),
    (49969, 31620, 56337, 19699, 49342, 12913, 43909, 145, 5575, 41365, 20196, 43690, 39055, 44785, 33422, 2819,
     14811, 43261, 45203, 39170),
    (52645, 41154, 43574, 26144, 17243, 51196, 21418, 21816, 54635, 13619, 2874, 17124, 16391, 45504, 55157,
     13527, 33756, 45948, 21693, 3374),
    (36700, 12636, 35933, 9290, 975, 42757, 5197, 41484, 11101, 10798, 19309, 4748, 38047, 34424, 42575, 38653,
     43514, 36621, 35862, 28877),
    (20398, 27976, 25154, 54951, 18249, 20, 55911, 55500, 186, 38592, 48834, 4119, 11926, 25099, 54824, 48339,
     43320, 3754, 24752, 11457),
    (46959, 13392, 8626, 55276, 26976, 33992, 16264, 44518, 30741, 39375, 34387, 4537, 12291, 6658, 20542, 18832,
     44508, 20867, 30517, 37982),
    (26754, 6166, 16478, 1561, 59790, 9000, 43538, 4868, 34394, 21017, 37970, 14324, 46481, 52564, 40462, 50910,
     48934, 2070, 3811, 21865),
    (13369, 54382, 20231, 14627, 43491, 15178, 2253, 14073, 31816, 1870, 34302, 5359, 36903, 41308, 45210, 50448,
     21174, 57606, 22846, 54399)
]

_cifar10_1 = [
    (33553, 9427, 199, 12447, 39489, 42724, 10822, 49498, 36958, 43106, 38695, 1414, 18471, 15118, 13466, 26497, 24148,
     41514, 30263, 24712)
]


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        # 32 24 24
        self.dropout1 = nn.Dropout2d()
        # maxpool --
        # 32 12 12
        self.conv2 = nn.Conv2d(32, 64, 5)
        # 64 8 8
        self.dropout2 = nn.Dropout2d()
        # maxpool --
        # 64 4 4
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.dropout3 = nn.Dropout()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(self.dropout1(F.relu(self.conv1(x))), 2)
        x = F.max_pool2d(self.dropout2(F.relu(self.conv2(x))), 2)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc2(self.dropout3(F.relu(self.fc1(x))))
        return F.log_softmax(x, dim=1)


class CIFAR10Net(nn.Module):
    """
    CNN from Mean Teacher paper
    # taken from: https://github.com/EricArazo/PseudoLabeling/blob/2fbbbd3ca648cae453e3659e2e2ed44f71be5906/utils_pseudoLab/ssl_networks.py
    """

    def __init__(self, num_classes=10, drop_prob=.5):
        super(CIFAR10Net, self).__init__()
        self.activation = nn.LeakyReLU(0.1)
        self.conv1a = weight_norm(nn.Conv2d(3, 128, 3, padding=1))
        self.bn1a = nn.BatchNorm2d(128)
        self.conv1b = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1b = nn.BatchNorm2d(128)
        self.conv1c = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1c = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop = nn.Dropout(drop_prob)

        self.conv2a = weight_norm(nn.Conv2d(128, 256, 3, padding=1))
        self.bn2a = nn.BatchNorm2d(256)
        self.conv2b = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2b = nn.BatchNorm2d(256)
        self.conv2c = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2c = nn.BatchNorm2d(256)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)

        self.conv3a = weight_norm(nn.Conv2d(256, 512, 3, padding=0))
        self.bn3a = nn.BatchNorm2d(512)
        self.conv3b = weight_norm(nn.Conv2d(512, 256, 1, padding=0))
        self.bn3b = nn.BatchNorm2d(256)
        self.conv3c = weight_norm(nn.Conv2d(256, 128, 1, padding=0))
        self.bn3c = nn.BatchNorm2d(128)
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)

        self.fc1 = weight_norm(nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.activation(self.bn1a(self.conv1a(x)))
        x = self.activation(self.bn1b(self.conv1b(x)))
        x = self.activation(self.bn1c(self.conv1c(x)))
        x = self.mp1(x)
        x = self.drop(x)

        x = self.activation(self.bn2a(self.conv2a(x)))
        x = self.activation(self.bn2b(self.conv2b(x)))
        x = self.activation(self.bn2c(self.conv2c(x)))
        x = self.mp2(x)
        x = self.drop(x)

        x = self.activation(self.bn3a(self.conv3a(x)))
        x = self.activation(self.bn3b(self.conv3b(x)))
        x = self.activation(self.bn3c(self.conv3c(x)))
        x = self.ap3(x)
        x = x.view(-1, 128)
        return F.log_softmax(self.fc1(x), dim=-1)
