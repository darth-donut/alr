import torch.utils.data as torchdata
import torchvision
from torch import nn
from torch.nn import functional as F
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

    def get(self, root: Optional[str] = 'data') -> Tuple[torchdata.Dataset, torchdata.Dataset]:
        r"""
        Return (train, test) tuple of datasets.

        Args:
            root (str, optional): root path where data will be read from or downloaded to

        Returns:
            tuple: a 2-tuple of (train, test) datasets
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

        Returns:
            tuple: a 2-tuple of mean and standard deviation
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
        }
        return params[self]

    def get_fixed(self, root: Optional[str] = 'data', which: Optional[int] = 0) -> Tuple[torchdata.Dataset, torchdata.Dataset, torchdata.Dataset]:
        r"""
        Returns a fixed train, pool, and test datasets. This is only used for experiments.

        Args:
            root (str, optional): root path where data will be read from or downloaded to.
            which (int, optional): there are multiple possible sets of fixed points for a given dataset.
                This argument specifies which of the multiple possible ones to choose from.

        Returns:
            tuple: A tuple of train, pool, and test datasets.
        """
        if self is not Dataset.MNIST:
            raise NotImplementedError(f"Fixed points for {self} is not available yet.")

        train, test = self.get(root)
        assert which < len(_mnist_20), f"Only {len(_mnist_20)} sets are available for {self}."
        idxs = _mnist_20[which]
        cidxs = set(range(len(train))) - set(idxs)
        pool = torchdata.Subset(train, list(cidxs))
        train = torchdata.Subset(train, idxs)
        return train, pool, test

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
    def __init__(self):
        # architecture (almost) similar to https://keras.io/examples/cifar10_cnn/
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.drop1 = nn.Dropout()
        # 30x30
        # max pool => 15x15
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.drop2 = nn.Dropout()
        # 14x14
        # max pool => 7x7
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, 10)
        self.drop3 = nn.Dropout()

    def forward(self, x):
        x = F.relu(self.conv2(F.relu(self.conv1(x))))
        x = self.drop1(F.max_pool2d(x, 2))

        x = F.relu(self.conv4(F.relu(self.conv3(x))))
        x = self.drop2(F.max_pool2d(x, 2))

        x = x.view(-1, 7 * 7 * 64)
        x = self.drop3(F.relu(self.fc1(x)))

        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

