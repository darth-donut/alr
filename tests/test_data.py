import numpy as np
import torch
import torch.utils.data as torchdata
import itertools

from alr.data import DataManager, UnlabelledDataset
from alr.acquisition import AcquisitionFunction


class MockAcquisitionFunction(AcquisitionFunction):
    """ return the first b points of X_pool"""

    def __call__(self, X_pool: torchdata.Dataset, b: int) -> np.array:
        return np.arange(b)


class DummyData(torchdata.Dataset):
    def __init__(self, n, target=None):
        self.x = torch.arange(n)
        self.target = target
        if target:
            self.y = torch.arange(n)

    def __getitem__(self, idx):
        if self.target:
            return self.x[idx], self.y[idx]
        return self.x[idx]

    def __len__(self):
        return len(self.x)


def dummy_label(transform=lambda x: x):
    def _dummy_label(ds: torchdata.Dataset):
        features = []
        labels = []
        for x in ds:
            features.append(x)
            labels.append(transform(x))
        return torchdata.TensorDataset(torch.Tensor(features),
                                       torch.Tensor(labels))

    return _dummy_label


def test_unlabelled_dataset_label_with_unlabelled():
    N = 15
    data = DummyData(N)
    ud = UnlabelledDataset(data, dummy_label(lambda x: x + 42))
    points_to_label = {5, 8, 3}
    points_left = set(range(N)) - points_to_label
    labelled = ud.label(list(points_to_label))

    assert len(ud) == len(points_left)
    assert len(labelled) == len(points_to_label)

    # in the first label call:
    #       *   *     *
    # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14

    # in the second label call:
    #   -   * - *     * -
    # 0 1 2   3   4 5   6  7  8  9 10 11
    # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
    # second call to label
    labelled = torchdata.ConcatDataset([
        labelled, ud.label([1, 3, 6])
    ])
    points_to_label = points_to_label.union({1, 4, 9})
    points_left = set(range(N)) - points_to_label
    assert len(ud) == len(points_left)
    assert len(labelled) == len(points_to_label)

    # in the third label call:
    #   - + * - *     * -  +     +     +
    # 0   1       2 3      4  5  6  7  8
    # 0 1 2   3   4 5   6  7  8  9 10 11
    # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
    # second call to label
    labelled = torchdata.ConcatDataset([
        labelled, ud.label([1, 4, 6, 8])
    ])
    points_to_label = points_to_label.union({2, 10, 12, 14})
    points_left = set(range(N)) - points_to_label
    assert len(ud) == len(points_left)
    assert len(labelled) == len(points_to_label)

    # test labelled indices
    assert [i.item() for i in ud.labelled_indices] == list(points_to_label)

    # dummy_label used transform = x + 42
    for x, y in labelled:
        assert y == (x + 42)
        points_to_label.remove(x.item())
    assert len(points_to_label) == 0

    # check remaining points in ud
    for x in ud: points_left.remove(x.item())
    assert len(points_left) == 0

    # check reset works
    ud.reset()
    assert len(ud) == N
    assert ud.labelled_indices.size(0) == 0
    full_dataset = set(range(N))
    for x in ud: full_dataset.remove(x.item())
    assert len(full_dataset) == 0


def test_unlabelled_dataset_label_with_labelled():
    N = 15
    data = DummyData(N, target=True)
    # don't have to provide labeler
    ud = UnlabelledDataset(data)
    points_to_label = {7, 4, 5, 1, 8, 12}
    points_left = set(range(N)) - points_to_label
    labelled = ud.label(list(points_to_label))

    assert len(ud) == len(points_left)
    assert len(labelled) == len(points_to_label)

    # in the first label call:
    #   *     * *   * *          *
    # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14

    # in the second label call:
    # - *   - * * - * * -        *
    # 0   1 2     3     4  5  6  7  8  9
    # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
    # second call to label
    labelled = torchdata.ConcatDataset([
        labelled, ud.label([0, 2, 3, 4])
    ])
    points_to_label = points_to_label.union({0, 3, 6, 9})
    points_left = set(range(N)) - points_to_label
    assert len(ud) == len(points_left)
    assert len(labelled) == len(points_to_label)

    # in the third label call:
    # - * + - * * - * * -  +     *  +
    #     0                1  2     3  4
    # 0   1 2     3     4  5  6  7  8  9
    # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
    # second call to label
    labelled = torchdata.ConcatDataset([
        labelled, ud.label([0, 1, 3])
    ])
    points_to_label = points_to_label.union({2, 10, 13})
    points_left = set(range(N)) - points_to_label
    assert len(ud) == len(points_left)
    assert len(labelled) == len(points_to_label)

    # test labelled indices
    assert [i.item() for i in ud.labelled_indices] == list(points_to_label)

    # dummy_label used transform = identity
    for x, y in labelled:
        assert x == y
        points_to_label.remove(x.item())
    assert len(points_to_label) == 0

    # check remaining points in ud
    for x in ud: points_left.remove(x.item())
    assert len(points_left) == 0

    # check reset works
    ud.reset()
    assert len(ud) == N
    assert ud.labelled_indices.size(0) == 0
    full_dataset = set(range(N))
    for x in ud: full_dataset.remove(x.item())
    assert len(full_dataset) == 0


def test_data_manager():
    N_LABELLED = 15
    N_UNLABELLED = N_LABELLED * 10
    train_pool = DummyData(N_LABELLED, target=True)
    pool = UnlabelledDataset(DummyData(N_UNLABELLED), dummy_label())
    dm = DataManager(train_pool, pool, MockAcquisitionFunction())

    acquire = 10
    dm.acquire(acquire)
    assert dm.n_unlabelled == N_UNLABELLED - acquire
    assert dm.n_labelled == N_LABELLED + acquire
    # since the implementation currently uses concat
    newly_acquired = itertools.islice(reversed(dm.labelled), acquire)

    # since the unlabelled dataset has range 0-N_UNLABELLED
    # and MockAcquisition returns the first `acquire` points
    should_acquire = set(range(acquire))
    for x, y in newly_acquired:
        x = x.item();
        y = y.item()
        assert x == y
        should_acquire.remove(x)
    assert len(should_acquire) == 0

    # second acquire will now take acquire - 2*acquire
    dm.acquire(acquire)
    assert dm.n_unlabelled == N_UNLABELLED - 2 * acquire
    assert dm.n_labelled == N_LABELLED + 2 * acquire
    # since the implementation currently uses concat
    newly_acquired = itertools.islice(reversed(dm.labelled), acquire)
    should_acquire = set(range(acquire, acquire * 2))
    for x, y in newly_acquired:
        x = x.item();
        y = y.item()
        assert x == y
        should_acquire.remove(x)
    assert len(should_acquire) == 0

    # test reset
    dm.reset()
    assert dm.n_labelled == N_LABELLED
    assert dm.n_unlabelled == N_UNLABELLED
    assert dm.labelled is train_pool
    assert dm.unlabelled is pool
