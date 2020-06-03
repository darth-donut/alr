from alr.training.samplers import RandomFixedLengthSampler, MinLabelledSampler
import torch.utils.data as torchdata
import torch
from collections import Counter
import numpy as np


class Data(torchdata.Dataset):
    def __init__(self, n):
        self._n = n
        self._arr = list(range(self._n))

    def __getitem__(self, idx):
        return self._arr[idx]

    def __len__(self):
        return self._n


class OddData(torchdata.Dataset):
    def __init__(self, n):
        self._arr = list(range(1, n * 2 + 1, 2))

    def __len__(self):
        return len(self._arr)

    def __getitem__(self, idx):
        return self._arr[idx]


class EvenData(torchdata.Dataset):
    def __init__(self, n):
        self._arr = list(range(0, n * 2, 2))

    def __len__(self):
        return len(self._arr)

    def __getitem__(self, idx):
        return self._arr[idx]

def test_random_fixed_length_sampler_target_length():
    ds_len = 10
    ds = Data(ds_len)
    target_length = 2 ** 12
    assert target_length > ds_len
    bs = 256
    sampler = RandomFixedLengthSampler(ds, length=target_length)
    assert len(sampler) == target_length
    loader = torchdata.DataLoader(ds, batch_size=bs, sampler=sampler)
    assert len(loader) == (target_length // bs)
    store = [x for x in loader]
    for x in store:
        assert x.size() == (bs,)
    store = torch.cat(store)
    assert store.size() == (target_length,)


def test_random_fixed_length_sampler_short_no_shuffle():
    ds_len = 2**13
    target_length = 2 ** 12
    assert target_length < ds_len
    ds = Data(ds_len)
    bs = 256
    sampler = RandomFixedLengthSampler(ds, length=target_length, shuffle=False)
    assert len(sampler) == ds_len
    loader = torchdata.DataLoader(ds, batch_size=bs, sampler=sampler)
    assert len(loader) == (ds_len // bs)
    store = [x for x in loader]
    for x in store:
        assert x.size() == (bs,)
    store = torch.cat(store)
    assert store.size() == (ds_len,)
    # since we're short of target_length and specified shuffle=False, we expect
    # the sampler to return sequential indices
    equals = []
    for target, item in enumerate(store.tolist()):
        equals.append(target == item)
    assert all(equals)


def test_random_fixed_length_sampler_short_shuffle():
    ds_len = 2**13
    target_length = 2 ** 12
    assert target_length < ds_len
    ds = Data(ds_len)
    bs = 256
    sampler = RandomFixedLengthSampler(ds, length=target_length, shuffle=True)
    assert len(sampler) == ds_len
    loader = torchdata.DataLoader(ds, batch_size=bs, sampler=sampler)
    assert len(loader) == (ds_len // bs)
    store = [x for x in loader]
    for x in store:
        assert x.size() == (bs,)
    store = torch.cat(store)
    assert store.size() == (ds_len,)
    # since we're short of target_length and specified shuffle=True, we expect
    # the sampler to return random indices
    equals = []
    for target, item in enumerate(store.tolist()):
        equals.append(target == item)

    # some indices might happen to be the the same but it's nigh impossible to
    # get sequential indices
    assert not all(equals)


def test_min_labelled_sampler():
    labelled = OddData(20)
    unlabelled = EvenData(1230)
    min_labelled = 32
    mls = MinLabelledSampler(
        labelled, unlabelled, batch_size=64, min_labelled=.5
    )
    c = Counter()
    for i in mls:
        indices = np.array(i)
        # there is exactly min_labelled in each batch
        assert (indices < len(labelled)).sum() == min_labelled
        c += Counter(i)

    for i in range(len(labelled), len(unlabelled)):
        # there is exactly one from unlabelled pool
        assert c[i] == 1
