r"""
Useful samplers when training with small datasets
"""
import torch.utils.data as torchdata
import numpy as np
from typing import Optional, Union
from itertools import chain


class EpochExtender(torchdata.Sampler):
    def __init__(self, dataset: torchdata.Dataset, by: int):
        super().__init__(dataset)
        assert by >= 1
        self._by = by
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset) * self._by

    def __iter__(self):
        return chain.from_iterable(
            np.random.permutation(len(self._dataset)) for _ in range(self._by)
        )


class RandomFixedLengthSampler(torchdata.Sampler):
    # Adapted from BatchBALD Redux with modifications
    # https://github.com/BlackHC/batchbald_redux/blob/110161db3208d4df1d47146a7ac76a9794d1cab7/batchbald_redux/active_learning.py#L120Args:
    def __init__(self, dataset: torchdata.Dataset, length: int, shuffle: Optional[bool] = False):
        r"""
        Extends the epoch by sampling with replacement from the provided dataset
        until `length` samples are drawn. The number of samples in
        one epoch is `max(length, len(dataset))`.
        In other words, if `len(dataset)` > `length`, then this sampler
        behaves exactly like a `SequentialSampler` if `shuffle` is `False`, and
        like a `RandomSampler` if `shuffle` is `True`. The random state is affected
        by numpy's seed.

        Args:
            dataset (torch.utils.data.Dataset): dataset object
            length (int): the target length to achieve.
            shuffle (bool, optional): shuffle the indices if `len(dataset)` > `length`. This
                parameter is ignored otherwise (default = `False`). The random state depends on
                numpy's RNG.
        """
        super().__init__(dataset)
        assert length > 0, "What are you trying to pull?"
        self._dataset = dataset
        self._length = length
        self._shuffle = shuffle

    def __iter__(self):
        if self._length > len(self._dataset):
            return iter(
                np.random.permutation(self._length) % len(self._dataset)
            )
        else:
            if self._shuffle:
                return iter(np.random.permutation(len(self._dataset)))
            return iter(range(len(self._dataset)))

    def __len__(self):
        return max(self._length, len(self._dataset))


class MinLabelledSampler(torchdata.Sampler):
    def __init__(self,
                 labelled: torchdata.Dataset,
                 pseudo_labelled: torchdata.Dataset,
                 batch_size: int,
                 min_labelled: Union[int, float]):
        r"""
        Given labelled and pseudo_labelled datasets, returns a batch sampler that always yields
        exactly `min_labelled` points from the `labelled` dataset. If there is not enough
        points from `labelled`, then the points are recycled. Note that all the data points
        are shuffled. Note, the concatenated dataset is assumed to have labelled followed by
        pseudo_labelled, i.e. `torch.utils.data.ConcatDataset((labelled, pseudo_labelled))`.

        Args:
            labelled (torch.utils.data.Dataset): labelled dataset
            pseudo_labelled (torch.utils.data.Dataset): pseudo_labelled dataset
            batch_size (int): batch size
            min_labelled (int, float): min number of points that comes from `labelled`. Must be smaller
                than `batch_size`. If `float` is provided, then this argument is treated as a proportion.
        """
        min_labelled = min_labelled if type(min_labelled) == int else round(min_labelled * batch_size + .5)
        assert batch_size > min_labelled
        self._labelled = labelled
        self._pseudo_labelled = pseudo_labelled
        self._batch_size = batch_size
        self._min_labelled = min_labelled
        self._unlabelled_batch_size = batch_size - min_labelled

    def __len__(self):
        return round(len(self._pseudo_labelled) / self._unlabelled_batch_size + .5)

    def __iter__(self):
        num_unlabelled = self._batch_size - self._min_labelled
        labelled_indices = np.random.permutation(len(self) * self._min_labelled) % len(self._labelled)
        unlabelled_indices = np.random.permutation(len(self._pseudo_labelled))
        for i in range(len(self)):
            r1 = labelled_indices[i * self._min_labelled: (i + 1) * self._min_labelled]
            r2 = unlabelled_indices[i * num_unlabelled: (i + 1) * num_unlabelled] + len(self._labelled)
            res = np.r_[r1, r2]
            assert len(res) == self._batch_size
            yield res
