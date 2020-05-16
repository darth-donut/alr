r"""
Useful samplers when training with small datasets
"""
import torch.utils.data as torchdata
import numpy as np
from typing import Optional


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
                np.random.choice(
                    len(self._dataset),
                    size=self._length, replace=True
                )
            )
        else:
            if self._shuffle:
                return iter(np.random.permutation(len(self._dataset)))
            return iter(range(len(self._dataset)))

    def __len__(self):
        return max(self._length, len(self._dataset))