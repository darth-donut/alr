from typing import Callable, Sequence, Optional

import torch
import torch.utils.data as torchdata

from alr.acquisition import AcquisitionFunction


class UnlabelledDataset(torchdata.Dataset):
    def __init__(self,
                 dataset: torchdata.Dataset,
                 label_fn: Optional[Callable[[torchdata.Dataset], torchdata.Dataset]] = None):
        r"""
        A wrapper class to manage the unlabelled `dataset` by providing a simple
        interface to :meth:`label` specific points and remove from the underlying dataset.
        Furthermore, if the `label_fn` is not provided, this class automatically infers that
        the provided "unlabelled" dataset is, in fact, labelled. This is especially for
        benchmarking studies!

        :param dataset: unlabelled dataset
        :type dataset: :class:`torch.utils.data.Dataset`
        :param label_fn: a function that takes an unlabelled dataset and returns another
            dataset that's fully labelled. If this is not provided, then `dataset` should
            be labelled.
        :type label_fn: Callable: Dataset :math:`\rightarrow` Dataset, optional
        """
        self._dataset = dataset
        self._label_fn = label_fn
        self._mask = torch.ones(len(dataset), dtype=torch.bool)
        self._len = len(dataset)

    def label(self, idxs: Sequence[int]) -> torchdata.Dataset:
        r"""
        Label and return points specified by `idxs` according to provided `label_fn`.
        These labelled points will no longer be part of this dataset.

        :param idxs: indices of points to label
        :type idxs: `Sequence[int]`
        :return: a labelled dataset where each point is specified by `idxs` and labelled
            by `label_fn`.
        :rtype: :class:`torch.utils.data.Dataset`
        """
        # indices of data where it hasn't been labelled yet
        local_mask = torch.nonzero(self._mask).flatten()

        # can't acquire something that's not in the pool anymore
        assert self._mask[local_mask[idxs]].all(), "Can't label points that have been labelled."
        assert self._len, "There are no remaining unlabelled points."
        labelled = torchdata.Subset(self._dataset, local_mask[idxs])
        if self._label_fn:
            labelled = self._label_fn(labelled)
        self._mask[local_mask[idxs]] = 0
        self._len -= len(idxs)
        return labelled

    def __getitem__(self, idx) -> torch.Tensor:
        if self._label_fn:
            # user provided x only
            return self._dataset[self._mask][idx]
        # user provided (x, y) => return x only
        return self._dataset[self._mask][0][idx]

    def __len__(self) -> int:
        return self._len

    @property
    def labelled_indices(self) -> torch.Tensor:
        r"""
        Returns a 1-D tensor of indices that were labelled in the past.

        :return: all the indices that were labelled by :meth:`label`
        :rtype: `torch.Tensor`
        """
        return torch.nonzero(~self._mask).flatten()

    def reset(self) -> None:
        r"""
        Reset to initial state -- all labelled points are unlabelled and
        introduced back into the pool.

        :return: None
        :rtype: NoneType
        """
        self._mask = torch.ones(len(self._dataset), dtype=torch.bool)
        self._len = len(self._dataset)


class DataManager:
    def __init__(self,
                 labelled: torchdata.Dataset,
                 unlabelled: UnlabelledDataset,
                 acquisition_fn: AcquisitionFunction):
        r"""
        A stateful data manager class

        The :attr:`labelled` and :attr:`unlabelled` datasets are updated according to the points
        acquired by :meth:`acquire`. `acquisition_fn` dictates which points should
        be chosen from the unlabelled pool.

        :param labelled: training data with labelled points
        :type labelled: :class:`~torch.utils.data.Dataset`
        :param unlabelled: unlabelled pool
        :type unlabelled: :class:`UnlabelledDataset`
        :param acquisition_fn: acquisition function
        :type acquisition_fn: :class:`~alr.acquisition.AcquisitionFunction`
        """
        self._old_labelled = labelled
        self._labelled = labelled
        self._unlabelled = unlabelled
        self._a_fn = acquisition_fn

    def acquire(self, b: int) -> None:
        r"""
        Acquire `b` points from the :attr:`unlabelled` dataset and adds
        it to the :attr:`labelled` dataset.

        :param b: number of points to acquire at once
        :type b: `int`
        :return: None
        :rtype: NoneType
        """
        assert b <= self.n_unlabelled
        idxs = self._a_fn(self._unlabelled, b)
        assert idxs.shape == (b,)
        labelled = self._unlabelled.label(idxs)
        # TODO(optim): is there a better way to do this?
        self._labelled = torchdata.ConcatDataset(
            (self._labelled, labelled)
        )

    @property
    def n_labelled(self) -> int:
        r"""
        Current number of :attr:`labelled` points.

        :return: size of dataset
        :rtype: `int`
        """
        return len(self._labelled)

    @property
    def n_unlabelled(self) -> int:
        r"""
        Current number of :attr:`unlabelled` points.

        :return: size of dataset
        :rtype: `int`
        """
        return len(self._unlabelled)

    @property
    def labelled(self) -> torchdata.Dataset:
        r"""
        The current labelled dataset after considering previous acquisitions.

        :return: labelled dataset
        :rtype: :class:`torch.utils.data.Dataset`
        """
        return self._labelled

    @property
    def unlabelled(self) -> torchdata.Dataset:
        r"""
        The current unlabelled dataset after considering previous acquisitions.

        :return: unlabelled dataset
        :rtype: :class:`torch.utils.data.Dataset`
        """
        return self._unlabelled

    def reset(self) -> None:
        r"""
        Resets the state of this data manager. All acquired points are removed
        from the :attr:`labelled` dataset and added back into the :attr:`unlabelled` dataset.

        :return: None
        :rtype: NoneType
        """
        self._unlabelled.reset()
        self._labelled = self._old_labelled
