from collections import namedtuple
from typing import Optional, Union, Tuple

import numpy as np
import torch
import torch.utils.data as torchdata

# type aliases
_DeviceType = Optional[Union[str, torch.device]]
_ActiveLearningDataset = namedtuple('ActiveLearningDataset', 'training unlabelled')


def stratified_partition(ds: torchdata.Dataset, classes: int, size: int) \
        -> Tuple[torchdata.Dataset, torchdata.Dataset]:
    r"""
    Partitions `ds` into training pool and a faux unlabelled pool. The "unlabelled"
    pool will contain `len(ds) - size` data points and the training pool will contain
    `size` data points where the target class is as balanced as possible. Note,
    the faux "unlabelled" pool will contain target labels since it's
    derived from the training pool.

    :param ds: dataset containing input features and target class
    :type ds: :class:`torch.utils.data.Dataset`
    :param classes: number of target classes contained in `ds`
    :type classes: int
    :param size: size of resulting training pool
    :type size: int
    :return: (training pool, unlabelled pool)
    :rtype: tuple
    """
    assert size < len(ds)
    c = size // classes
    extra = size % classes
    count = {cls: c for cls in range(classes)}
    original_idxs = set(range(len(ds)))
    sampled_idxs = []
    # the first `extra` classes gets the extra counts
    while extra:
        count[extra] += 1
        extra -= 1
    for idx in np.random.permutation(len(ds)):
        if all(i == 0 for i in count.values()):
            break
        y = ds[idx][1]
        if count[y]:
            count[y] -= 1
            sampled_idxs.append(idx)
    return _ActiveLearningDataset(training=torchdata.Subset(ds, sampled_idxs),
                                  unlabelled=torchdata.Subset(ds, list(original_idxs - set(sampled_idxs))))


def eval_fwd_exp(model: 'MCDropout'):
    r"""
    A helper function that returns a function that
    sets model to eval mode, calls stochastic forward, and exponentiates the output.
    This is useful for acquisition functions that
    expect :meth:`alr.MCDropout.stochastic_forward` to return non-log probabilities.

    Examples:
        .. code:: python

            model = MCDropout(...)
            bald = BALD(eval_fwd_exp(model), ...)

    Args:
        model (MCDropout): MCDropout model. The stochastic forward output of this model
                            is expected to be log-softmax probabilities.

    Returns:
        Callable: a function that takes a tensor and returns a
        tensor that contains (non log-) probabilities
        from the model's stochastic forward pass
    """
    def _fwd(x: torch.Tensor) -> torch.Tensor:
        model.eval()
        return model.stochastic_forward(x).exp()
    return _fwd


def eval_fwd(model: 'MCDropout'):
    r"""
    A helper function that returns a function that
    sets model to eval mode and calls stochastic forward.
    This is useful for acquisition functions that
    expect :meth:`alr.MCDropout.stochastic_forward` to run on eval mode.

    Examples:
        .. code:: python

            model = MCDropout(...)
            bald = BALD(eval_fwd(model), ...)

    Args:
        model (MCDropout): MCDropout model. The stochastic forward output of this model
                            is expected to be softmax probabilities.

    Returns:
        Callable: a function that takes a tensor and returns a
        tensor that contains probabilities from the model's stochastic forward pass
    """
    def _fwd(x: torch.Tensor) -> torch.Tensor:
        model.eval()
        return model.stochastic_forward(x)
    return _fwd
