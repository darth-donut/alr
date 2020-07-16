import torch
import numpy as np
import os
import random

from typing import Optional, Sequence, Union
from pathlib import Path

from alr.utils.time_utils import Elapsed, timeop, time_this, Time
from alr.utils.experiment_helpers import stratified_partition, eval_fwd, eval_fwd_exp
from alr.utils._type_aliases import _DeviceType
from alr.utils.progress_bar import progress_bar, range_progress_bar

__all__ = [
    'Elapsed', 'timeop', 'time_this', 'Time',
    'stratified_partition', 'eval_fwd', 'eval_fwd_exp',
    'progress_bar', 'range_progress_bar',
    'manual_seed'
]


def manual_seed(seed: Optional[int] = 42, det_cudnn: Optional[bool] = False) -> int:
    r"""
    To ensure reproducibility, set the seeds and make cuDNN deterministic.

    Args:
        seed (int, optional): Sets numpy, torch, and random's seed. Also sets the
            environment variable `PYTHONHASHSEED`.
        det_cudnn (bool, optional): Make cuDNN deterministic and disables benchmark for
            reproducibility at the expense of speed. This argument is ignored if
            cuda is not available.

    Returns:
        int: the seed used.
    """
    # torch
    torch.manual_seed(seed)
    if det_cudnn and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # numpy
    np.random.seed(seed)
    # random
    random.seed(seed)
    # env
    os.environ['PYTHONHASHSEED'] = str(seed)
    return seed


def _map_device(xs: Sequence[torch.Tensor], device: _DeviceType):
    if device is not None:
        return [x.to(device) for x in xs]
    return xs


def savefig(filename, fig=None, pad_inches=0.05):
    """Get rid of them pesky padding"""
    import matplotlib.pyplot as plt
    if not fig:
        fig = plt.gcf()
    fig.savefig(filename, pad_inches=pad_inches, bbox_inches='tight')
