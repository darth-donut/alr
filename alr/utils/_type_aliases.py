from collections import namedtuple
from typing import Optional, Union, Callable

import torch

_DeviceType = Optional[Union[str, torch.device]]
_ActiveLearningDataset = namedtuple("ActiveLearningDataset", "training unlabelled")
_Loss_fn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
