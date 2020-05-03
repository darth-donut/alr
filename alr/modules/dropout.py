r"""
Modify Dropout :class:`torch.nn.Modules` to _always_ activate in training and inference.
The classes in this module are taken from `PyTorch <https://github.com/pytorch/pytorch/tree/master/torch>`_ *as-is*.
The main function you should be concerned with is :func:`replace_dropout`.
"""
import torch
import torch.nn.functional as F
import copy
import sys
import inspect
import re
import warnings

from torch.nn.modules.dropout import _DropoutNd
from typing import Optional

# The Dropout classes below are taken as-is from torch


class PersistentDropout(_DropoutNd):
    r"""During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution. Each channel will be zeroed out independently on every forward
    call.
    This has proven to be an effective technique for regularization and
    preventing the co-adaptation of neurons as described in the paper
    `Improving neural networks by preventing co-adaptation of feature
    detectors`_ .
    Furthermore, the outputs are scaled by a factor of :math:`\frac{1}{1-p}` during
    training. This means that during evaluation the module simply computes an
    identity function.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    Examples::
        >>> m = nn.Dropout(p=0.2)
        >>> input = torch.randn(20, 16)
        >>> output = m(input)

    .. _Improving neural networks by preventing co-adaptation of feature
        detectors: https://arxiv.org/abs/1207.0580
    """

    def forward(self, input):
        return F.dropout(input, self.p, True, self.inplace)


class PersistentDropout2d(_DropoutNd):
    r"""Randomly zero out entire channels (a channel is a 2D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 2D tensor :math:`\text{input}[i, j]`).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.
    Usually the input comes from :class:`nn.Conv2d` modules.
    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.
    In this case, :func:`nn.Dropout2d` will help promote independence between
    feature maps and should be used instead.

    Args:
        p (float, optional): probability of an element to be zero-ed.
        inplace (bool, optional): If set to ``True``, will do this operation in-place

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples::
        >>> m = nn.Dropout2d(p=0.2)
        >>> input = torch.randn(20, 16, 32, 32)
        >>> output = m(input)

    .. _Efficient Object Localization Using Convolutional Networks:
       http://arxiv.org/abs/1411.4280
    """

    def forward(self, input):
        return F.dropout2d(input, self.p, True, self.inplace)


class PersistentDropout3d(_DropoutNd):
    r"""Randomly zero out entire channels (a channel is a 3D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 3D tensor :math:`\text{input}[i, j]`).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.
    Usually the input comes from :class:`nn.Conv3d` modules.
    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.
    In this case, :func:`nn.Dropout3d` will help promote independence between
    feature maps and should be used instead.

    Args:
        p (float, optional): probability of an element to be zeroed.
        inplace (bool, optional): If set to ``True``, will do this operation in-place

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples::
        >>> m = nn.Dropout3d(p=0.2)
        >>> input = torch.randn(20, 16, 4, 32, 32)
        >>> output = m(input)

    .. _Efficient Object Localization Using Convolutional Networks:
       http://arxiv.org/abs/1411.4280
    """

    def forward(self, input):
        return F.dropout3d(input, self.p, True, self.inplace)


class PersistentAlphaDropout(_DropoutNd):
    r"""Applies Alpha Dropout over the input.
    Alpha Dropout is a type of Dropout that maintains the self-normalizing
    property.
    For an input with zero mean and unit standard deviation, the output of
    Alpha Dropout maintains the original mean and standard deviation of the
    input.
    Alpha Dropout goes hand-in-hand with SELU activation function, which ensures
    that the outputs have zero mean and unit standard deviation.
    During training, it randomly masks some of the elements of the input
    tensor with probability *p* using samples from a bernoulli distribution.
    The elements to masked are randomized on every forward call, and scaled
    and shifted to maintain zero mean and unit standard deviation.
    During evaluation the module simply computes an identity function.
    More details can be found in the paper `Self-Normalizing Neural Networks`_ .

    Args:
        p (float): probability of an element to be dropped. Default: 0.5
        inplace (bool, optional): If set to ``True``, will do this operation in-place

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    Examples::
        >>> m = nn.AlphaDropout(p=0.2)
        >>> input = torch.randn(20, 16)
        >>> output = m(input)

    .. _Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515
    """

    def forward(self, input):
        return F.alpha_dropout(input, self.p, True)


class PersistentFeatureAlphaDropout(_DropoutNd):
    def forward(self, input):
        return F.feature_alpha_dropout(input, self.p, True)


def replace_dropout(module: torch.nn.Module,
                    inplace: Optional[bool] = True) -> torch.nn.Module:
    r"""
    Recursively replaces dropout modules in `module` such that dropout is performed
    regardless of the model's mode. That is, dropout is performed during training
    (`model.train()`) and inference (`model.eval()`) modes.

    Args:
        module (`torch.nn.Module`): PyTorch module object
        inplace (bool, optional): If `True`, the `model` is modified *in-place*. If `False`, `model` is not modified and a new model is cloned.

    Returns:
        `torch.nn.Module`: Same `module` instance if `inplace` is `False`, else a brand new module.
    """
    def _replace_dropout(parent):
        for name, mod in parent.named_children():
            if isinstance(mod, _DropoutNd):
                try:
                    # replace dropout module with one that always does dropout regardless of the model's mode
                    parent.add_module(name,
                                      getattr(sys.modules[__name__],
                                              'Persistent' + type(mod).__name__)(p=mod.p, inplace=mod.inplace))
                except AttributeError:
                    raise NotImplementedError(f"{type(mod).__name__} hasn't been implemented yet.")
            _replace_dropout(mod)

    if not inplace:
        module = copy.deepcopy(module)
    _replace_dropout(module)
    _inspect_forward(module)
    return module


def _inspect_forward(module: torch.nn.Module):
    src = inspect.getsource(module.forward).strip()
    src = re.sub(r"\s", "", src)
    if re.search(r".*dropout(\dd)?\(.*\).*", src):
        warnings.warn("Found usage of non-module dropout in module's forward function."
                      " Please make sure that the training flag is set to True during eval mode too.", UserWarning)
