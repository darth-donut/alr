"""
Main alr module
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from alr.acquisition import AcquisitionFunction
from alr.modules.dropout import replace_dropout
from alr.utils import _DeviceType

__version__ = '0.0.0b2'


class ALRModel(nn.Module, ABC):
    def __init__(self):
        """
        A :class:`ALRModel` provides generic methods required for common
        operations in active learning experiments.
        """
        super(ALRModel, self).__init__()
        self._models = []

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Regular forward pass. Usually reserved for training.

        :param x: input tensor
        :type x: `torch.Tensor`
        :return: output tensor
        :rtype: `torch.Tensor`
        """
        pass

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Override this function if predict has a different behaviour from :meth:`forward`.
        For example, if :meth:`forward` returns logits, this function could augment the
        output with softmax. Another example would be if the base model is a :class:`MCDropout` model,
        in which case, this function should return multiple stochastic forward passes.

        :param x: input tensor
        :type x: `torch.Tensor`
        :return: output tensor
        :rtype: `torch.Tensor`
        """
        return self.forward(x)

    def reset_weights(self) -> None:
        """
        Resets the model's weights.

        :return: None
        :rtype: NoneType
        """
        for m, state in self._models:
            # reload initial states
            m.load_state_dict(state)

    def __setattr__(self, key, value):
        super(ALRModel, self).__setattr__(key, value)
        if isinstance(value, nn.Module):
            # register nn.Module
            self._models.append((value, value.state_dict()))


class MCDropout(ALRModel):
    def __init__(self, model: nn.Module, forward: Optional[int] = 100, clone: Optional[bool] = False):
        """
        Implements `Monte Carlo Dropout <https://arxiv.org/abs/1506.02142>`_ (MCD). The difference between
        :meth:`forward` and :meth:`predict`
        is that the former returns a single forward pass (logits) while the
        latter returns the mean of `forward` number of passes (softmax scores).

        :param model: base `torch.nn.Module` object
        :type model: `nn.Module`
        :param forward: number of stochastic forward passes
        :type forward: int, optional
        :param clone: If `False`, the `model` is modified *in-place* such that the
                        dropout layers are replaced with :mod:`~alr.modules.dropout` layers.
                        If `True`, `model` is not modified and a new model is cloned.
        :type clone: `bool`, optional
        """
        super(MCDropout, self).__init__()
        self.base_model = replace_dropout(model, clone=clone)
        self.n_forward = forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Regular forward pass. Raises exception if `self.training` is `False`.

        :param x: input tensor
        :type x: `torch.Tensor`
        :return: output tensor
        :rtype: `torch.Tensor`
        :raises RuntimeError: raises `RuntimeError` if this method is used during evaluation;
            :meth:`predict` should be used instead.
        """
        if self.training:
            assert self.base_model.training
            return self.base_model(x)
        raise RuntimeError('Use model.predict(x) during evaluation.')

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the mean softmax score of :meth:`stochastic_forward` passes.

        Equivalent to:

        .. code:: python

            def predict(x):
                return torch.mean(stochastic_forward(x), dim=0)

        :param x: input tensor
        :type x: `torch.Tensor`
        :return: output tensor
        :rtype: `torch.Tensor`
        """
        return torch.mean(self.stochastic_forward(x), dim=0)

    def stochastic_forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Returns a :math:`m \times N \times C` `torch.Tensor` where:

            1. :math:`m` is equal to `self.n_forward`
            2. :math:`N` is the batch size, equal to `x.size(0)`
            3. :math:`C` is the number of units in the final layer (e.g. number of classes in a classification model)

        :param x: input tensor
        :type x: `torch.Tensor`
        :return: output tensor of shape :math:`m \times N \times C`
        :rtype: `torch.Tensor`
        """
        preds = torch.stack(
            [F.softmax(self.base_model(x), dim=1) for _ in range(self.n_forward)]
        )
        assert preds.size(0) == self.n_forward
        return preds
