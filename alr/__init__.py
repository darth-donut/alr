"""
Main alr module
"""

import copy
import math
from abc import ABC, abstractmethod
from typing import Optional, Callable

import torch
from torch import nn

from alr.acquisition import AcquisitionFunction
from alr.modules.dropout import replace_dropout, replace_consistent_dropout
from alr.utils import range_progress_bar, progress_bar
from alr.utils._type_aliases import _DeviceType

__version__ = "0.0.0b8"


class ALRModel(nn.Module, ABC):
    def __init__(self):
        """
        A :class:`ALRModel` provides generic methods required for common
        operations in active learning experiments.
        """
        super(ALRModel, self).__init__()
        self._snapshot = None

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
        Sets the model mode to eval and calls forward.

        :param x: input tensor
        :type x: `torch.Tensor`
        :return: output tensor
        :rtype: `torch.Tensor`
        """
        self.eval()
        return self(x)

    def reset_weights(self) -> None:
        """
        Resets the model's weights to the last saved snapshot.

        :return: None
        :rtype: NoneType
        """
        assert self._snapshot is not None, "Snapshot was never taken"
        self.load_state_dict(self._snapshot, strict=True)

    def snap(self) -> None:
        r"""
        Take and store a snapshot of the current state.

        Returns:
            NoneType: None
        """
        # update snapshot
        self._snapshot = copy.deepcopy(self.state_dict())


class MCDropout(ALRModel):
    def __init__(
        self,
        model: nn.Module,
        forward: Optional[int] = 100,
        reduce: Optional[str] = "logsumexp",
        inplace: Optional[bool] = True,
        output_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        fast: Optional[bool] = False,
        consistent: Optional[bool] = False,
    ):
        r"""
        A wrapper that turns a regular PyTorch module into one that implements
        `Monte Carlo Dropout <https://arxiv.org/abs/1506.02142>`_ (Gal & Ghahramani, 2016).

        Args:
            model (`nn.Module`): `torch.nn.Module` object. This model's forward pass
                                  should return (log) probabilities. I.e. the final layer should
                                  be `softmax` or `log_softmax`. Otherwise, `output_transform` can
                                  be used to convert `model`'s output into probabilities.
            forward (int, optional): number of stochastic forward passes
            reduce (str, optional): either `"logsumexp"` or `"mean"`. This is used to reduce the
                n `forward` stochastic passes during evaluation. If `model` or `output_transform`
                returns probabilities (i.e. `F.softmax`), this should be `"mean"`;
                otherwise it should be "logsumexp" if they return log-probabilities (i.e. `F.log_softmax`).
                [default = `"logsumexp"`]
            inplace (bool, optional): if `True`, the `model` is modified *in-place* when the dropout layers are
                                        replaced. If `False`, `model` is not modified and a new model is cloned.
            output_transform (callable, optional): model's output is given as input and the output of this
                                                    callable is expected to return (log) probabilities.
            fast (bool, optional): if true, :meth:`stochastic_forward` will stack the batch dimension for faster
                          MC dropout passes. If false, then forward passes are called in a for-loop. Note,
                          the former will consume `forward` times more memory.
            consistent (bool, optional): if true, the dropout layers will be replaced with consistent variants.
        Attributes:
              base_model (`nn.Module`): provided base model (a clone if `inplace=True`)
              n_forward (int): number of forward passes (`forward`)
        """
        super(MCDropout, self).__init__()
        if consistent:
            self.base_model = replace_consistent_dropout(model, inplace=inplace)
        else:
            self.base_model = replace_dropout(model, inplace=inplace)
        self.n_forward = forward
        self._output_transform = (
            output_transform if output_transform is not None else lambda x: x
        )
        self._reduce = reduce.lower()
        assert self._reduce in {"logsumexp", "mean"}
        self._fast = fast
        self.snap()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass. *Note, this function has a different behaviour in eval mode*.
        It returns the (log) mean score of :meth:`stochastic_forward` passes. In other words,
        if `self.training` is `False`, the following is returned instead:

        .. code:: python

            # if reduce = "logsumexp"
            torch.logsumexp(self.stochastic_forward(x), dim=0) - log(self.n_forward)

            # if reduce = "mean"
            torch.mean(self.stochastic_forward(x), dim=0)


        Args:
            x (`torch.Tensor`): input tensor, any size

        Returns:
            `torch.Tensor`:
                output tensor of size :math:`N \times C` where
                :math:`N` is the batch size and :math:`C` is the number of target classes.

        Note:
              if a single forward pass is required during eval mode, one could use the following
              instead: `base_model(x)`
        """
        if self.training:
            return self._output_transform(self.base_model(x))
        if self._reduce == "mean":
            return torch.mean(self.stochastic_forward(x), dim=0)
        # if self._reduce == "logsumexp"
        return torch.logsumexp(self.stochastic_forward(x), dim=0) - math.log(
            self.n_forward
        )

    def stochastic_forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Returns a :math:`m \times N \times C` `torch.Tensor` where:

            1. :math:`m` is equal to `self.n_forward`
            2. :math:`N` is the batch size, equal to `x.size(0)`
            3. :math:`C` is the number of units in the final layer (e.g. number of classes in a classification model)

        Args:
            x (`torch.Tensor`): input tensor

        Returns:
            `torch.Tensor`: output tensor of shape :math:`m \times N \times C`

        Raises:
            RuntimeError: Occurs when the machine runs out of memory and `fast` was set to true.
        """
        if self._fast:
            size = x.size()
            x = self._repeat_n(x, self.n_forward)
            assert x.size() == (size[0] * self.n_forward, *size[1:])
            try:
                preds = self._output_transform(self.base_model(x))
                preds = preds.view(self.n_forward, -1, *preds.size()[1:])
            except RuntimeError as e:
                raise RuntimeError(
                    r"Ran out of memory. Try reducing batch size or"
                    "reducing the number of MC dropout samples. Alternatively, switch off"
                    "fast MC dropout."
                ) from e
        else:
            preds = torch.stack(
                [
                    self._output_transform(self.base_model(x))
                    for _ in range(self.n_forward)
                ]
            )
        assert preds.size(0) == self.n_forward
        return preds

    @staticmethod
    def _repeat_n(x: torch.Tensor, n: int) -> torch.Tensor:
        r"""
        Repeat the data in x `n` times along the batch dimension.

        Args:
            x (torch.Tensor): input tensor, the batch dimension is assumed to be 0.
            n (int): number of repeats

        Returns:
            torch.Tensor: output tensor

        Raises:
            RuntimeError: Occurs when the machine runs out of memory.
        """
        try:
            out = x.repeat(n, *([1] * (x.ndim - 1)))
        except RuntimeError as e:
            raise RuntimeError(
                r"Ran out of memory. Try reducing batch size or"
                "reducing the number of MC dropout samples. Alternatively, switch off"
                "fast MC dropout."
            ) from e
        return out
