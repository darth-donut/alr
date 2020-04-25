"""
Main alr module
"""

import copy
import sys
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, Union

import numpy as np
import torch
import torch.utils.data as torchdata
from torch import nn
from torch.nn import functional as F

from alr.acquisition import AcquisitionFunction
from alr.modules.dropout import replace_dropout
from alr.utils import _DeviceType, range_progress_bar, progress_bar

__version__ = '0.0.0b4'


@dataclass
class FitResult:
    r"""
    A result object returned by :class:`ALRModel`'s :meth:`~ALRModel.fit` method.
    """
    train_loss: np.ndarray
    train_acc: Union[np.ndarray, float, None] = None
    val_loss: Union[np.ndarray, float, None] = None
    val_acc: Union[np.ndarray, float, None] = None

    def reduce(self, op: str, inplace: Optional[bool] = False) -> 'FitResult':
        r"""
        Reduces the results according to `op`.

        :param op: reduction operation
        :type op: str
        :param inplace: whether to perform operation in-place
        :type inplace: bool
        :return: a copy of itself if inplace is True, else itself
        :rtype: :class:`FitResult`
        :raises ValueError: if numpy does not support this operation. I.e. `np.<op>` does not exist.
        """
        if not hasattr(np, op):
            raise ValueError(f"Numpy does not support {op} operation.")
        result = self if inplace else copy.deepcopy(self)
        for attr, v in result.__dict__.items():
            if v is not None:
                setattr(result, attr, getattr(np, op)(v))
        return result


class ALRModel(nn.Module, ABC):
    # criterion is of type nn.Module, we wrap it so it wouldn't register in this module
    _CompileParams = namedtuple('CompileParams', 'criterion optimiser')

    def __init__(self):
        """
        A :class:`ALRModel` provides generic methods required for common
        operations in active learning experiments.
        """
        super(ALRModel, self).__init__()
        self._models = []
        self._compile_params = None

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

    def compile(self, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                optimiser: torch.optim.Optimizer) -> None:
        r"""
        Compiles the model. Similar to Keras' API, the model saves the criterion
        and optimiser for use in :meth:`fit` later on.

        :param criterion: a function that takes model outputs and target classes as inputs and
                            returns a score tensor
        :type criterion: torch.Tensor :math:`\times` torch.Tensor :math:`\rightarrow` torch.Tensor
        :param optimiser: optimser object
        :type optimiser: torch.optim.Optimizer
        :return: None
        :rtype: NoneType
        """
        self._compile_params = ALRModel._CompileParams(criterion, optimiser)

    def fit(self, train_loader: torchdata.DataLoader,
            train_acc: Optional[bool] = True,
            val_loader: Optional[torchdata.DataLoader] = None,
            val_loss: Optional[bool] = False,
            epochs: Optional[int] = 1,
            device: _DeviceType = None) -> FitResult:
        r"""
        A regular training loop much like Keras' fit function.

        :param train_loader: training data's DataLoader
        :type train_loader: torch.utils.data.DataLoader
        :param train_acc: at the end of each epoch, the training accuracy is calculated if set to true
        :type train_acc: bool, optional
        :param val_loader: validation data's DataLoader
        :type val_loader: torch.utils.data.DataLoader, optional
        :param val_loss: at the end of each epoch, the validation loss is calculated if set to true.
        :type val_loss: bool, optional
        :param epochs: number of epochs
        :type epochs: int, optional
        :param device: device type
        :type device: str, torch.device, None
        :return: :class:`FitResult` object containing training statistics
        :rtype: :class:`FitResult`
        """
        assert not val_loss or val_loader is not None, "If val_loss is True, val_loader must be provided."
        if self._compile_params is None:
            raise RuntimeError("Compile must be invoked before fitting model.")
        criterion = self._compile_params.criterion
        optimiser = self._compile_params.optimiser
        training_loss = []
        training_acc = []
        validation_loss = []
        validation_acc = []
        tepochs = range_progress_bar(epochs, leave=False, file=sys.stdout)
        for _ in tepochs:
            # beware: self.eval() resets the state when we call self.evaluate()
            self.train()
            e_training_loss = []

            # train
            for x, y in train_loader:
                if device:
                    x, y = x.to(device), y.to(device)
                preds = self(x)
                loss = criterion(preds, y)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                e_training_loss.append(loss.item())

            pfix = {}
            # get accuracies
            if val_loader is not None:
                v_acc, v_loss = self.evaluate(
                    val_loader, with_loss=val_loss, device=device)
                validation_loss.append(v_loss)
                validation_acc.append(v_acc)
                if val_loss:
                    pfix['val_loss'] = v_loss
                pfix['val_acc'] = v_acc
            if train_acc:
                t_acc, _ = self.evaluate(
                    train_loader, with_loss=False, device=device)
                training_acc.append(t_acc)
                pfix['train_acc'] = t_acc

            training_loss.append(np.mean(e_training_loss))
            pfix['train_loss'] = training_loss[-1]

            # update tqdm
            tepochs.set_postfix(**pfix)

        return FitResult(
            train_loss=np.array(training_loss),
            train_acc=(np.array(training_acc) if train_acc else None),
            val_loss=(np.array(validation_loss) if val_loss else None),
            val_acc=(np.array(validation_acc) if val_loader else None)
        )

    def evaluate(self, data: torchdata.DataLoader,
                 with_loss: Optional[bool] = False,
                 device: _DeviceType = None) -> Tuple[float, Union[float, None]]:
        r"""
        Evaluate this model and return the mean accuracy and loss.

        :param data: dataset DataLoader
        :type data: torch.utils.data.DataLoader
        :param with_loss: if true, then calculate the loss as well. This incurs additional
                            computational time as this method uses
                            :meth:`predict` to calculate accuracies
                            and :meth:`forward` to calculate losses.
        :type with_loss: bool, optional
        :param device: device type
        :type device: str, torch.device, None
        :return: 2-tuple of mean accuracy and losses. Losses is None if `with_loss` is false.
        :rtype: tuple
        """
        self.eval()
        if self._compile_params is None and with_loss:
            raise RuntimeError("Compile must be invoked before evaluating model with loss.")
        score = total = 0
        losses = []
        tqdm_load = progress_bar(data, desc="Evaluating model", leave=False, file=sys.stdout)
        with torch.no_grad():
            for x, y in tqdm_load:
                if device:
                    x, y = x.to(device), y.to(device)
                _, preds = torch.max(self.predict(x), dim=1)
                score += (preds == y).sum().item()
                total += y.size(0)
                if with_loss:
                    losses.append(
                        self._compile_params.criterion(self(x), y).item()
                    )
        return score / total, (np.mean(losses) if losses else None)  # noqa


class MCDropout(ALRModel):
    def __init__(self, model: nn.Module, forward: Optional[int] = 100, inplace: Optional[bool] = True):
        """
        Implements `Monte Carlo Dropout <https://arxiv.org/abs/1506.02142>`_ (MCD). The difference between
        :meth:`forward` and :meth:`predict`
        is that the former returns a single forward pass (logits) while the
        latter returns the mean of `forward` number of passes (softmax scores).

        :param model: base `torch.nn.Module` object
        :type model: `nn.Module`
        :param forward: number of stochastic forward passes
        :type forward: int, optional
        :param inplace: If `True`, the `model` is modified *in-place* when the dropout layers are
                        replaced. If `False`, `model` is not modified and a new model is cloned.
        :type inplace: `bool`, optional
        """
        super(MCDropout, self).__init__()
        self.base_model = replace_dropout(model, inplace=inplace)
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
        r"""
        Returns the mean softmax score of :meth:`stochastic_forward` passes.

        Equivalent to:

        .. code:: python

            def predict(x):
                return torch.mean(stochastic_forward(x), dim=0)

        :param x: input tensor, any size
        :type x: `torch.Tensor`
        :return: output tensor of size :math:`N \times C` where :math:`N` is the
                    batch size and :math:`C` is the number of target classes.
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
