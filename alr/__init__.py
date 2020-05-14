"""
Main alr module
"""

import copy
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

__version__ = '0.0.0b8'


@dataclass
class FitResult:
    r"""
    A result object returned by :class:`ALRModel`'s :meth:`~ALRModel.fit` method.
    Each array has length equal to number of epochs.
    """
    train_loss: Union[np.ndarray, float]
    train_acc: Union[np.ndarray, float, None] = None
    val_loss: Union[np.ndarray, float, None] = None
    val_acc: Union[np.ndarray, float, None] = None

    def reduce(self, op: str, inplace: Optional[bool] = False) -> 'FitResult':
        r"""
        Reduces the results according to `op`.

        :param op: reduction operation. Any one of numpy's array reduction operation with
                    a signature `op(<array>)`. Two additional supported operations are
                    `"first"` and `"last"`
        :type op: str
        :param inplace: whether to perform operation in-place
        :type inplace: bool
        :return: a copy of itself if inplace is True, else itself
        :rtype: :class:`FitResult`
        :raises ValueError: if numpy does not support this operation
                             (i.e. `np.<op>` does not exist) or not one of `"first"` or `"last"`
        """
        op = op.lower().strip()
        if not hasattr(np, op) and op not in {"first", "last"}:
            raise ValueError(f"ALRModel.reduce does not support {op} operation.")
        result = self if inplace else copy.deepcopy(self)
        if op in {"first", "last"}:
            func = lambda x: x[0 if op == "first" else -1]  # noqa
        else:
            func = getattr(np, op)
        for attr, v in result.__dict__.items():
            if v is not None:
                setattr(result, attr, func(v))
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
            epochs: Optional[int] = 1,
            quiet: Optional[bool] = False,
            device: _DeviceType = None) -> FitResult:
        r"""
        A regular training loop much like Keras' fit function.

        :param train_loader: training data's DataLoader
        :type train_loader: torch.utils.data.DataLoader
        :param train_acc: at the end of each epoch, the training accuracy is calculated if set to true
        :type train_acc: bool, optional
        :param val_loader: validation data's DataLoader. If provided, the validation accuracy and loss
                            are recorded at the end of each epoch.
        :type val_loader: torch.utils.data.DataLoader, optional
        :param epochs: number of epochs
        :type epochs: int, optional
        :param quiet: display `tqdm` loading bar if `False`.
        :type quiet: bool, optional
        :param device: device type
        :type device: str, torch.device, None
        :return: :class:`FitResult` object containing training statistics
        :rtype: :class:`FitResult`
        """
        if self._compile_params is None:
            raise RuntimeError("Compile must be invoked before fitting model.")
        criterion = self._compile_params.criterion
        optimiser = self._compile_params.optimiser
        training_loss = []
        training_acc = []
        validation_loss = []
        validation_acc = []
        tepochs = range_progress_bar(epochs, desc="Epoch", leave=False) if not quiet else range(epochs)
        for _ in tepochs:
            # beware: self.eval() resets the state when we call self.evaluate()
            self.train()
            e_training_loss = []

            tbatch = progress_bar(train_loader, desc="Train batch", leave=False) if not quiet else train_loader
            # train
            for x, y in tbatch:
                if device:
                    x, y = x.to(device), y.to(device)
                preds = self(x)
                loss = criterion(preds, y)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                e_training_loss.append(loss.item())

            pfix = {}
            # get accuracies and losses
            if val_loader is not None:
                v_acc, v_loss = self.evaluate(val_loader, device=device)
                validation_loss.append(v_loss)
                validation_acc.append(v_acc)
                pfix['val_loss'] = v_loss
                pfix['val_acc'] = v_acc
            if train_acc:
                t_acc, _ = self.evaluate(train_loader, device=device, quiet=quiet)
                training_acc.append(t_acc)
                pfix['train_acc'] = t_acc

            training_loss.append(np.mean(e_training_loss))
            pfix['train_loss'] = training_loss[-1]

            # update tqdm
            if not quiet:
                tepochs.set_postfix(**pfix)

        return FitResult(
            train_loss=np.array(training_loss),
            train_acc=(np.array(training_acc) if train_acc else None),
            val_loss=(np.array(validation_loss) if val_loader else None),
            val_acc=(np.array(validation_acc) if val_loader else None)
        )

    def evaluate(self, data: torchdata.DataLoader, quiet: Optional[bool] = False,
                 device: _DeviceType = None) -> Tuple[float, float]:
        r"""
        Evaluate this model and return the mean accuracy and loss.

        :param data: dataset DataLoader
        :type data: torch.utils.data.DataLoader
        :param quiet: display `tqdm` loading bar if `False`.
        :type quiet: bool, optional
        :param device: device type
        :type device: str, torch.device, None
        :return: 2-tuple of mean accuracy and losses.
        :rtype: tuple
        """
        self.eval()
        if self._compile_params is None:
            raise RuntimeError("Compile must be invoked before evaluating model.")
        score = 0
        losses = []
        tqdm_load = progress_bar(data, desc="Evaluating model", leave=False) if not quiet else data
        with torch.no_grad():
            for x, y in tqdm_load:
                if device:
                    x, y = x.to(device), y.to(device)
                pred = self.predict(x)
                _, pred_lab = torch.max(pred, dim=1)
                score += (pred_lab == y).sum().item()
                losses.append(self._compile_params.criterion(pred, y).item())
        return score / len(data.dataset), np.mean(losses)


class MCDropout(ALRModel):
    def __init__(self, model: nn.Module,
                 forward: Optional[int] = 100,
                 inplace: Optional[bool] = True,
                 output_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 fast: Optional[bool] = False):
        r"""
        A wrapper that turns a regular PyTorch module into one that implements
        `Monte Carlo Dropout <https://arxiv.org/abs/1506.02142>`_ (Gal & Ghahramani, 2016).

        Args:
            model (`nn.Module`): `torch.nn.Module` object. This model's forward pass
                                  should return (log) probabilities. I.e. the final layer should
                                  be `softmax` or `log_softmax`. Otherwise, `output_transform` can
                                  be used to convert `model`'s output into probabilities.
            forward (int, optional): number of stochastic forward passes
            inplace (bool, optional): If `True`, the `model` is modified *in-place* when the dropout layers are
                                        replaced. If `False`, `model` is not modified and a new model is cloned.
            output_transform (callable, optional): model's output is given as input and the output of this
                                                    callable is expected to return (log) probabilities.
            fast (bool): If true, :meth:`stochastic_forward` will stack the batch dimension for faster
                          MC dropout passes. If false, then forward passes are called in a for-loop. Note,
                          the former will consume (`forward`) more memory.

        Attributes:
              base_model (`nn.Module`): provided base model (a clone if `inplace=True`)
              n_forward (int): number of forward passes (`forward`)
        """
        super(MCDropout, self).__init__()
        self.base_model = replace_dropout(model, inplace=inplace)
        self.n_forward = forward
        self._output_transform = output_transform if output_transform is not None else lambda x: x
        self._fast = fast

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass. *Note, this function has a different behaviour in eval mode*.
        It returns the mean score of :meth:`stochastic_forward` passes. In other words,
        if `self.training` is `False`, the following is returned instead:

        .. code:: python

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
        return torch.mean(self.stochastic_forward(x), dim=0)

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
                raise RuntimeError(r"Ran out of memory. Try reducing batch size or"
                                   "reducing the number of MC dropout samples. Alternatively, switch off"
                                   "fast MC dropout.") from e
        else:
            preds = torch.stack(
                [self._output_transform(self.base_model(x)) for _ in range(self.n_forward)]
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
            raise RuntimeError(r"Ran out of memory. Try reducing batch size or"
                               "reducing the number of MC dropout samples. Alternatively, switch off"
                               "fast MC dropout.") from e
        return out

