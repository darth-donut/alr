import warnings
from abc import ABC, abstractmethod
from typing import Optional, Callable, Sequence

import numpy as np
import torch
import torch.distributions as dist
import torch.utils.data as torchdata

from alr.utils import _DeviceType

_BayesianCallable = Callable[[torch.Tensor], torch.Tensor]


class AcquisitionFunction(ABC):
    """
    A base class for all acquisition functions. All subclasses should
    override the `__call__` method.
    """
    @abstractmethod
    def __call__(self, X_pool: torchdata.Dataset, b: int) -> np.array:
        """
        Given unlabelled data pool `X_pool`, return the best `b`
        points for labelling by an oracle, where the best points
        are determined by this acquisition function and its parameters.

        :param X_pool: Unlabelled dataset
        :type X_pool: `torch.utils.data.Dataset`
        :param b: number of points to acquire
        :type b: int
        :return: array of indices to `X_pool`.
        :rtype: `np.array`
        """
        pass


class RandomAcquisition(AcquisitionFunction):
    """
    Implements random acquisition. Uniformly sample `b` indices.
    """
    def __call__(self, X_pool: torchdata.Dataset, b: int) -> np.array:
        return np.random.choice(len(X_pool), b, replace=False)


class BALD(AcquisitionFunction):
    def __init__(self, pred_fn: _BayesianCallable,
                 subset: Optional[int] = -1,
                 device: _DeviceType = None,
                 **data_loader_params):
        r"""
        Implements `BALD <https://arxiv.org/abs/1112.5745>`_.

        .. math::

            \begin{align}
                -\sum_c\left(\frac{1}{T}\sum_t\hat{p}^t_c \right)
                 log \left( \frac{1}{T}\sum_t\hat{p}^t_c \right) +
                 \frac{1}{T}\sum_{c,t}\hat{p}^t_c log \hat{p}^t_c
            \end{align}

        where :math:`\hat{p}^t_c` is the softmax output of class :math:`c`
        on the :math:`t^{th}` stochastic iteration.

        .. code:: python

            model = MCDropout(...)
            bald = BALD(eval_fwd_exp(model), subset=-1, device=device,
                        batch_size=512, pin_memory=True,
                        num_workers=2)
            bald(X_pool, b=10)


        :param pred_fn: A callable that returns a tensor of shape :math:`K \times N \times C` where
                        :math:`K` is the number of inference samples,
                        :math:`N` is the number of instances,
                        and :math:`C` is the number of classes.
                        **This function should return probabilities, not *log* probabilities!**
        :type pred_fn: `Callable`
        :param subset: Size of the subset of `X_pool`. Use -1 to denote the entire pool.
        :type subset: int, optional
        :param device: Move data to specified device when passing input data into `pred_fn`.
        :type device: `None`, `str`, `torch.device`
        :param data_loader_params: params to be passed into `DataLoader` when
                                   iterating over `X_pool`.

        .. warning::
            Do not set `shuffle=True` in `data_loader_params`! The indices will be
            incorrect if the `DataLoader` object shuffles `X_pool`!
        """
        self._pred_fn = pred_fn
        self._device = device
        self._subset = subset
        self._dl_params = data_loader_params
        assert not self._dl_params.get('shuffle', False)

    def __call__(self, X_pool: torchdata.Dataset, b: int) -> np.array:
        pool_size = len(X_pool)
        idxs = np.arange(pool_size)
        if self._subset != -1:
            r = min(self._subset, pool_size)
            assert b <= r, "Can't acquire more points that pool size"
            if b == r: return idxs
            idxs = np.random.choice(pool_size, r, replace=False)
            X_pool = torchdata.Subset(X_pool, idxs)
        dl = torchdata.DataLoader(X_pool, **self._dl_params)
        with torch.no_grad():
            mc_preds: torch.Tensor = torch.cat(
                [self._pred_fn(x.to(self._device) if self._device else x) for x in dl],
                dim=1
            )
            mc_preds = mc_preds.double()
            assert mc_preds.size()[1] == pool_size
            mean_mc_preds = mc_preds.mean(dim=0)
            H = -(mean_mc_preds * torch.log(mean_mc_preds + 1e-5)).sum(dim=1)
            E = (mc_preds * torch.log(mc_preds + 1e-5)).sum(dim=2).mean(dim=0)
            I = (H + E).cpu()
            assert torch.isfinite(I).all()
            assert I.shape == (pool_size,)
            result = torch.argsort(I, descending=True).numpy()[:b]
            return idxs[result]


class ICAL(AcquisitionFunction):
    def __init__(self, pred_fn: _BayesianCallable,
                 kernel_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 subset: Optional[int] = 200,
                 greedy_acquire: Optional[int] = 1,
                 use_one_hot: Optional[bool] = True,
                 sample_softmax: Optional[bool] = True,
                 device: _DeviceType = None,
                 **data_loader_params):
        r"""
        Implements 'normal' `ICAL <https://arxiv.org/abs/2002.07916>`_. :math:`R` points
        are randomly drawn from the pool and the average of the candidate batch's kernels
        is used instead. Thus, the dependency measure reduces to :math:`d = 2`.

        .. math::

            \frac{1}{|\mathcal{R}|} d\text{HSIC}(\displaystyle\sum_{x'\in\mathcal{R}} k^{x'},
            \frac{1}{B} \displaystyle\sum_{i = 1}^{B} k^{x_i})

        .. code:: python

            model = MCDropout(...)
            ical = ICAL(eval_fwd_exp(model), device=device,
                        batch_size=512,
                        pin_memory=True, num_workers=2)
            ical(X_pool, b=10)

        :param pred_fn: A callable that returns a tensor of shape :math:`K \times N \times C` where
                        :math:`K` is the number of inference samples,
                        :math:`N` is the number of instances,
                        and :math:`C` is the number of classes.
                        **This function should return probabilities, not *log* probabilities!**
        :type pred_fn: `Callable`
        :param kernel_fn: Kernel function, see static methods of :class:`ICAL`. Defaults to
            weighted a rational quadratic kernel. This is the default kernel in the paper.
        :type kernel_fn: Callable[[torch.Tensor], torch.Tensor]], optional
        :param subset: Normal ICAL uses a subset of `X_pool`. `subset` specifies the
                  size of this subset (:math:`|\mathcal{R}|` in the paper).
                  Use -1 to denote the entire pool.
        :type subset: int, optional
        :param greedy_acquire: how many points to acquire at once in each acquisition step.
        :type greedy_acquire: int, optional
        :param use_one_hot: use one_hot_encoding when calculating kernel matrix. This is the
            default behaviour in the paper.
        :type use_one_hot: bool, optional
        :param sample_softmax: sample the softmax probabilities. If this is `True`, then
            `use_one_hot` is automatically overriden to be `True`. This is the default
            behaviour in the paper.
        :type sample_softmax: bool, optional
        :param device: Move data to specified device when passing input data into `pred_fn`.
        :type device: `None`, `str`, `torch.device`
        :param data_loader_params: params to be passed into `DataLoader` when
                                   iterating over `X_pool`.

        .. warning::
            Do not set `shuffle=True` in `data_loader_params`! The indices will be
            incorrect if the `DataLoader` object shuffles `X_pool`!
        """
        self._r = subset
        self._pred_fn = pred_fn
        self._dl_params = data_loader_params
        self._device = device
        self._l = greedy_acquire
        self._use_oh = True if sample_softmax else use_one_hot
        self._sample_softmax = sample_softmax
        if kernel_fn is None:
            self._kernel = ICAL.rational_quadratic()
        else:
            self._kernel = kernel_fn
        assert not self._dl_params.get('shuffle', False)
        assert subset != 0

    def __call__(self, X_pool: torchdata.Dataset, b: int) -> np.array:
        l = self._l
        pool_size = len(X_pool)
        r = self._r if self._r != -1 else pool_size
        dl = torchdata.DataLoader(X_pool, **self._dl_params)
        with torch.no_grad():
            mc_preds = torch.cat([
                self._pred_fn(x.to(self._device) if self._device else x) for x in dl
            ], dim=1)
        mc_preds = mc_preds.detach_()
        n_forward, pool_size, C = mc_preds.size()
        if self._sample_softmax:
            assert self._use_oh
            cat_dist = dist.categorical.Categorical(mc_preds.view(n_forward * pool_size, -1))
            # mc_preds is now a vector of sampled class idx
            mc_preds = cat_dist.sample([1])[0]
            assert mc_preds.size() == (n_forward * pool_size,)
        if self._use_oh:
            if not self._sample_softmax:
                mc_preds = mc_preds.view(n_forward * pool_size, -1).argmax(dim=-1)
            assert mc_preds.size() == (n_forward * pool_size,)
            mc_preds = (torch.eye(C)[mc_preds]                      # shape [N * B x C]
                        .view(n_forward, pool_size, C))  # shape [N x B x C]
        assert mc_preds.size() == (n_forward, pool_size, C)
        kernel_matrices = self._kernel(mc_preds)
        assert kernel_matrices.size() == (n_forward, n_forward, pool_size)
        # [Pool_size x N x N]
        kernel_matrices = kernel_matrices.permute(2, 0, 1)
        # indices of points current in batch (a possible maximum of b by the
        # end of the iteration)
        batch_idxs = []

        while len(batch_idxs) < b:
            # always re-sample subset (what if we don't?)
            random_subset = np.random.choice(pool_size, size=r, replace=False)
            # a la theorem 2 - it suggested sum but we're using mean here - shouldn't make a difference
            pool_kernel = kernel_matrices[random_subset].mean(0)  # [N x N]
            # normal ICAL uses average batch kernels
            batch_kernels = (kernel_matrices + kernel_matrices[batch_idxs].sum(0, keepdim=True)) \
                            / (len(batch_idxs) + 1)  # [Pool_size x N x N]
            scores = self._dHSIC(
                torch.cat([
                    # TODO: can remove repeat?: potentially expensive!
                    pool_kernel.unsqueeze(0).repeat(batch_kernels.size(0), 1, 1).unsqueeze(-1),
                    batch_kernels.unsqueeze(-1)
                ], dim=-1)  # [Pool_size x N x N x 2]
            )
            assert scores.size() == (pool_size,)
            # mask chosen indices
            scores[batch_idxs] = -np.inf
            # greedily take top l scores
            idxs = torch.argsort(scores, descending=True)
            for idx in idxs[:l]: batch_idxs.append(idx.item())
        # greedily taking top l might sometimes acquire extra points if
        # b is not divisible by l, hence, truncate the output
        return np.array(batch_idxs[:b])

    @staticmethod
    def rational_quadratic(alphas: Optional[Sequence[float]] = (.2, .5, 1, 2, 5),
                           weights: Optional[Sequence[float]] = None) -> Callable:
        def _rational_quadratic(x: torch.Tensor) -> torch.Tensor:
            """
            :param x: tensor of shape [N x M x C]
            :return: tensor of shape [N x N x M]
            """
            N, M, _ = x.size()
            _alphas = x.new_tensor(alphas).view(-1, 1, 1, 1)
            if weights:
                _weights = x.new_tensor(weights)
            else:
                _weights = x.new_tensor(1.0 / _alphas.size(0)).repeat(_alphas.size(0))
            assert _weights.size(0) == _alphas.size(0)
            distances = (x.unsqueeze(0) - x.unsqueeze(1)).pow_(2).sum(-1)
            assert distances.size() == (N, N, M)

            distances = distances.unsqueeze_(0)   # 1 N N M
            # TODO: is logspace really necessary?
            log = torch.log1p(distances / (2 * _alphas))
            assert torch.isfinite(log).all()
            res = torch.einsum('w,wijk->ijk', _weights, torch.exp(-_alphas * log))
            assert torch.isfinite(res).all()
            return res
        return _rational_quadratic

    @staticmethod
    def _dHSIC(x: torch.Tensor) -> torch.Tensor:
        r"""
        Computes HSIC for d-variables in a batch of size :math:`K`.

        .. note::

            While the values are computed in logspace for numerical stability,
            the returned value is casted back to its original space.

        :param x: tensor of shape :math:`K \times N \times N \times D` where:

            * :math:`K` is the batch size
            * :math:`N` is the number of samples in each variable
            * :math:`D` is the number of variables
        :return: dHSIC scores, a tensor of shape :math:`K`.
        """
        K, N, N2, D = x.size()
        assert N == N2
        # trivial case, definition 2.6 https://arxiv.org/pdf/1603.00285.pdf
        if N < 2 * D:
            warnings.warn(f"The number of samples is lesser than twice "
                          f"the number of variables in dHISC. Trivial "
                          f"case of 0; this may or may not be intended.")
            return x.new_zeros(size=(K,))
        # https://github.com/NiklasPfister/dHSIC/blob/master/dHSIC/R/dhsic.R
        # logspace
        x = torch.log(x)
        logn = np.log(N)
        term1 = torch.sum(x, dim=-1).logsumexp(dim=(1, 2)) - 2 * logn
        term2 = torch.logsumexp(x, dim=(1, 2)).sum(dim=-1) - (2 * D * logn)
        term3 = (torch.logsumexp(x, dim=1)
                 .sum(dim=-1)
                 .logsumexp(dim=-1) + np.log(2) - (D + 1) * logn)
        assert term1.size() == term2.size() == term3.size() == (K,)
        # subtract max for numerical stabilisation
        term_max = torch.stack([term1, term2, term3], dim=0).max(dim=0)[0]
        assert term_max.size() == (K,)
        res = (term1 - term_max).exp_() + (term2 - term_max).exp_() - (term3 - term_max).exp_()
        res *= term_max.exp_()
        assert torch.isfinite(res).all()
        return res
