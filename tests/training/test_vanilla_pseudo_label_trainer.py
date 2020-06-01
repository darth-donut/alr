import numpy as np
import torch
from torch.nn import functional as F
import torch.utils.data as torchdata
from alr.training.pseudo_label_trainer import soft_nll_loss, soft_cross_entropy
from alr.data.datasets import Dataset
from alr.utils.math import entropy
from ignite.engine import Engine, Events


def test_soft_nll_loss():
    F = torch.nn.functional
    target_logits = torch.randn(size=(100, 10))
    logits = torch.randn(size=(100, 10))

    # calculated expected cross entropy
    target_dist = F.softmax(target_logits, dim=-1)
    probs = F.softmax(logits, dim=-1)
    cross_entropy = -(target_dist * torch.log(probs)).sum(dim=-1).mean()

    # now, test nll
    res = soft_nll_loss(preds=F.log_softmax(logits, dim=-1), target=F.log_softmax(target_logits, dim=-1))
    assert torch.isclose(res, cross_entropy)


def test_soft_nll_loss_trivial():
    N = 1000
    target_logits = torch.randperm(N) % 10
    one_hot = torch.eye(10)[target_logits]
    assert torch.eq(torch.argmax(one_hot, dim=-1), target_logits).all()

    logits = torch.randn(size=(N, 10))
    preds = F.log_softmax(logits, dim=-1)
    a = F.nll_loss(preds, target_logits)
    b = soft_nll_loss(preds, target=(one_hot + 1e-8).log())
    assert torch.isclose(a, b)


def test_soft_cross_entropy():
    target_logits = torch.randn(size=(100, 10))
    logits = torch.randn(size=(100, 10))

    # calculated expected cross entropy
    target_dist = F.softmax(target_logits, dim=-1)
    probs = F.softmax(logits, dim=-1)
    cross_entropy = -(target_dist * torch.log(probs)).sum(dim=-1).mean()

    # now, test cross entropy
    res = soft_cross_entropy(logits, target=target_logits)
    assert torch.isclose(res, cross_entropy)


def test_soft_cross_entropy_trivial():
    N = 1000
    target_logits = torch.randperm(N) % 10
    target_logits_r = torch.randn(size=(N, 10))
    # make the argmax value so high that it will cause the softmax dist to be essentially one-hot
    target_logits_r[torch.arange(N), target_logits] = 1e10

    logits = torch.randn(size=(N, 10))
    a = F.cross_entropy(logits, target_logits)
    b = soft_cross_entropy(logits, target=target_logits_r)
    assert torch.isclose(a, b)
