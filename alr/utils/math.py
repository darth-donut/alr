import torch
from torch.nn import functional as F


def cross_entropy(
    x: torch.Tensor, y: torch.Tensor, mode="logsoftmax", eps=1e-6
) -> torch.Tensor:
    assert mode in {"logsoftmax", "logits", "softmax"}
    if mode == "logsoftmax":
        return -(x.exp() * y)
    if mode == "logits":
        x, y = F.softmax(x, dim=-1), F.log_softmax(y, dim=-1)
        return -(x * y)
    return -(x * (y + eps).log())


def entropy(x: torch.Tensor, mode="logsoftmax", eps=1e-6) -> torch.Tensor:
    return cross_entropy(x, x, mode, eps)


def entropy_from_logits(x: torch.Tensor) -> torch.Tensor:
    return cross_entropy(x, x, mode="logits")
