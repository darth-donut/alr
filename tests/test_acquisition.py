from alr.acquisition import BALD, RandomAcquisition, ICAL
import numpy as np
import torch
import torch.utils.data as torchdata


class FromArray(torchdata.Dataset):
    def __init__(self, arr: np.ndarray):
        self._data = arr

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return self._data.shape[0]


def test_BALD_best_case():
    n_classes = n_forward = 10
    global_counter = 0
    best_point = 42

    def pred_fn(x):
        nonlocal global_counter
        if global_counter == best_point:
            global_counter += 1
            return torch.stack([torch.eye(n_classes) + 1e-6 for _ in range(x.size(0))], dim=1)
        else:
            global_counter += 1
            return torch.softmax(torch.randn(size=(n_forward, x.size(0), n_classes)), dim=-1)
    bald = BALD(
        pred_fn=pred_fn,
        exp=False,
        device='cpu',
        batch_size=1,
    )
    X_pool = FromArray(np.random.randint(0, 100, size=(100, 28)))
    assert bald(X_pool, b=10)[0] == best_point


def test_BALD_worst_case():
    n_classes = n_forward = 10
    global_counter = 0
    worst_point = 42

    def pred_fn(x):
        nonlocal global_counter
        if global_counter == worst_point:
            global_counter += 1
            arr = torch.zeros(size=(n_forward, n_classes), dtype=torch.float32)
            arr[:, np.random.randint(0, n_classes, size=(1,))] = 1
            return torch.stack([arr.clone() + 1e-6 for _ in range(x.size(0))], dim=1)
        else:
            global_counter += 1
            arr = torch.softmax(torch.randn(size=(n_forward, n_classes)), dim=-1)
            return torch.stack([arr.clone() for _ in range(x.size(0))], dim=1)
    bald = BALD(
        pred_fn=pred_fn,
        exp=False,
        device='cpu',
        batch_size=1,
    )
    X_pool = FromArray(np.random.randint(0, 100, size=(100, 28)))
    # need to acquire all points to find the worst!
    idxs = bald(X_pool, b=100)
    assert idxs[-1] == worst_point
