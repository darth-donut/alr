import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pathlib import Path
from alr.data.datasets import Dataset
from alr import MCDropout
import torchvision as tv
import torch.utils.data as torchdata
from alr.acquisition import _bald_score
from alr.utils import eval_fwd_exp

def main(root, reps, result):
    root = Path(root)
    assert root.is_dir()
    result = Path(result)
    result.mkdir(parents=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    kwargs = dict(num_workers=4, pin_memory=True)

    _, cifar_test = Dataset.CIFAR10.get()
    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])
    svhn_test = tv.datasets.SVHN("data", split="test", transform=transform, download=True)
    test = torchdata.ConcatDataset((cifar_test, svhn_test))
    test_loader = torchdata.DataLoader(
        test, shuffle=False, batch_size=512, **kwargs,
    )
    for rep in range(1, reps + 1):
        print(f"=== Rep {rep} of {reps} ===")
        weights = list(root.glob(f"rep_{rep}*"))
        total = len(weights)
        for i, w in enumerate(weights, 1):
            if i % 5 == 0:
                print(f"Loading weights for {i} of {total}")
            iteration = int(str(w).split("_")[-1][:-3])
            model = MCDropout(Dataset.CIFAR10.model, forward=20, fast=False).to(device)
            model.load_state_dict(torch.load(w), strict=True)
            scores = _bald_score(eval_fwd_exp(model), test_loader, device)
            np.save(result / f"rep_{rep}_iter_{iteration}", scores)

if __name__ == '__main__':
    main("weights",
         reps=6,
         result="bald_scores"
    )

