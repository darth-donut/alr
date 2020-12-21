import torch
import numpy as np
import pickle

from pathlib import Path
from alr.data.datasets import Dataset
from alr import MCDropout
import torchvision as tv
import torch.utils.data as torchdata
from alr.utils import stratified_partition, manual_seed


def xlogy(x, y):
    res = x * torch.log(y)
    res[y == 0] = 0.0
    assert torch.isfinite(res).all()
    return res


def get_scores(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        mc_preds: torch.Tensor = torch.cat(
            [model.stochastic_forward(x.to(device)).exp() for x, _ in dataloader], dim=1
        )
    # K N C
    mc_preds = mc_preds.double()
    # N C
    mean_mc_preds = mc_preds.mean(dim=0)

    H = -(mean_mc_preds * torch.log(mean_mc_preds + 1e-5)).sum(dim=1).cpu().numpy()
    H_1 = -(xlogy(mean_mc_preds, mean_mc_preds)).sum(dim=1).cpu().numpy()

    E = (mc_preds * torch.log(mc_preds + 1e-5)).sum(dim=2).mean(dim=0).cpu().numpy()
    E_1 = (xlogy(mc_preds, mc_preds)).sum(dim=2).mean(dim=0).cpu().numpy()

    I = H + E
    I_1 = H_1 + E_1

    assert np.isfinite(I).all()
    assert np.isfinite(I_1).all()

    confidence, argmax = mean_mc_preds.max(dim=1)
    confidence, argmax = confidence.cpu().numpy(), argmax.cpu().numpy()

    assert E.shape == H.shape == I.shape == confidence.shape

    return {
        "average_entropy": -E,
        "predictive_entropy": H,
        "average_entropy2": -E_1,
        "predictive_entropy2": H_1,
        "bald_score": I,
        "bald_score2": I_1,
        "confidence": confidence,
        "class": argmax,
    }


def main(root, reps, result):
    manual_seed(42)
    root = Path(root)
    assert root.is_dir()
    result = Path(result)
    result.mkdir(parents=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kwargs = dict(num_workers=4, pin_memory=True)

    _, mnist_test = Dataset.MNIST.get()
    _, fmnist_test = Dataset.FashionMNIST.get()
    test = torchdata.ConcatDataset((mnist_test, fmnist_test))
    test_loader = torchdata.DataLoader(
        test,
        shuffle=False,
        batch_size=512,
        **kwargs,
    )

    for rep in range(1, reps + 1):
        print(f"=== Rep {rep} of {reps} ===")
        weights = list(root.glob(f"rep_{rep}*"))
        for i, w in enumerate(weights, 1):
            iteration = int(str(w).split("_")[-1][:-3])
            model = MCDropout(Dataset.MNIST.model, forward=20, fast=False).to(device)
            model.load_state_dict(torch.load(w), strict=True)
            scores = get_scores(model, test_loader, device)
            scores_another = get_scores(model, test_loader, device)
            with open(result / f"rep_{rep}_iter_{iteration}.pkl", "wb") as fp:
                pickle.dump((scores, scores_another), fp)


if __name__ == "__main__":
    main(
        "weights",
        reps=6,
        result="scores",
    )
