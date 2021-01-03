import torch
import numpy as np
import pickle

from pathlib import Path
from alr.data.datasets import Dataset
from alr import MCDropout
import torch.utils.data as torchdata
from alr.utils import manual_seed, stratified_partition
import torchvision as tv

from models.resnet import resnet18_v2
from models.wide_resnet import WRN28_2_wn
from models.vgg import vgg16_cinic10_bn
from models.pre_resnet_18 import PreactResNet18_WNdrop
from models.efficient import EfficientNet


class Noise(torchdata.Dataset):
    def __init__(self, length: int, channels=3, img_shape=(32, 32)):
        self.length = length
        black = torch.zeros(size=(1, channels, img_shape[0], img_shape[1]))
        white = torch.ones(size=(1, channels, img_shape[0], img_shape[1]))
        # noise
        std = 0.15
        n = length - 2
        weak = torch.randn(size=(n // 2, channels, img_shape[0], img_shape[1])) * std
        strong = (
            torch.randn(size=(n // 2 + (n % 2), channels, img_shape[0], img_shape[1]))
            * std
            * 2
        )
        self.data = torch.cat([weak, strong, black, white])
        assert self.data.shape == (length, channels, *img_shape)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx], 0


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


def main(root, model_name, reps, result):
    manual_seed(42)
    root = Path(root)
    assert root.is_dir()
    result = Path(result)
    result.mkdir(parents=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kwargs = dict(num_workers=4, pin_memory=True)

    _, cifar_test = Dataset.CIFAR10.get()
    transform = tv.transforms.Compose(
        [
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ]
    )
    svhn_test = tv.datasets.SVHN(
        "data", split="test", transform=transform, download=True
    )
    subset, _ = stratified_partition(svhn_test, classes=10, size=10_000)
    noise = Noise(length=20)

    with open("subset_idxs.pkl", "wb") as fp:
        pickle.dump(subset.indices, fp)

    # 20,020 in length
    test = torchdata.ConcatDataset((cifar_test, svhn_test, noise))
    test_loader = torchdata.DataLoader(
        test,
        shuffle=False,
        batch_size=512,
        **kwargs,
    )

    for rep in range(1, reps + 1):
        print(f"=== Rep {rep} of {reps} ===")
        weights = list(root.glob(f"rep_{rep}*"))
        total = len(weights)
        for i, w in enumerate(weights, 1):
            if i % 5 == 0:
                print(f"Loading weights for {i} of {total}")
            iteration = int(str(w).split("_")[-1][:-3])
            if model_name == "vgg":
                # 1D (0.5 by default in FC)
                model = vgg16_cinic10_bn(num_classes=10)
            elif model_name == "wres":
                # 1d weights + fc
                model = WRN28_2_wn(num_classes=10, dropout=0.5)
            elif model_name == "res":
                # 2d
                model = resnet18_v2(
                    num_classes=10, dropout_rate=0.3, fc_dropout_rate=0.3
                )
            elif model_name == "pres":
                # 2d
                model = PreactResNet18_WNdrop(drop_val=0.3, num_classes=10)
            elif model_name == "13cnn":
                model = Dataset.CIFAR10.model
            elif model_name == "eff":
                model = EfficientNet(version=3, dropout_rate=0.5, num_classes=10)
            else:
                raise ValueError(f"Unknown model architecture {model_name}.")
            model = MCDropout(model, forward=20, fast=False).to(device)
            model.load_state_dict(torch.load(w), strict=True)
            scores = get_scores(model, test_loader, device)
            scores_another = get_scores(model, test_loader, device)
            with open(result / f"rep_{rep}_iter_{iteration}.pkl", "wb") as fp:
                pickle.dump((scores, scores_another), fp)


if __name__ == "__main__":
    dataset = "cifar"
    for model_name in ["13cnn", "eff", "pres", "res", "vgg", "wres"]:
        main(
            f"saved_models/{model_name}_{dataset}_aug",
            model_name=model_name,
            reps=1,
            result=f"scores/{model_name}_{dataset}",
        )
