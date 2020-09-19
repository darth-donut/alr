import torch
import numpy as np
import pickle

from pathlib import Path
from alr.data.datasets import Dataset
from alr import MCDropout
import torch.utils.data as torchdata
from alr.utils import manual_seed, stratified_partition
import torchvision as tv

import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import weight_norm

# PreactResNet18_WNdrop(drop_val=0.3, num_classes=10)
def conv3x3_wn(in_planes, out_planes, stride=1):
    return weight_norm(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False))


class PreActBlock_WNdrop(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(PreActBlock_WNdrop, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3_wn(in_planes, planes, stride)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3_wn(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                weight_norm(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class ResNet_wn(nn.Module):
    def __init__(self, block, num_blocks, drop_val=0.0, num_classes=100):
        super(ResNet_wn, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3_wn(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.drop = drop_val
        self.layer1 = self._make_layer(block, 64, num_blocks[0], dropout_rate=self.drop, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], dropout_rate=self.drop, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], dropout_rate=self.drop, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], dropout_rate=self.drop, stride=2)
        self.linear = weight_norm(nn.Linear(512 * block.expansion, num_classes))

    def _make_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, lin=0, lout=5):
        out = x
        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
        if lin < 2 and lout > 0:
            out = self.layer1(out)
        if lin < 3 and lout > 1:
            out = self.layer2(out)
        if lin < 4 and lout > 2:
            out = self.layer3(out)
        if lin < 5 and lout > 3:
            out = self.layer4(out)
        if lout > 4:
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return F.log_softmax(out, dim=1)


def PreactResNet18_WNdrop(drop_val=0.3, num_classes=10):
    return ResNet_wn(PreActBlock_WNdrop, [2, 2, 2, 2], drop_val=drop_val, num_classes=num_classes)


class Noise(torchdata.Dataset):
    def __init__(self, length: int, channels=3, img_shape=(32, 32)):
        self.length = length
        black = torch.zeros(size=(1, channels, img_shape[0], img_shape[1]))
        white = torch.ones(size=(1, channels, img_shape[0], img_shape[1]))
        # noise
        std = 0.15
        n = length - 2
        weak = torch.randn(size=(n // 2, channels, img_shape[0], img_shape[1])) * std
        strong = torch.randn(size=(n // 2 + (n % 2), channels, img_shape[0], img_shape[1])) * std * 2
        self.data = torch.cat([weak, strong, black, white])
        assert self.data.shape == (length, channels, *img_shape)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx], 0


def xlogy(x, y):
    res = x * torch.log(y)
    res[y == 0] = .0
    assert torch.isfinite(res).all()
    return res


def get_scores(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        mc_preds: torch.Tensor = torch.cat(
            [model.stochastic_forward(x.to(device)).exp() for x, _ in dataloader],
            dim=1
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
        'average_entropy': -E,
        'predictive_entropy': H,
        'average_entropy2': -E_1,
        'predictive_entropy2': H_1,
        'bald_score': I,
        'bald_score2': I_1,
        'confidence': confidence,
        'class': argmax,
    }


def main(root, reps, result):
    manual_seed(42)
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
    subset, _ = stratified_partition(svhn_test, classes=10, size=10_000)
    noise = Noise(length=20)

    with open("subset_idxs.pkl", "wb") as fp:
        pickle.dump(subset.indices, fp)

    # 20,020 in length
    test = torchdata.ConcatDataset((cifar_test, svhn_test, noise))
    test_loader = torchdata.DataLoader(
        test, shuffle=False, batch_size=1, **kwargs,
    )

    for rep in range(1, reps + 1):
        print(f"=== Rep {rep} of {reps} ===")
        weights = list(root.glob(f"rep_{rep}*"))
        total = len(weights)
        for i, w in enumerate(weights, 1):
            if i % 5 == 0:
                print(f"Loading weights for {i} of {total}")
            iteration = int(str(w).split("_")[-1][:-3])
            model = PreactResNet18_WNdrop(drop_val=0.1, num_classes=10)
            # NOTE: it HAS to be fast=True when consistent=True and batch size must be 1
            # because each forward pass must use a different mask for a single sample.
            # The "consistency" comes from the fact that every sample ends up using the
            # same set of (20) forward pass masks. However, even this is reset each time 
            # model.eval() is called
            model = MCDropout(model, forward=20, fast=True, consistent=True).to(device)
            model.load_state_dict(torch.load(w), strict=True)
            scores = get_scores(model, test_loader, device)
            scores_another = get_scores(model, test_loader, device)
            with open(result / f"rep_{rep}_iter_{iteration}.pkl", "wb") as fp:
                pickle.dump((scores, scores_another), fp)


if __name__ == '__main__':
    dataset = "cifar"
    model_name = 'pres'
    main(f"saved_models/{model_name}_{dataset}_aug",
         reps=1,
         result=f"scores/{model_name}_{dataset}",
    )


