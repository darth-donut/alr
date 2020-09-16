from alr.utils import manual_seed, timeop, stratified_partition
from alr.data.datasets import Dataset
from alr.training import Trainer
from alr.training.samplers import RandomFixedLengthSampler
from alr.training.utils import PLPredictionSaver
from alr import MCDropout

import torch
import pickle
import torch.utils.data as torchdata
from torch.nn import functional as F
from ignite.engine import create_supervised_evaluator
from pathlib import Path
from torch import nn
from torch.nn.utils import weight_norm



class CNN13(nn.Module):
    """
    CNN from Mean Teacher paper
    # taken from: https://github.com/EricArazo/PseudoLabeling/blob/2fbbbd3ca648cae453e3659e2e2ed44f71be5906/utils_pseudoLab/ssl_networks.py
    """

    def __init__(self, num_classes=10, drop_prob=.5):
        super(CNN13, self).__init__()
        self.activation = nn.LeakyReLU(0.1)
        self.conv1a = weight_norm(nn.Conv2d(3, 128, 3, padding=1))
        self.bn1a = nn.BatchNorm2d(128)
        self.conv1b = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1b = nn.BatchNorm2d(128)
        self.conv1c = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1c = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop = nn.Dropout(drop_prob)

        self.conv2a = weight_norm(nn.Conv2d(128, 256, 3, padding=1))
        self.bn2a = nn.BatchNorm2d(256)
        self.conv2b = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2b = nn.BatchNorm2d(256)
        self.conv2c = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2c = nn.BatchNorm2d(256)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)

        self.conv3a = weight_norm(nn.Conv2d(256, 512, 3, padding=0))
        self.bn3a = nn.BatchNorm2d(512)
        self.conv3b = weight_norm(nn.Conv2d(512, 256, 1, padding=0))
        self.bn3b = nn.BatchNorm2d(256)
        self.conv3c = weight_norm(nn.Conv2d(256, 128, 1, padding=0))
        self.bn3c = nn.BatchNorm2d(128)
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)

        self.fc1 = weight_norm(nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.activation(self.bn1a(self.conv1a(x)))
        x = self.activation(self.bn1b(self.conv1b(x)))
        x = self.activation(self.bn1c(self.conv1c(x)))
        x = self.mp1(x)
        x = self.drop(x)

        x = self.activation(self.bn2a(self.conv2a(x)))
        x = self.activation(self.bn2b(self.conv2b(x)))
        x = self.activation(self.bn2c(self.conv2c(x)))
        x = self.mp2(x)
        x = self.drop(x)

        x = self.activation(self.bn3a(self.conv3a(x)))
        x = self.activation(self.bn3b(self.conv3b(x)))
        x = self.activation(self.bn3c(self.conv3c(x)))
        x = self.ap3(x)
        x = x.view(-1, 128)
        return F.log_softmax(self.fc1(x), dim=-1)

class VGG(nn.Module):
    """VGG with BatchNorm performs best.
    We only add MCDropout in the classifier head (where VGG used dropout before, too)."""

    def __init__(self, features, num_classes=10, init_weights=True, smaller_head=False):
        super().__init__()

        self.features = features
        if smaller_head:
            # self.avgpool = nn.Identity()
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512 * 1 * 1, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(True),
                nn.Linear(512, num_classes),
            )
        else:
            # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                # nn.Linear(512 * 7 * 7, 4096),
                nn.Linear(512 * 1 * 1, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )

        if init_weights:
            self.apply(self.initialize_weights)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x

    @staticmethod
    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "D2D_03":
            pass
        elif v == "D2D_04":
            pass  # layers += [mc_dropout.MCDropout2d(0.4)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

def _vgg(cfg, batch_norm, **kwargs):
    return VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)

def vgg16_cinic10_bn(**kwargs):
    """
    VGG 16-layer model (configuration "D") with batch normalization
    Inspired by: https://github.com/geifmany/cifar-vgg/blob/master/cifar100vgg.py to follow
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7486599 and then gave up on Dropout in Conv layers
    and just used the smaller classifier head and reduced dropout.
    """
    return _vgg(
        cfg="D",
        batch_norm=True,
        smaller_head=True,
        **kwargs,
    )




def calc_calib_metrics(loader, model: nn.Module, log_dir, device):
    evaluator = create_supervised_evaluator(
        model, metrics=None, device=device
    )
    pds = PLPredictionSaver(log_dir)
    pds.attach(evaluator)
    evaluator.run(loader)


def main(model_name, repeats):
    manual_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    kwargs = dict(num_workers=4, pin_memory=True)

    # ========= CONSTANTS ===========
    BATCH_SIZE = 64
    # with early stopping, this'll probably be lesser
    EPOCHS = 200
    # at least have this much points in one epoch (see RandomFixedLengthSampler)
    MIN_TRAIN_LENGTH = 20_000
    VAL_SIZE = 5_000

    REPEATS = repeats

    # ========= SETUP ===========
    train, test = Dataset.CINIC10.get(augmentation=True)
    train, val = torchdata.random_split(train, (len(train) - VAL_SIZE, VAL_SIZE))
    val_loader = torchdata.DataLoader(
        val, batch_size=512, shuffle=False, **kwargs,
    )
    test_loader = torchdata.DataLoader(
        test, batch_size=512, shuffle=False, **kwargs,
    )
    accs = []

    template = f"{model_name}"
    metrics = Path("metrics") / template
    calib_metrics = Path("calib_metrics") / template
    saved_models = Path("saved_models") / template
    metrics.mkdir(parents=True)
    calib_metrics.mkdir(parents=True)
    saved_models.mkdir(parents=True)

    # since we need to know which points were taken for val dataset
    with open(metrics / "train_idxs.pkl", "wb") as fp:
        pickle.dump(train.indices, fp)

    for r in range(1, REPEATS + 1):
        if model_name == "vgg":
            model = MCDropout(vgg16_cinic10_bn(num_classes=10), forward=20, fast=False).to(device)
        elif model_name == "13cnn":
            model = MCDropout(CNN13(), forward=20, fast=False).to(device)
        else:
            raise ValueError(f"Unknown model {model_name}.")
        trainer = Trainer(
            model, F.nll_loss, optimiser='Adam',
            patience=6, reload_best=True, device=device
        )
        train_loader = torchdata.DataLoader(
            train, batch_size=BATCH_SIZE,
            sampler=RandomFixedLengthSampler(train, MIN_TRAIN_LENGTH, shuffle=True),
            **kwargs
        )
        with timeop() as t:
            history = trainer.fit(train_loader, val_loader, epochs=EPOCHS)

        # eval
        test_metrics = trainer.evaluate(test_loader)
        print(f"=== Repeat {r} of {REPEATS} ===")
        print(f"\ttrain: {len(train)}; val: {len(val)}; "
              f"test: {len(test)}")
        print(f"\t[test] acc: {test_metrics['acc']:.4f}, time: {t}")
        accs.append(test_metrics['acc'])

        # save stuff
        # test calib
        calc_calib_metrics(
            test_loader, model, calib_metrics / "test" / f"rep_{r}",
            device=device
        )

        with open(metrics / f"rep_{r}.pkl", "wb") as fp:
            payload = {
                'history': history,
                'test_metrics': test_metrics,
            }
            pickle.dump(payload, fp)

        torch.save(model.state_dict(), saved_models / f"rep_{r}.pt")

        # flush results frequently for the impatient
        with open(template + "_accs.pkl", "wb") as fp:
            pickle.dump(accs, fp)

if __name__ == '__main__':
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--model", choices=['13cnn', 'vgg'])
    args.add_argument("--reps", default=1, type=int)
    args = args.parse_args()

    main(args.model, repeats=args.reps)

