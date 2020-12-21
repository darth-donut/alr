import numpy as np
from alr.utils import manual_seed, eval_fwd_exp, timeop
from alr.data.datasets import Dataset
from alr.data import (
    DataManager,
    UnlabelledDataset,
    TransformedDataset,
    disable_augmentation,
)
from alr.acquisition import BALD, RandomAcquisition
from alr.training.samplers import RandomFixedLengthSampler
from alr.training.utils import PLPredictionSaver, EarlyStopper
from alr import MCDropout

from alr.utils import _map_device
from ignite.engine import Engine, create_supervised_evaluator, Events
from ignite.metrics import Loss, Accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import pickle
import torch.utils.data as torchdata
from torch.nn import functional as F
from collections import defaultdict
from pathlib import Path
from torch import nn
import torchvision as tv

from torch.nn.utils import weight_norm

# pre-act-resnet-18 with dropout2D

# PreactResNet18_WNdrop(drop_val=0.3, num_classes=10)
def conv3x3_wn(in_planes, out_planes, stride=1):
    return weight_norm(
        nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
    )


class PreActBlock_WNdrop(nn.Module):
    """Pre-activation version of the BasicBlock."""

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
                weight_norm(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    )
                )
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
        self.layer1 = self._make_layer(
            block, 64, num_blocks[0], dropout_rate=self.drop, stride=1
        )
        self.layer2 = self._make_layer(
            block, 128, num_blocks[1], dropout_rate=self.drop, stride=2
        )
        self.layer3 = self._make_layer(
            block, 256, num_blocks[2], dropout_rate=self.drop, stride=2
        )
        self.layer4 = self._make_layer(
            block, 512, num_blocks[3], dropout_rate=self.drop, stride=2
        )
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
    return ResNet_wn(
        PreActBlock_WNdrop, [2, 2, 2, 2], drop_val=drop_val, num_classes=num_classes
    )


def calc_calib_metrics(loader, model: nn.Module, log_dir, device):
    evaluator = create_supervised_evaluator(model, metrics=None, device=device)
    pds = PLPredictionSaver(log_dir)
    pds.attach(evaluator)
    evaluator.run(loader)


def mixup_data(x, y, alpha, device):
    # from https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# from https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def create_mixup_trainer(model: nn.Module, loss_fn, optimiser, device, alpha):
    def _step(_, batch):
        model.train()
        x, y = batch
        x, y = _map_device([x, y], device)
        xp, y1, y2, lamb = mixup_data(x, y, alpha=alpha, device=device)
        preds = model(xp)
        loss = mixup_criterion(loss_fn, preds, y1, y2, lamb)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        return loss.item()

    return Engine(_step)


def train_model(
    model: nn.Module,
    loss_fn,
    optimiser,
    device,
    alpha,
    train_loader: torchdata.DataLoader,
    val_loader: torchdata.DataLoader,
    patience: int,
    epochs: int,
):
    scheduler = ReduceLROnPlateau(
        optimiser,
        mode="max",
        factor=0.1,
        patience=10,
        verbose=True,
    )

    val_eval = create_supervised_evaluator(
        model, metrics={"acc": Accuracy(), "nll": Loss(F.nll_loss)}, device=device
    )
    history = {"acc": [], "loss": []}

    trainer = create_mixup_trainer(model, loss_fn, optimiser, device, alpha)

    @trainer.on(Events.EPOCH_COMPLETED)
    def _log(e: Engine):
        metrics = val_eval.run(val_loader).metrics
        acc, loss = metrics["acc"], metrics["nll"]
        print(f"\tepoch {e.state.epoch:3}: " f"[val] acc, loss = {acc:.4f}, {loss:.4f}")
        history["acc"].append(acc)
        history["loss"].append(loss)
        scheduler.step(acc)

    es = EarlyStopper(model, patience, trainer, key="acc", mode="max")
    es.attach(val_eval)

    trainer.run(
        train_loader,
        max_epochs=epochs,
    )
    es.reload_best()
    return history


def evaluate_model(loader, model: nn.Module, loss_fn, device):
    evaluator = create_supervised_evaluator(
        model, metrics={"acc": Accuracy(), "loss": Loss(loss_fn)}, device=device
    )
    return evaluator.run(loader).metrics


def main(acq_name, alpha, b, iters, repeats):
    manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kwargs = dict(num_workers=4, pin_memory=True)

    # ========= CONSTANTS ===========
    BATCH_SIZE = 128
    # with early stopping, this'll probably be lesser
    EPOCHS = 200
    # at least have this much points in one epoch (see RandomFixedLengthSampler)
    MIN_TRAIN_LENGTH = 20_000
    VAL_SIZE = 5_000
    LR = 0.1
    MOMENTUM = 0.9
    DECAY = 1e-4
    PATIENCE = 25

    REPEATS = repeats
    ITERS = iters

    # ========= SETUP ===========
    standard_augmentation = [
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.RandomHorizontalFlip(),
    ]
    regular_transform = [
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(*Dataset.CIFAR10.normalisation_params),
    ]
    train, pool, test = Dataset.CIFAR10.get_fixed(raw=True)
    pool, val = torchdata.random_split(pool, (len(pool) - VAL_SIZE, VAL_SIZE))
    pool_idxs = pool.indices

    # apply augmentation to train and pool only
    train = TransformedDataset(
        train, transform=regular_transform, augmentation=standard_augmentation
    )
    # when acquiring(scoring) points, we'll temporarily disable augmentation
    pool = UnlabelledDataset(
        TransformedDataset(
            pool, transform=regular_transform, augmentation=standard_augmentation
        )
    )
    # no augmentation in validation set
    val = TransformedDataset(val, transform=regular_transform)

    val_loader = torchdata.DataLoader(
        val,
        batch_size=512,
        shuffle=False,
        **kwargs,
    )
    test_loader = torchdata.DataLoader(
        test,
        batch_size=512,
        shuffle=False,
        **kwargs,
    )
    accs = defaultdict(list)

    template = f"{acq_name}_{b}_alpha_{alpha}"
    metrics = Path("metrics") / template
    calib_metrics = Path("calib_metrics") / template
    saved_models = Path("saved_models") / template
    metrics.mkdir(parents=True)
    calib_metrics.mkdir(parents=True)
    saved_models.mkdir(parents=True)
    bald_scores = None

    # since we need to know which points were taken for val dataset
    with open(metrics / "pool_idxs.pkl", "wb") as fp:
        pickle.dump(pool_idxs, fp)

    for r in range(1, REPEATS + 1):
        print(f"- [{acq_name} (b={b})] repeat #{r} of {REPEATS}-")
        model = MCDropout(
            PreactResNet18_WNdrop(0.1, num_classes=10), forward=20, fast=False
        ).to(device)
        if acq_name == "bald":
            acq_fn = BALD(eval_fwd_exp(model), device=device, batch_size=512, **kwargs)
        elif acq_name == "random":
            acq_fn = RandomAcquisition()
        else:
            raise Exception("Done goofed.")

        dm = DataManager(train, pool, acq_fn)
        dm.reset()  # this resets pool

        for i in range(1, ITERS + 1):
            model.reset_weights()
            # because in each iteration, we modify the learning rate.
            optimiser = torch.optim.SGD(
                model.parameters(),
                lr=LR,
                momentum=MOMENTUM,
                weight_decay=DECAY,
            )
            train_loader = torchdata.DataLoader(
                dm.labelled,
                batch_size=BATCH_SIZE,
                sampler=RandomFixedLengthSampler(
                    dm.labelled, MIN_TRAIN_LENGTH, shuffle=True
                ),
                **kwargs,
            )
            with timeop() as t:
                history = train_model(
                    model,
                    F.nll_loss,
                    optimiser,
                    device,
                    alpha,
                    train_loader,
                    val_loader,
                    patience=PATIENCE,
                    epochs=EPOCHS,
                )

            # eval
            test_metrics = evaluate_model(test_loader, model, F.nll_loss, device=device)

            print(f"=== Iteration {i} of {ITERS} ({i/ITERS:.2%}) ===")
            print(
                f"\ttrain: {dm.n_labelled}; val: {len(val)}; "
                f"pool: {dm.n_unlabelled}; test: {len(test)}"
            )
            print(f"\t[test] acc: {test_metrics['acc']:.4f}, time: {t}")
            accs[dm.n_labelled].append(test_metrics["acc"])

            # save stuff

            # pool calib
            with dm.unlabelled.tmp_debug():
                pool_loader = torchdata.DataLoader(
                    dm.unlabelled,
                    batch_size=512,
                    shuffle=False,
                    **kwargs,
                )
                calc_calib_metrics(
                    pool_loader,
                    model,
                    calib_metrics / "pool" / f"rep_{r}" / f"iter_{i}",
                    device=device,
                )
            calc_calib_metrics(
                test_loader,
                model,
                calib_metrics / "test" / f"rep_{r}" / f"iter_{i}",
                device=device,
            )

            with open(metrics / f"rep_{r}_iter_{i}.pkl", "wb") as fp:
                payload = {
                    "history": history,
                    "test_metrics": test_metrics,
                    "labelled_classes": dm.unlabelled.labelled_classes,
                    "labelled_indices": dm.unlabelled.labelled_indices,
                    "bald_scores": bald_scores,
                }
                pickle.dump(payload, fp)
            torch.save(model.state_dict(), saved_models / f"rep_{r}_iter_{i}.pt")
            # flush results frequently for the impatient
            with open(template + "_accs.pkl", "wb") as fp:
                pickle.dump(accs, fp)

            # finally, acquire points
            dm.acquire(b=b)


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--acq", choices=["bald", "random"], default="bald")
    args.add_argument("--alpha", default=0.4, type=float)
    args.add_argument("--b", default=10, type=int, help="Batch acq size (default = 10)")
    args.add_argument("--iters", default=199, type=int)
    args.add_argument("--reps", default=1, type=int)
    args = args.parse_args()

    main(args.acq, alpha=args.alpha, b=args.b, iters=args.iters, repeats=args.reps)
