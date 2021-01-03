from alr.data.datasets import Dataset
from alr.data import DataManager, UnlabelledDataset
from alr.acquisition import RandomAcquisition
from alr.training import Trainer
from alr.training.samplers import RandomFixedLengthSampler
from alr.training.utils import PLPredictionSaver
from alr.utils import manual_seed, timeop, stratified_partition
from alr import MCDropout

from models.resnet import resnet18_v2
from models.wide_resnet import WRN28_2_wn
from models.vgg import vgg16_cinic10_bn
from models.pre_resnet_18 import PreactResNet18_WNdrop
from models.efficient import EfficientNet

import torch
import pickle
import torch.utils.data as torchdata
from torch.nn import functional as F
from collections import defaultdict
from ignite.engine import create_supervised_evaluator
from pathlib import Path
from torch import nn


def calc_calib_metrics(loader, model: nn.Module, log_dir, device):
    evaluator = create_supervised_evaluator(model, metrics=None, device=device)
    pds = PLPredictionSaver(log_dir)
    pds.attach(evaluator)
    evaluator.run(loader)


def main(model_name, aug, dataset, iters, repeats):
    manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kwargs = dict(num_workers=4, pin_memory=True)

    # ========= CONSTANTS ===========
    BATCH_SIZE = 64
    # with early stopping, this'll probably be lesser
    EPOCHS = 200
    # at least have this much points in one epoch (see RandomFixedLengthSampler)
    MIN_TRAIN_LENGTH = 20_000
    VAL_SIZE = 5_000

    REPEATS = repeats
    ITERS = iters

    # ========= SETUP ===========
    if dataset == "cifar":
        train, test = Dataset.CIFAR10.get(augmentation=aug)
    elif dataset == "cinic":
        train, test = Dataset.CINIC10.get(augmentation=aug)
    else:
        raise ValueError("dataset only accepts two arguments: cinic or cifar")
    train, pool = stratified_partition(train, classes=10, size=20)
    pool_idxs = pool.indices[:]
    pool, val = torchdata.random_split(pool, (len(pool) - VAL_SIZE, VAL_SIZE))
    pool = UnlabelledDataset(pool)
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

    template = f"{model_name}_{dataset}" + ("_aug" if aug else "")
    metrics = Path("metrics") / template
    calib_metrics = Path("calib_metrics") / template
    saved_models = Path("saved_models") / template
    metrics.mkdir(parents=True)
    calib_metrics.mkdir(parents=True)
    saved_models.mkdir(parents=True)

    # since we need to know which points were taken for val dataset
    with open(metrics / "pool_idxs.pkl", "wb") as fp:
        pickle.dump((pool_idxs, pool._dataset.indices), fp)

    for r in range(1, REPEATS + 1):
        print(f"- [{model_name}] repeat #{r} of {REPEATS}-")
        if model_name == "vgg":
            # 1D (0.5 by default in FC)
            model = vgg16_cinic10_bn(num_classes=10)
        elif model_name == "wres":
            # 1d weights + fc
            model = WRN28_2_wn(num_classes=10, dropout=0.5)
        elif model_name == "res":
            # 2d
            model = resnet18_v2(num_classes=10, dropout_rate=0.3, fc_dropout_rate=0.3)
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
        dm = DataManager(train, pool, RandomAcquisition())
        dm.reset()  # this resets pool

        for i in range(1, ITERS + 1):
            model.reset_weights()
            trainer = Trainer(
                model,
                F.nll_loss,
                optimiser="Adam",
                patience=10,
                reload_best=True,
                device=device,
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
                history = trainer.fit(train_loader, val_loader, epochs=EPOCHS)

            # eval
            test_metrics = trainer.evaluate(test_loader)
            print(f"=== Iteration {i} of {ITERS} ({i / ITERS:.2%}) ===")
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
                }
                pickle.dump(payload, fp)
            torch.save(model.state_dict(), saved_models / f"rep_{r}_iter_{i}.pt")
            # flush results frequently for the impatient
            with open(template + "_accs.pkl", "wb") as fp:
                pickle.dump(accs, fp)
            dm.acquire(b=200)


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--model", choices=["vgg", "wres", "res", "pres", "13cnn", "eff"])
    args.add_argument("--data", choices=["cinic", "cifar"])
    args.add_argument("--iters", default=11, type=int)
    args.add_argument("--reps", default=1, type=int)
    args.add_argument("--aug", action="store_true")
    args = args.parse_args()
    main(args.model, args.aug, args.data, args.iters, args.reps)
