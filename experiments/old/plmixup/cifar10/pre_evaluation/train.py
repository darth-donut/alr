# TODO(run): pre-evaluation SSL
from alr.training import Trainer
from alr.training.pl_mixup import PLMixupTrainer, temp_ds_transform
from alr.training.samplers import RandomFixedLengthSampler
from alr.utils import manual_seed, eval_fwd_exp, timeop
from alr.data.datasets import Dataset
from alr.data import DataManager, UnlabelledDataset
from alr.acquisition import BALD, RandomAcquisition, _bald_score
from alr.training.utils import PLPredictionSaver
from alr import MCDropout, ALRModel

import torch
import pickle
import torch.utils.data as torchdata
import torchvision as tv
from collections import defaultdict
from ignite.engine import create_supervised_evaluator
from pathlib import Path
from torch import nn
from torch.nn import functional as F


class Net(ALRModel):
    def __init__(self, model):
        super(Net, self).__init__()
        self.model = model
        self.snap()

    def forward(self, x):
        return self.model(x)

def calc_calib_metrics(loader, model: nn.Module, log_dir, device):
    evaluator = create_supervised_evaluator(
        model, metrics=None, device=device
    )
    pds = PLPredictionSaver(log_dir)
    pds.attach(evaluator)
    evaluator.run(loader)


def main(acq_name: str,
         alpha: float,
         b: int,
         augment: bool,
         iters: int,
         repeats: int):
    manual_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    kwargs = dict(num_workers=4, pin_memory=True)

    # ========= CONSTANTS ===========
    BATCH_SIZE = 100
    BATCH_SIZE_SUPERVISED = 64
    # at least have this much points in one epoch (see RandomFixedLengthSampler)
    MIN_TRAIN_LENGTH = 20_000
    VAL_SIZE = 5_000
    MIN_LABELLED = 16
    # stage 1 and stage 2 patience
    PATIENCE = (5, 25)
    # how many epochs before LR is reduced
    LR_PATIENCE = 10
    # stage 1 and stage 2 of PL mixup training
    EPOCHS = (100, 400)
    # for supervised training
    EPOCHS_SUPERVISED = 200

    REPEATS = repeats
    ITERS = iters

    # ========= SETUP ===========
    train, pool, test = Dataset.CIFAR10.get_fixed(raw=True)
    pool, val = torchdata.random_split(pool, (len(pool) - VAL_SIZE, VAL_SIZE))
    pool = UnlabelledDataset(pool)
    train_transform = test_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(
            *Dataset.CIFAR10.normalisation_params
        ),
    ])
    test_loader = torchdata.DataLoader(
        test, batch_size=512, shuffle=False, **kwargs,
    )
    val_loader = torchdata.DataLoader(
        temp_ds_transform(test_transform, with_targets=True)(val),
        batch_size=512, shuffle=False, **kwargs,
    )
    ds_transformer = temp_ds_transform(test_transform)
    if augment:
        data_augmentation = Dataset.CIFAR10.get_augmentation
    else:
        data_augmentation = None
    accs = defaultdict(list)

    template = f"{acq_name}_{b}_alpha_{alpha}" + ("_aug" if augment else "")
    metrics = Path("metrics") / template
    calib_metrics = Path("calib_metrics") / template
    saved_models = Path("saved_models") / template
    metrics.mkdir(parents=True)
    calib_metrics.mkdir(parents=True)
    saved_models.mkdir(parents=True)
    bald_scores = None

    # since we need to know which points were taken for val dataset
    with open(metrics / "pool_idxs.pkl", "wb") as fp:
        pickle.dump(pool._dataset.indices, fp)

    for r in range(1, REPEATS + 1):
        print(f"- [{acq_name} (b={b})] repeat #{r} of {REPEATS}-")
        model = MCDropout(Dataset.CIFAR10.model, forward=20, fast=False).to(device)
        # no mcdropout
        model_regular = Net(Dataset.CIFAR10.model).to(device)
        if acq_name == "bald":
            acq_fn = BALD(eval_fwd_exp(model), device=device, batch_size=512, **kwargs)
        elif acq_name == "random":
            acq_fn = RandomAcquisition()
        else:
            raise Exception("Done goofed.")

        dm = DataManager(train, pool, acq_fn)
        dm.reset()  # this resets pool

        for i in range(1, ITERS + 1):
            model_regular.reset_weights()
            trainer = PLMixupTrainer(
                model_regular, 'SGD', train_transform, test_transform,
                {'lr': .1, 'momentum': .9, 'weight_decay': 1e-4},
                kwargs, log_dir=None,
                rfls_length=MIN_TRAIN_LENGTH, alpha=alpha,
                min_labelled=MIN_LABELLED,
                data_augmentation=data_augmentation,
                batch_size=BATCH_SIZE,
                patience=PATIENCE, lr_patience=LR_PATIENCE,
                device=device
            )
            with dm.unlabelled.tmp_debug():
                with timeop() as t:
                    history = trainer.fit(
                        dm.labelled, val, dm.unlabelled, epochs=EPOCHS
                    )

            # eval
            test_metrics = trainer.evaluate(test_loader)
            print(f"=== Iteration {i} of {ITERS} ({i / ITERS:.2%}) ===")
            print(f"\ttrain: {dm.n_labelled}; val: {len(val)}; "
                  f"pool: {dm.n_unlabelled}; test: {len(test)}")
            print(f"\t[test] acc: {test_metrics['acc']:.4f}, time: {t}")
            accs[dm.n_labelled].append(test_metrics['acc'])

            # save stuff

            # pool calib
            with dm.unlabelled.tmp_debug():
                pool_loader = torchdata.DataLoader(
                    temp_ds_transform(test_transform, with_targets=True)(dm.unlabelled),
                    batch_size=512, shuffle=False,
                    **kwargs,
                )
                calc_calib_metrics(
                    pool_loader, model_regular, calib_metrics / "pool" / f"rep_{r}" / f"iter_{i}",
                    device=device
                )
            calc_calib_metrics(
                test_loader, model_regular, calib_metrics / "test" / f"rep_{r}" / f"iter_{i}",
                device=device
            )

            with open(metrics / f"rep_{r}_iter_{i}.pkl", "wb") as fp:
                payload = {
                    'history': history,
                    'test_metrics': test_metrics,
                    'labelled_classes': dm.unlabelled.labelled_classes,
                    'labelled_indices': dm.unlabelled.labelled_indices,
                    'bald_scores': bald_scores,

                }
                pickle.dump(payload, fp)
            torch.save(model_regular.state_dict(), saved_models / f"rep_{r}_iter_{i}.pt")
            # flush results frequently for the impatient
            with open(template + "_accs.pkl", "wb") as fp:
                pickle.dump(accs, fp)

            if i == ITERS: continue

            # before acquiring points, train using regular trainer
            model.reset_weights()
            trainer = Trainer(
                model, F.nll_loss, optimiser='Adam',
                patience=3, reload_best=True, device=device
            )
            train_loader = torchdata.DataLoader(
                temp_ds_transform(train_transform, with_targets=True)(dm.labelled),
                batch_size=BATCH_SIZE_SUPERVISED,
                sampler=RandomFixedLengthSampler(dm.labelled, MIN_TRAIN_LENGTH, shuffle=True),
                **kwargs
            )
            with timeop() as t:
                history = trainer.fit(train_loader, val_loader, epochs=EPOCHS_SUPERVISED)

            test_metrics = trainer.evaluate(test_loader)
            print(f"\ttrain: {dm.n_labelled} "
                  f"[test] supervised acc: {test_metrics['acc']:.4f}, time: {t}")

            # pool calib
            with dm.unlabelled.tmp_debug():
                pool_loader = torchdata.DataLoader(
                    temp_ds_transform(test_transform, with_targets=True)(dm.unlabelled),
                    batch_size=512, shuffle=False,
                    **kwargs,
                )
                calc_calib_metrics(
                    pool_loader, model, calib_metrics / "supervised" / "pool" / f"rep_{r}" / f"iter_{i}",
                    device=device
                )
            calc_calib_metrics(
                test_loader, model, calib_metrics / "supervised" / "test" / f"rep_{r}" / f"iter_{i}",
                device=device
            )

            with open(metrics / f"supervised_rep_{r}_iter_{i}.pkl", "wb") as fp:
                payload = {
                    'history': history,
                    'test_metrics': test_metrics,
                }
                pickle.dump(payload, fp)

            torch.save(model.state_dict(), saved_models / f"supervised_rep_{r}_iter_{i}.pt")

            # finally, acquire points
            # transform pool samples toTensor and normalise them (since we used raw above!)
            acquired_idxs, acquired_ds = dm.acquire(b=b, transform=ds_transformer)
            acquired_ds = ds_transformer(acquired_ds)
            # if bald, store ALL bald scores and the acquired idx so we can map the top b scores
            # to the b acquired_idxs
            if acq_name == "bald":
                # acquired_idxs has the top b scores from recent_score
                bald_scores = (acquired_idxs, acq_fn.recent_score)
            # if RA, then store the acquired indices and their associated bald scores
            else:
                # compute bald scores of Random Acq. points
                bald_scores = _bald_score(
                    pred_fn=eval_fwd_exp(model),
                    dataloader=torchdata.DataLoader(
                        acquired_ds, batch_size=512,
                        shuffle=False,  # don't shuffle to get 1-1 pairing with acquired_idxs
                        **kwargs,
                    ),
                    device=device,
                )
                assert acquired_idxs.shape[0] == bald_scores.shape[0], \
                    f"Acquired idx length {acquired_idxs.shape[0]} does not" \
                    f" match bald scores length {bald_scores.shape[0]}"
                bald_scores = list(zip(acquired_idxs, bald_scores))


if __name__ == '__main__':
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--acq", choices=['bald', 'random'], default='bald')
    args.add_argument("--alpha", type=float, default=0.4)
    args.add_argument("--b", default=10, type=int, help="Batch acq size (default = 10)")
    args.add_argument("--augment", action='store_true')
    args.add_argument("--iters", default=199, type=int)
    args.add_argument("--reps", default=1, type=int)
    args = args.parse_args()

    main(
        args.acq,
        alpha=args.alpha,
        b=args.b,
        augment=args.augment,
        iters=args.iters,
        repeats=args.reps
    )
