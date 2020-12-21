from typing import Optional, Tuple

import torch
import numpy as np
import math
import torch.utils.data as torchdata
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from alr.training.pl_mixup import (
    IndexMarker,
    PDS,
    onehot_transform,
    create_warmup_trainer,
    create_plmixup_trainer,
    PLMixupTrainer,
)
from alr.training.progress_bar.ignite_progress_bar import ProgressBar
from alr.training.samplers import RandomFixedLengthSampler, MinLabelledSampler
from alr.training.utils import EarlyStopper, PerformanceTracker


class CyclicPLMixupTrainer(PLMixupTrainer):
    def fit(
        self,
        train: torchdata.Dataset,
        val: torchdata.Dataset,
        pool: torchdata.Dataset,
        epochs: Optional[Tuple[int, int, int]] = (50, 400, 60),
    ):
        if isinstance(self._patience, int):
            pat1, pat2 = self._patience
        else:
            pat1, pat2 = self._patience[0], self._patience[1]
        history = {
            "val_loss": [],
            "val_acc": [],
            "override_acc": [],
        }
        optimiser = self._instantiate_optimiser()
        train = PDS(
            IndexMarker(train, mark=IndexMarker.LABELLED),
            transform=self._train_transform,
            augmentation=self._data_augmentation,
            target_transform=onehot_transform(self._num_classes),
        )
        pool = PDS(
            IndexMarker(pool, mark=IndexMarker.PSEUDO_LABELLED),
            transform=self._train_transform,
            augmentation=self._data_augmentation,
        )
        val = PDS(
            IndexMarker(val, mark=None),
            transform=self._test_transform,
        )
        val._with_metadata = False
        train_loader = torchdata.DataLoader(
            train,
            batch_size=self._batch_size,
            sampler=RandomFixedLengthSampler(train, self._rfls_length, shuffle=True),
            **self._loader_kwargs,
        )
        pool_loader = torchdata.DataLoader(
            pool, batch_size=512, shuffle=False, **self._loader_kwargs
        )
        val_loader = torchdata.DataLoader(
            val, batch_size=512, shuffle=False, **self._loader_kwargs
        )
        pbar = ProgressBar(desc=lambda _: "Stage 1")

        # warm up
        with train.no_fluff():
            val_eval = create_supervised_evaluator(
                self._model,
                metrics={"acc": Accuracy(), "loss": Loss(F.nll_loss)},
                device=self._device,
            )
            trainer = create_warmup_trainer(
                self._model,
                optimiser=optimiser,
                device=self._device,
            )
            es = EarlyStopper(
                self._model, patience=pat1, trainer=trainer, key="acc", mode="max"
            )
            es.attach(val_eval)

            @trainer.on(Events.EPOCH_COMPLETED)
            def _log(e: Engine):
                metrics = val_eval.run(val_loader).metrics
                acc, loss = metrics["acc"], metrics["loss"]
                pbar.log_message(
                    f"\tStage 1 epoch {e.state.epoch}/{e.state.max_epochs} "
                    f"[val] acc, loss = "
                    f"{acc:.4f}, {loss:.4f}"
                )
                history["val_acc"].append(acc)
                history["val_loss"].append(loss)

            pbar.attach(trainer)
            trainer.run(train_loader, max_epochs=epochs[0])
            es.reload_best()

        # pseudo-label points
        with pool.no_augmentation():
            with pool.no_fluff():
                pseudo_labels = []
                with torch.no_grad():
                    self._model.eval()
                    for x, _ in pool_loader:
                        x = x.to(self._device)
                        # add (softmax) probability, hence .exp()
                        pseudo_labels.append(self._model(x).exp().detach().cpu())
        pool.override_targets(torch.cat(pseudo_labels))
        plab_acc = pool.override_accuracy
        pbar.log_message(f"\t*End of stage 1*: overridden labels' acc: {plab_acc}")
        history["override_acc"].append(plab_acc)

        # start training with PL
        full_dataset = torchdata.ConcatDataset((train, pool))
        fds_loader = torchdata.DataLoader(
            full_dataset,
            batch_sampler=MinLabelledSampler(
                train,
                pool,
                batch_size=self._batch_size,
                min_labelled=self._min_labelled,
            ),
            **self._loader_kwargs,
        )
        val_eval = create_supervised_evaluator(
            self._model,
            metrics={"acc": Accuracy(), "loss": Loss(F.nll_loss)},
            device=self._device,
        )
        optimiser = self._instantiate_optimiser()
        scheduler = ReduceLROnPlateau(
            optimiser,
            mode="max",
            factor=0.1,
            patience=self._lr_patience,
            verbose=True,
            min_lr=1e-3,
        )
        trainer = create_plmixup_trainer(
            self._model,
            optimiser,
            pool,
            alpha=self._alpha,
            num_classes=self._num_classes,
            log_dir=self._log_dir,
            device=self._device,
        )
        es = EarlyStopper(
            self._model, patience=pat2, trainer=trainer, key="acc", mode="max"
        )
        es.attach(val_eval)

        pbar = ProgressBar(desc=lambda _: "Stage 2")

        @trainer.on(Events.EPOCH_COMPLETED)
        def _log(e: Engine):
            metrics = val_eval.run(val_loader).metrics
            acc, loss = metrics["acc"], metrics["loss"]
            pbar.log_message(
                f"\tEpoch {e.state.epoch}/{e.state.max_epochs} "
                f"[val] acc, loss = "
                f"{acc:.4f}, {loss:.4f}"
            )
            history["val_acc"].append(acc)
            history["val_loss"].append(loss)
            history["override_acc"].append(pool.override_accuracy)
            scheduler.step(acc)

        pbar.attach(trainer)
        trainer.run(fds_loader, max_epochs=epochs[1])
        es.reload_best()

        ####
        # save the best weight so far just in case we wander off
        pt = PerformanceTracker(self._model, patience=0)
        # es.reload_best() would've given us this accuracy, so we store it now
        # before restarting the SGD learning rate in case we never recover from moving away from this local minima
        pt.step(max(history["val_acc"]))

        # reset SGD learning rate to 0.2 and start cyclic learning
        init_lr = 0.2
        optimiser = torch.optim.SGD(
            self._model.parameters(), lr=init_lr, momentum=0.9, weight_decay=1e-4
        )

        # budget number of epochs
        B = epochs[2]
        # number of snapshots
        M = 6
        # total number of training iterations for all B epochs:
        #  len(fds_loader) = number of iterations need for ONE epoch
        T = len(fds_loader) * B
        print("Starting cyclic learning")
        trainer = create_plmixup_trainer(
            self._model,
            optimiser,
            pool,
            alpha=self._alpha,
            num_classes=self._num_classes,
            log_dir=self._log_dir,
            device=self._device,
        )
        val_eval = create_supervised_evaluator(
            self._model,
            metrics={"acc": Accuracy(), "loss": Loss(F.nll_loss)},
            device=self._device,
        )

        @trainer.on(Events.EPOCH_COMPLETED)
        def _log2(e: Engine):
            metrics = val_eval.run(val_loader).metrics
            acc, loss = metrics["acc"], metrics["loss"]
            print(
                f"\tEpoch {e.state.epoch}/{e.state.max_epochs} "
                f"[val] acc, loss = "
                f"{acc:.4f}, {loss:.4f}"
            )
            history["val_acc"].append(acc)
            history["val_loss"].append(loss)
            history["override_acc"].append(pool.override_accuracy)
            pt.step(acc)

        @trainer.on(Events.ITERATION_COMPLETED)
        def _anneal(e: Engine):
            iteration = e.state.iteration
            assert iteration > 0
            for param_group in optimiser.param_groups:
                param_group["lr"] = cyclic_annealer(iteration, T, M, init_lr)

        trainer.run(fds_loader, max_epochs=B)
        # always want the best set of weights:
        #  if the cyclic learning scheduler ended up with better weights, use it, otherwise,
        #  revert to the set of weights before starting cyclic learning
        pt.reload_best()
        soft_label_history = pool.label_history
        self.soft_label_history = torch.stack(soft_label_history, dim=0)
        return history


def cyclic_annealer(t, T, M, init_lr=0.2):
    return (init_lr / 2) * (
        np.cos((np.pi * np.mod(t - 1, math.ceil(T / M))) / math.ceil(T / M)) + 1
    )
