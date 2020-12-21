import os

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib import cm

import io
import torch
from alr.utils import savefig


custom_lines = [
    Line2D([0], [0], color="red", lw=4),
    Line2D([0], [0], color="blue", lw=4),
]

# https://github.com/pytorch/pytorch/issues/16797
class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


# HP params and Augment vs No Augment
alpha = [1, 0.1, 0.4, 4]
drop = [0.1, 0.5]
root = Path(
    "/Users/harry/Documents/workspace/thesis/experiments/plmixup/cifar10/tryouts/explore/results"
)
cmap = cm.get_cmap("tab10")

i = 0
for a in alpha:
    for d in drop:
        for aug in ["_aug", ""]:
            fname = root / f"SGD_alpha_{a}_drop_{d}{aug}" / "metrics.pkl"
            if not fname.exists():
                continue
            with open(fname, "rb") as fp:
                accs = CPUUnpickler(fp).load()
            if not aug:
                plt.plot(accs["val_acc"], color=cmap(i), linestyle="dotted")
            else:
                plt.plot(
                    accs["val_acc"],
                    color=cmap(i),
                    label=fr"$\alpha = {a}$; $\theta$ = {d}",
                    linestyle="solid",
                )
            plt.ylim(top=1)
        i += 1
plt.title("CIFAR-10 validation accuracy\n1,000 class-balanced points")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(loc="upper left", ncol=3, fontsize="small")
plt.grid()
savefig("/Users/harry/Documents/workspace/thesis/figures/4/eph2_pl_mixup_hp_search.pdf")


# Jitter vs no Jitter
with open(
    "/Users/harry/Documents/workspace/thesis/experiments/plmixup/cifar10/tryouts/with_jitter/results/SGD_alpha_1_drop_0.5_size_1000_jitter/metrics.pkl",
    "rb",
) as fp:
    metrics = CPUUnpickler(fp).load()
plt.plot(metrics["val_acc"], label="Standard Aug. + Colour Jittering")
with open(
    "/Users/harry/Documents/workspace/thesis/experiments/plmixup/cifar10/tryouts/without_jitter/results/SGD_alpha_0.1_drop_0.5_size_1000_aug/metrics.pkl",
    "rb",
) as fp:
    metrics = CPUUnpickler(fp).load()
plt.plot(metrics["val_acc"], label="Standard Aug.")
with open(
    "/Users/harry/Documents/workspace/thesis/experiments/plmixup/cifar10/tryouts/explore/results/SGD_alpha_0.1_drop_0.5/metrics.pkl",
    "rb",
) as fp:
    metrics = CPUUnpickler(fp).load()
plt.plot(metrics["val_acc"], label="No Aug.")
plt.title("CIFAR-10 validation accuracy\n1,000 class-balanced points")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
savefig("/Users/harry/Documents/workspace/thesis/figures/4/eph2_da_importance.pdf")
