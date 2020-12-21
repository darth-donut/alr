# SVHN vs CIFAR-10 (MODEL SELECTION)
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import torch.utils.data as torchdata
import matplotlib.patches as mpatches
import colorcet

from pathlib import Path
from torch import nn
from torch.nn import functional as F
from alr.utils import savefig

import os

os.chdir("/Users/harry/Documents/workspace/thesis/reports/07_model_selection")
SAVE = True
DATA = "CIFAR"


def sort_files(files):
    return sorted(files, key=lambda x: int(str(x).split("_")[-1][:-4]))


def score_sample_plot(ax, scores, size, ylab, xlab, top=None, cifar_idx=10_000):
    if not top:
        # take everything
        top = len(scores)
    idxs = np.argsort(scores)[::-1][:top]
    cifar_mask = idxs < cifar_idx
    svhn_mask = idxs >= cifar_idx
    assert cifar_mask.sum() + svhn_mask.sum() == top
    ax.scatter(
        np.nonzero(cifar_mask)[0], scores[idxs[cifar_mask]], alpha=1, label=DATA, s=2
    )
    ax.scatter(
        np.nonzero(~cifar_mask)[0],
        scores[idxs[svhn_mask]],
        color="red",
        label="SVHN",
        s=2,
    )
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(f"Size {size}")


def itc(iteration, batch_size=400, initial=20):
    # iteration to counts
    return initial + (iteration - 1) * batch_size


def bald_range(ax, scores, size, xlabel, ylabel):
    cifar_scores = scores[:10000]
    svhn_scores = scores[10000:]
    ax.boxplot([cifar_scores, svhn_scores])
    if xlabel:
        ax.set_xticklabels([DATA, "SVHN"])
    if ylabel:
        ax.set_ylabel("BALD")
    ax.set_title(f"Size {size}")


def entropy_range(ax, entropy, etype, size, xlabel, ylabel):
    cifar_ent = entropy[:10000]
    svhn_ent = entropy[10000:]
    ax.boxplot([cifar_ent, svhn_ent])
    if xlabel:
        ax.set_xticklabels([DATA, "SVHN"])
    if ylabel:
        ax.set_ylabel(etype)
    ax.set_title(f"Size {size}")


def proportion_plot(ax, scores, etype, size, xlabel, ylabel, gt=True):
    cifar = scores[:10_000]
    svhn = scores[10_000:]
    x = np.linspace(min(scores), max(scores), num=1000)
    cy = []
    sy = []
    for i in x:
        if gt:
            cy.append((cifar >= i).mean())
            sy.append((svhn >= i).mean())
        else:
            cy.append((cifar <= i).mean())
            sy.append((svhn <= i).mean())
    ax.plot(x, cy, label=f"{DATA}-10")
    ax.plot(x, sy, label="SVHN")
    if xlabel:
        ax.set_xlabel(etype)
    if ylabel:
        ax.set_ylabel("Proportion")
    ax.set_title(f"Size {size}")


dataset = "cifar"
root = Path(f"./data_files/{dataset}_sgd")
files = list(root.glob("*_accs.pkl"))
models = [m.name.split("_")[0] for m in files]
with open(root / "subset_idxs.pkl", "rb") as fp:
    idxs = pickle.load(fp)
model = models[5]
files = sort_files((root / f"{model}_{dataset}").glob("*.pkl"))

scores_1 = []
scores_2 = []
ascores_1 = []
ascores_2 = []
predictive_entropy_1 = []
predictive_entropy_2 = []
apredictive_entropy_1 = []
apredictive_entropy_2 = []
average_entropy_1 = []
average_entropy_2 = []
aaverage_entropy_1 = []
aaverage_entropy_2 = []
confidence_1 = []
confidence_2 = []
class_1 = []
class_2 = []
map_containers = {
    "bald_score": (scores_1, scores_2),
    "bald_score2": (ascores_1, ascores_2),
    "predictive_entropy": (predictive_entropy_1, predictive_entropy_2),
    "predictive_entropy2": (apredictive_entropy_1, apredictive_entropy_2),
    "average_entropy": (average_entropy_1, average_entropy_2),
    "average_entropy2": (aaverage_entropy_1, aaverage_entropy_2),
    "confidence": (confidence_1, confidence_2),
    "class": (class_1, class_2),
}
for f in files:
    with open(f, "rb") as fp:
        trials = pickle.load(fp)
        # 'average_entropy', 'predictive_entropy', 'average_entropy2', 'predictive_entropy2',
        # 'bald_score', 'bald_score2', 'confidence', 'class'
        for k, containers in map_containers.items():
            assert len(trials) == len(containers)
            for trial, container in zip(trials, containers):
                cifar = trial[k][:10_000]
                svhn = trial[k][10_000:][idxs]
                noise = trial[k][36_032:]
                combined = np.r_[cifar, svhn, noise]
                assert combined.shape == (20_020,)
                container.append(combined)

bald_scores = np.stack([scores_1, scores_2])
proper_bald_score = np.stack([ascores_1, ascores_2])

predictive_entropy = np.stack([predictive_entropy_1, predictive_entropy_2])
proper_predictive_entropy = np.stack([apredictive_entropy_1, apredictive_entropy_2])

average_entropy = np.stack([average_entropy_1, average_entropy_2])
proper_average_entropy = np.stack([aaverage_entropy_1, aaverage_entropy_2])

confidence = np.stack([confidence_1, confidence_2])
classes = np.stack([class_1, class_2])

N = 6
interval = np.linspace(0, bald_scores.shape[1] - 1, num=N).astype(int)
fig_params = dict(nrows=2, ncols=3, figsize=(3 * 3, 2 * 3), sharex=True, sharey=True)

## Visual analysis
scores = bald_scores[0]
pent = predictive_entropy[0]
aent = average_entropy[0]
conf = confidence[0]

fig, axes = plt.subplots(**fig_params)
axes = axes.flatten()
for i, ax in enumerate(axes):
    ylab = "BALD" if i % fig_params["ncols"] == 0 else ""
    xlab = "Samples" if i >= fig_params["ncols"] else ""
    score_sample_plot(
        axes[i],
        scores[interval[i]],
        top=50,
        ylab=ylab,
        xlab=xlab,
        size=itc(interval[i] + 1),
    )
axes[i].legend()
fig.suptitle(f"Top 50 BALD scoring points ({model})")
if SAVE:
    savefig("/Users/harry/Documents/workspace/thesis/figures/4/pres_bald_sorted.pdf")

fig, axes = plt.subplots(**fig_params)
axes = axes.flatten()
for i, ax in enumerate(axes):
    bald_range(
        axes[i],
        scores[interval[i]],
        size=itc(interval[i] + 1),
        xlabel=(i >= fig_params["ncols"]),
        ylabel=(i % fig_params["ncols"] == 0),
    )
fig.suptitle(f"BALD scores")
if SAVE:
    savefig("/Users/harry/Documents/workspace/thesis/figures/4/pres_bald_dist.pdf")

fig, axes = plt.subplots(**fig_params)
axes = axes.flatten()
for i, ax in enumerate(axes):
    proportion_plot(
        axes[i],
        pent[interval[i]],
        etype="Predictive Entropy",
        size=itc(interval[i] + 1),
        xlabel=(i >= fig_params["ncols"]),
        ylabel=(i % fig_params["ncols"] == 0),
        gt=False,
    )
axes[i].legend()
fig.suptitle(f"Predictive Entropy")
if SAVE:
    savefig("/Users/harry/Documents/workspace/thesis/figures/4/pres_pred_ent.pdf")
