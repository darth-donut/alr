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

from alr.data.datasets import Dataset
import os

os.chdir("/Users/harry/Documents/workspace/thesis/reports/09_imbalanced_classes")
DATA = "CIFAR"
include_noise = False
SAVE = False
minority_classes = {0, 4, 9}
_, test = Dataset.CIFAR10.get()
minority_idxs = []
majority_idxs = []
for idx, (_, y) in enumerate(test):
    if y in minority_classes:
        minority_idxs.append(idx)
    else:
        majority_idxs.append(idx)


def sort_files(files):
    return sorted(files, key=lambda x: int(str(x).split("_")[-1][:-4]))


def score_sample_plot(
    ax, scores, size, ylab, xlab, top=None, cifar_idx=10_000, no_svhn=False
):
    if no_svhn:
        scores = scores[:10_000]
    if not top:
        # take everything
        top = len(scores)
    idxs = np.argsort(scores)[::-1][:top]
    cifar_minority_mask = np.isin(idxs, minority_idxs)
    cifar_majority_mask = np.isin(idxs, majority_idxs)
    if not no_svhn:
        svhn_mask = idxs >= cifar_idx
        svhn_counts = svhn_mask.sum()
    else:
        svhn_counts = 0
    minority_counts = cifar_minority_mask.sum()
    majority_counts = cifar_majority_mask.sum()
    assert svhn_counts + minority_counts + majority_counts == top
    ax.scatter(
        np.nonzero(cifar_majority_mask)[0],
        scores[idxs[cifar_majority_mask]],
        alpha=1,
        label="majority",
        s=2,
    )
    if not no_svhn:
        ax.scatter(
            np.nonzero(svhn_mask)[0],
            scores[idxs[svhn_mask]],
            color="red",
            label="SVHN",
            s=2,
        )
    ax.scatter(
        np.nonzero(cifar_minority_mask)[0],
        scores[idxs[cifar_minority_mask]],
        color="orange",
        label="minority",
        s=2,
    )
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(
        f"Size {size} {svhn_counts/top:.2%} SVHN;\n{minority_counts/top:.2%}Minority; {majority_counts/top:.2%} Majority",
        fontsize=8,
    )


def itc(iteration, batch_size=400, initial=800):
    # iteration to counts
    return initial + (iteration - 1) * batch_size


def bald_range(ax, scores, size, xlabel, ylabel):
    cifar_scores = scores[:10000]
    svhn_scores = scores[10000:]
    minority_scores = cifar_scores[minority_idxs]
    majority_scores = cifar_scores[majority_idxs]
    ax.boxplot([svhn_scores, minority_scores, majority_scores])
    if xlabel:
        ax.set_xticklabels(["SVHN", "Minority", "Majority"])
    if ylabel:
        ax.set_ylabel("BALD")
    ax.set_title(f"Size {size}")


def entropy_range(ax, entropy, etype, size, xlabel, ylabel):
    cifar_ent = entropy[:10000]
    svhn_ent = entropy[10000:]
    minority_scores = cifar_ent[minority_idxs]
    majority_scores = cifar_ent[majority_idxs]
    ax.boxplot([svhn_ent, minority_scores, majority_scores])
    if xlabel:
        ax.set_xticklabels(["SVHN", "Minority", "Majority"])
    if ylabel:
        ax.set_ylabel(etype)
    ax.set_title(f"Size {size}")


def proportion_plot(ax, scores, etype, size, xlabel, ylabel, gt=True):
    cifar = scores[:10_000]
    svhn = scores[10_000:]
    x = np.linspace(min(scores), max(scores), num=1000)
    miy = []
    may = []
    sy = []
    for i in x:
        if gt:
            miy.append((cifar[minority_idxs] >= i).mean())
            may.append((cifar[majority_idxs] >= i).mean())
            sy.append((svhn >= i).mean())
        else:
            miy.append((cifar[minority_idxs] <= i).mean())
            may.append((cifar[majority_idxs] <= i).mean())
            sy.append((svhn <= i).mean())
    ax.plot(x, may, label=f"Majority")
    ax.plot(x, miy, label="Minority", color="orange")
    ax.plot(x, sy, label="SVHN", color="red")
    if xlabel:
        ax.set_xlabel(etype)
    if ylabel:
        ax.set_ylabel("Proportion")
    ax.set_title(f"Size {size}")


dataset = "cifar"
root = Path(f"./data_files/cnn13_ens")
files = list(root.glob("*_accs.pkl"))
models = [m.name.split("_")[0] for m in files]
with open(root / "subset_idxs.pkl", "rb") as fp:
    idxs = pickle.load(fp)
model = models[0]
files = sort_files(root.glob("rep_1*.pkl"))

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
    "bald_score": (scores_1,),
    "bald_score2": (ascores_1,),
    "predictive_entropy": (predictive_entropy_1,),
    "predictive_entropy2": (apredictive_entropy_1,),
    "average_entropy": (average_entropy_1,),
    "average_entropy2": (aaverage_entropy_1,),
    "confidence": (confidence_1,),
    "class": (class_1,),
}
for f in files:
    with open(f, "rb") as fp:
        trials = pickle.load(fp)
        # 'average_entropy', 'predictive_entropy', 'average_entropy2', 'predictive_entropy2',
        # 'bald_score', 'bald_score2', 'confidence', 'class'
        for k, containers in map_containers.items():
            # assert len(trials) == len(containers)
            for trial, container in zip([trials], containers):
                cifar = trial[k][:10_000]
                svhn = trial[k][10_000:][idxs]
                noise = trial[k][36_032:]
                if include_noise:
                    combined = np.r_[cifar, svhn, noise]
                    assert combined.shape == (20_020,)
                else:
                    combined = np.r_[cifar, svhn]
                    assert combined.shape == (20_000,)
                container.append(combined)


bald_scores = np.stack(scores_1)
proper_bald_score = np.stack(ascores_1)

predictive_entropy = np.stack(predictive_entropy_1)
proper_predictive_entropy = np.stack(apredictive_entropy_1)

average_entropy = np.stack(average_entropy_1)
proper_average_entropy = np.stack(aaverage_entropy_1)

confidence = np.stack(confidence_1)
classes = np.stack(class_1)


N = 6
interval = np.linspace(0, bald_scores.shape[0] - 1, num=N).astype(int)
fig_params = dict(nrows=2, ncols=3, figsize=(3 * 3, 2 * 3), sharex=True, sharey=True)

## Visual analysis
scores = bald_scores
pent = predictive_entropy
aent = average_entropy
conf = confidence
TOP = 50

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
savefig("/Users/harry/Documents/workspace/thesis/figures/4/rough_bald_dist.pdf")

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
savefig("/Users/harry/Documents/workspace/thesis/figures/4/rough_pred_ent.pdf")
