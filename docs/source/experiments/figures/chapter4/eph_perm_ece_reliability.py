import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import torch.utils.data as torchdata

from pathlib import Path
from torch import nn
from torch.nn import functional as F
from alr.training.utils import (
    _accuracy,
    _expected_calibration_error,
    _confidence_threshold,
    _entropy,
)
from alr.utils import savefig

root = Path(
    "/Users/harry/Documents/workspace/thesis/experiments/vanilla_repeated_acquisition/mnist/permanent/pl_metrics/no_al_b=10_thresh=0.9/repeat_4"
)


def jitter(x):
    return x + np.random.normal(scale=0.13, size=(len(x),))


def reliability_hist_plot(
    bins_E_M,
    counts_E_N,
    axis,
    cmap: str = "viridis",
    xticklabels=True,
    title="Confidence histogram",
    bar=False,
):
    cmap = mpl.cm.get_cmap(cmap)
    E = bins_E_M.shape[0]
    for idx, (x, y) in enumerate(zip(bins_E_M, counts_E_N)):
        if bar:
            axis.bar(
                list(range(len(x) - 1)),
                y / y.sum(),
                label=f"epoch {idx + 1}",
                color=cmap(idx / E),
            )
        else:
            axis.scatter(
                jitter(list(range(len(x) - 1))),
                y / y.sum(),
                label=f"epoch {idx + 1}",
                color=cmap(idx / E),
            )
    bins = bins_E_M[0]
    axis.set_xticklabels(
        [f"({bins[idx]:.1f},{b:.1f}]" for idx, b in enumerate(bins[1:])], rotation=45
    )
    axis.set_ylim(top=1)
    axis.set_ylabel("Proportion")
    if xticklabels:
        axis.set_xticks(range(len(bins) - 1))
        axis.set_xlabel("Confidence")
    else:
        axis.set_xticks(())
    axis.set_title(title)


def reliability_plot(
    bins_E_M,
    accuracies_E_N,
    counts_E_N,
    axis,
    title: str = "Reliability plot",
    cmap: str = "viridis",
):
    assert accuracies_E_N.shape == counts_E_N.shape
    cmap = mpl.cm.get_cmap(cmap)
    E = bins_E_M.shape[0]
    for idx, (x, y, c) in enumerate(zip(bins_E_M, accuracies_E_N, counts_E_N)):
        y[c == 0] = np.nan
        axis.scatter(
            jitter(list(range(len(x) - 1))),
            y,
            label=f"epoch {idx + 1}",
            color=cmap(idx / E),
        )
    bins = bins_E_M[0]
    axis.set_xticklabels(
        [f"({bins[idx]:.1f},{b:.1f}]" for idx, b in enumerate(bins[1:])], rotation=45
    )
    axis.set_xticks(range(len(bins) - 1))
    axis.set_ylim(bottom=-0.05, top=1)
    axis.set_ylabel("Accuracy of pseudo-label")
    axis.set_xlabel("Confidence")
    if title:
        axis.set_title(title)
    axis.set_yticks(np.arange(0, 1.1, 0.1))
    axis.plot(
        range(len(bins) - 1),
        np.arange(0.1, 1.1, 0.1) - 0.05,
        color="grey",
        alpha=0.3,
        linestyle="-.",
    )


files = sorted(root.glob("*.pkl"), key=lambda x: int(str(x.name).split("_")[0]))
confidences, proportions, accuracies = [], [], []
bins, bin_accuracy, counts, ece = [], [], [], []
entropy = []
per_acc = []
for f in files:
    with open(f, "rb") as fp:
        data = pickle.load(fp)
        preds_N_C, targets_N = np.exp(data["preds"]), data["targets"]
        ct = _confidence_threshold(preds_N_C)
        confidences.append(ct[0])
        proportions.append(ct[1])
        acc = _accuracy(preds_N_C, targets_N)
        accuracies.append(acc.mean())
        _ece = _expected_calibration_error(preds_N_C, targets_N)
        bins.append(_ece[0])
        bin_accuracy.append(_ece[1])
        counts.append(_ece[2])
        ece.append(_ece[4])

        entropy.append(_entropy(preds_N_C))
        per_acc.append(acc)

confidences = np.array(confidences)
proportions = np.array(proportions)
accuracies = np.array(accuracies)
bins = np.array(bins)
bin_accuracy = np.array(bin_accuracy)
counts = np.array(counts)
ece = np.array(ece)


fig = plt.figure(constrained_layout=True, figsize=(8, 8))
spec = fig.add_gridspec(
    ncols=2,
    nrows=2,
    width_ratios=[29, 1],
    height_ratios=[2, 7],
)
axes = [
    fig.add_subplot(spec[0, 0]),
    fig.add_subplot(spec[1, 0]),
    fig.add_subplot(spec[:, -1]),
]
reliability_hist_plot(
    bins, counts, axes[0], xticklabels=False, title="(Permanent) Reliability plot"
)
reliability_plot(bins, bin_accuracy, counts, axes[1], title=None)
norm = mpl.colors.Normalize(vmin=1, vmax=accuracies.shape[0])
fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.get_cmap("viridis")),
    orientation="vertical",
    label="Epoch",
    cax=axes[2],
)
savefig("/Users/harry/Documents/workspace/thesis/figures/3/perm_reliability_plot.pdf")
perm_ece = ece

root = Path(
    "/Users/harry/Documents/workspace/thesis/experiments/vanilla_repeated_acquisition/mnist/reconsider/pl_metrics/no_al_b=10_thresh=0.9/repeat_1"
)
files = sorted(root.glob("*.pkl"), key=lambda x: int(str(x.name).split("_")[0]))
confidences, proportions, accuracies = [], [], []
bins, bin_accuracy, counts, ece = [], [], [], []
entropy = []
per_acc = []
for f in files:
    with open(f, "rb") as fp:
        data = pickle.load(fp)
        preds_N_C, targets_N = np.exp(data["preds"]), data["targets"]
        ct = _confidence_threshold(preds_N_C)
        confidences.append(ct[0])
        proportions.append(ct[1])
        acc = _accuracy(preds_N_C, targets_N)
        accuracies.append(acc.mean())
        _ece = _expected_calibration_error(preds_N_C, targets_N)
        bins.append(_ece[0])
        bin_accuracy.append(_ece[1])
        counts.append(_ece[2])
        ece.append(_ece[4])

        entropy.append(_entropy(preds_N_C))
        per_acc.append(acc)

confidences = np.array(confidences)
proportions = np.array(proportions)
accuracies = np.array(accuracies)
bins = np.array(bins)
bin_accuracy = np.array(bin_accuracy)
counts = np.array(counts)
ece = np.array(ece)


fig = plt.figure(constrained_layout=True, figsize=(8, 8))
spec = fig.add_gridspec(
    ncols=2,
    nrows=2,
    width_ratios=[29, 1],
    height_ratios=[2, 7],
)
axes = [
    fig.add_subplot(spec[0, 0]),
    fig.add_subplot(spec[1, 0]),
    fig.add_subplot(spec[:, -1]),
]
reliability_hist_plot(
    bins, counts, axes[0], xticklabels=False, title="(Ephemeral) Reliability plot"
)
reliability_plot(bins, bin_accuracy, counts, axes[1], title=None)
norm = mpl.colors.Normalize(vmin=1, vmax=accuracies.shape[0])
fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.get_cmap("viridis")),
    orientation="vertical",
    label="Epoch",
    cax=axes[2],
)
savefig("/Users/harry/Documents/workspace/thesis/figures/3/eph_reliability_plot.pdf")


def ece_plot(ece_E, label=None):
    E = ece_E.shape[0]
    if label:
        plt.plot(range(1, E + 1), ece_E, label=label)
    else:
        plt.plot(range(1, E + 1), ece_E)
    plt.title("Expected Calibration Error (ECE)")
    plt.ylabel("ECE")
    plt.xlabel("Epoch")
    plt.xticks(range(1, E + 1), range(1, E + 1), rotation=45)


ece_plot(perm_ece, label="Permanent")
ece_plot(ece, label="Ephemeral")
plt.legend()
savefig("/Users/harry/Documents/workspace/thesis/figures/3/eph_perm_ece.pdf")
