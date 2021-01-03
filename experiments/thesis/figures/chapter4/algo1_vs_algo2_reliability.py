import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import torch.utils.data as torchdata

from pathlib import Path
from torch import nn
from torch.nn import functional as F


root = Path("/Users/harry/Documents/workspace/thesis/figures/3/data")
with open(root / "eph_0.9_accs.pkl", "rb") as fp:
    data = pickle.load(fp)

print(np.mean(np.array(data[1000]) * 100))
print(np.std(np.array(data[1000]) * 100))

root = Path("/Users/harry/Documents/workspace/thesis/figures/3/data")
with open(root / "ssl_random_100_combined.pkl", "rb") as fp:
    data = pickle.load(fp)
print(np.mean(data[1020]) * 100)
print(np.std(np.array(data[1020]) * 100))


from alr.training.diagnostics import parse_calib_dir, reliability_hist_plot
from alr.utils import savefig


def reliability_plot(
    bins_E_M, accuracies_E_N, counts_E_N, axis, title="Reliability plot", cmap="viridis"
):
    assert accuracies_E_N.shape == counts_E_N.shape
    cmap = mpl.cm.get_cmap(cmap)
    E = bins_E_M.shape[0]
    for idx, (x, y, c) in enumerate(zip(bins_E_M, accuracies_E_N, counts_E_N)):
        y[c == 0] = np.nan
        axis.bar(
            list(range(len(x) - 1)),
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


def solo_reliability_plot(calib_metrics, title="Reliability plot"):
    (
        confidences_E_N,
        proportions_E_N,
        accuracies_E,
        bins_E_M,
        bin_accuracy_E_N,
        counts_E_N,
        ece_E,
        entropy_E_N,
        _,
    ) = parse_calib_dir(calib_metrics)

    fig = plt.figure(constrained_layout=True, figsize=(8, 8))
    spec = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2, 7])
    axes = [fig.add_subplot(spec[0, 0]), fig.add_subplot(spec[1, 0])]
    reliability_hist_plot(
        bins_E_M, counts_E_N, axes[0], xticklabels=False, title=title, bar=True
    )
    reliability_plot(bins_E_M, bin_accuracy_E_N, counts_E_N, axes[1], title=None)


root = Path(
    "/Users/harry/Documents/workspace/thesis/figures/3/data/cifar_eph_vs_eph2_calib/calib_metrics"
)
solo_reliability_plot(root / "eph", title="(Algorithm 1) Reliability plot")
savefig(
    "/Users/harry/Documents/workspace/thesis/figures/3/algo1_reliability_cifar10.pdf"
)

root = Path(
    "/Users/harry/Documents/workspace/thesis/figures/3/data/cifar_eph_vs_eph2_calib/calib_metrics"
)
solo_reliability_plot(root / "eph2", title="(Algorithm 2) Reliability plot")
savefig(
    "/Users/harry/Documents/workspace/thesis/figures/3/algo2_reliability_cifar10.pdf"
)
