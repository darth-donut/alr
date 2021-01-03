import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import torch.utils.data as torchdata
import colorcet

from pathlib import Path
from torch import nn
from torch.nn import functional as F
from alr.utils import savefig

root = Path(
    f"/Users/harry/Documents/workspace/thesis/reports/10_proper_imbalanced_class/data_files/bald10_imbalanced/metrics/"
)


# Plot the classes acquired by imbalanced BALD
def ridge_plot_discrete(
    distributions: np.array,
    title: str,
    xlabel: str,
    ylabel: str,
    cmap="cet_bmy_r",
    figsize=(8, 6),
    every=1,
    colour=True,
    idx_transform=lambda x: x,
):
    labels = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    # distributions is expected to be normalised
    if every != 1:
        distributions = distributions[::every]
    gs = mpl.gridspec.GridSpec(len(distributions), 1)
    fig = plt.figure(figsize=figsize)
    cmap = mpl.cm.get_cmap(cmap)

    # creating empty list
    ax_objs = []
    d_i = 0
    for i, dist in enumerate(distributions):
        # creating new axes object and appending to ax_objs
        ax_objs.append(fig.add_subplot(gs[i : i + 1, 0:]))
        # plotting the distribution
        ax_objs[-1].bar(*dist, color=cmap(i / len(distributions)), width=1.0, alpha=0.3)

        # setting uniform x and y lims
        ax_objs[-1].set_ylim(0, 1)

        rect = ax_objs[-1].patch
        rect.set_alpha(0)

        # remove borders, axis ticks, and labels
        ax_objs[-1].set_yticklabels([])
        ax_objs[-1].tick_params(axis="both", which="both", length=0)

        if i == len(distributions) - 1:
            ax_objs[-1].set_xlabel(xlabel, fontsize=13)
            ax_objs[-1].set_xticklabels(labels, rotation=45, fontsize=8)
            ax_objs[-1].set_xticks(range(10))
        else:
            ax_objs[-1].set_xticklabels([])

        spines = ["top", "right", "left", "bottom"]
        for s in spines:
            ax_objs[-1].spines[s].set_visible(False)
        ax_objs[-1].text(-1, 0, idx_transform(d_i + 1), fontsize=11, ha="right")
        d_i += every
    ax_objs[len(ax_objs) // 2].text(-1.95, 0, ylabel, fontsize=13, rotation=90)
    ax_objs[0].set_title(title)
    gs.update(hspace=-0.8)
    if colour:
        [
            i.set_color("red")
            for idx, i in enumerate(ax_objs[-1].get_xticklabels())
            if idx in {0, 4, 9}
        ]
    plt.tight_layout()


def itc(iteration):
    return 800 + (iteration - 1) * 10


seeds = [42, 24, 1008, 7, 96, 2020][:3]
for seed in seeds:
    files = [f for f in (root / f"bald_10_{seed}").glob(f"rep_1*")]
    files = sorted(files, key=lambda x: int(x.name.split("_")[-1][:-4]))
    class_dist = []
    for f in files:
        with open(f, "rb") as fp:
            metrics = pickle.load(fp)
            classes = metrics["labelled_classes"]
            if classes:
                counts = dict(zip(*np.unique(classes, return_counts=True)))
                # add in initial size
                for i in range(10):
                    if i in {0, 4, 9}:
                        if i in counts:
                            counts[i] += 10
                        else:
                            counts[i] = 10
                    else:
                        if i in counts:
                            counts[i] += 110
                        else:
                            counts[i] = 110
                target = list(counts.keys())
                count = list(counts.values())
                class_dist.append((target, count / sum(count)))
    ridge_plot_discrete(
        class_dist,
        title="Acquired class distribution over time",
        xlabel="Class",
        ylabel="Dataset size",
        every=5,
        idx_transform=itc,
    )
    savefig(
        f"/Users/harry/Documents/workspace/thesis/reports/10_proper_imbalanced_class/cumulative_class_dist_{seed}.pdf"
    )

with open(files[-1], "rb") as fp:
    metrics = pickle.load(fp)
    classes = metrics["labelled_classes"]
counts = dict(zip(*np.unique(classes, return_counts=True)))
# add in initial size
for i in range(10):
    if i in {0, 4, 9}:
        if i in counts:
            counts[i] += 10
        else:
            counts[i] = 10
    else:
        if i in counts:
            counts[i] += 110
        else:
            counts[i] = 110

with open(
    "/Users/harry/Documents/workspace/thesis/reports/10_proper_imbalanced_class/data_files/bald10_imbalanced/metrics/bald_10_24/rep_1_iter_151.pkl",
    "rb",
) as fp:
    metrics = pickle.load(fp)
    classes = metrics["labelled_classes"]
bald_count = dict(zip(*np.unique(classes, return_counts=True)))
for i in range(10):
    if i in [0, 4, 9]:
        bald_count[i] += 10
    else:
        bald_count[i] += 110
bald_count = np.array(list(bald_count.values()))
bald_count = bald_count / sum(bald_count)

original_counts = {c: 110 for c in range(10)}
for i in [0, 4, 9]:
    original_counts[i] = 10
original_counts = np.array(list(original_counts.values()))
ori_counts = original_counts / sum(original_counts)

with open(
    "/Users/harry/Documents/workspace/thesis/reports/10_proper_imbalanced_class/data_files/random50_imbalanced/metrics/random_50_24/rep_1_iter_31.pkl",
    "rb",
) as fp:
    metrics = pickle.load(fp)
    classes = metrics["labelled_classes"]
rand_count = dict(zip(*np.unique(classes, return_counts=True)))
for i in range(10):
    if i in [0, 4, 9]:
        rand_count[i] += 10
    else:
        rand_count[i] += 110
rand_count = np.array(list(rand_count.values()))
rand_count = rand_count / sum(rand_count)

# get from print_per_class_error.py
error = {
    0: 0.547,
    1: 0.094,
    2: 0.301,
    3: 0.375,
    4: 0.669,
    5: 0.205,
    6: 0.138,
    7: 0.199,
    8: 0.084,
    9: 0.315,
}

fig = plt.figure(constrained_layout=True, figsize=(8, 8))
spec = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[2, 7])
axes = [fig.add_subplot(spec[0]), fig.add_subplot(spec[1])]
axes[0].bar(list(error.keys()), list(error.values()))
axes[0].set_xticklabels(())
axes[0].set_ylabel("Per-class Test Error")
axes[0].set_title("Correlation between final class distribution and class test error")

width = 0.75
axes[1].bar(np.arange(10) - width / 3, bald_count, width=width / 3, label="BALD")
axes[1].bar(np.arange(10), rand_count, width=width / 3, label="Random")
axes[1].bar(np.arange(10) + width / 3, ori_counts, width=width / 3, label="Initial")
axes[1].legend()
labels = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
axes[1].set_xticks(range(10))
axes[1].set_ylabel("Per-class Proportion in final dataset (2,300 points)")
axes[1].set_xticklabels(labels, rotation=45)
[
    i.set_color("red")
    for idx, i in enumerate(axes[1].get_xticklabels())
    if idx in {0, 4, 9}
]
savefig("/Users/harry/Documents/workspace/thesis/figures/4/first_last_class_dist.pdf")
