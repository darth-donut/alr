import pickle
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from alr.data import UnlabelledDataset
from alr.data.datasets import Dataset
import torch.utils.data as torchdata
import matplotlib.gridspec as grid_spec
from sklearn.neighbors import KernelDensity
from matplotlib import cm
import torchvision as tv
import tqdm
import colorcet as cc
from scipy.stats import pearsonr

from alr.utils import savefig


bald = Path(
    "/Users/harry/Documents/workspace/thesis/figures/4/data/ens_bald_metrics/bald_10_96"
)

rep = 1
files = [f for f in bald.glob(f"rep_{rep}*")]
files = sorted(files, key=lambda x: int(x.name.split("_")[-1][:-4]))
bald_scores = []
for f in files:
    with open(f, "rb") as fp:
        bald_metrics = pickle.load(fp)
        if bald_metrics["bald_scores"]:
            bald_scores.append(bald_metrics["bald_scores"][-1])

# https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/
def ridge_plot(
    distributions: list,
    title: str,
    xlabel: str,
    ylabel: str,
    cmap="cet_bmy_r",
    figsize=(8, 6),
    every=1,
    transform=lambda x: x,
):
    if every != 1:
        distributions = distributions[::every]
    x_max = max(map(max, distributions))
    gs = grid_spec.GridSpec(len(distributions), 1)
    fig = plt.figure(figsize=figsize)
    cmap = cm.get_cmap(cmap)

    # creating empty list
    ax_objs = []

    d_i = 0

    for i, dist in tqdm.tqdm(enumerate(distributions)):
        # creating new axes object and appending to ax_objs
        ax_objs.append(fig.add_subplot(gs[i : i + 1, 0:]))

        x_d = np.linspace(0, x_max, 1000)

        kde = KernelDensity(bandwidth=0.03, kernel="gaussian")
        kde.fit(dist[:, None])

        logprob = kde.score_samples(x_d[:, None])

        # plotting the distribution
        ax_objs[-1].plot(x_d, np.exp(logprob), lw=1, color=cmap(i / len(distributions)))
        ax_objs[-1].fill_between(
            x_d, np.exp(logprob), color=cmap(i / len(distributions)), alpha=0.3
        )

        # setting uniform x and y lims
        ax_objs[-1].set_xlim(0, x_max)
        # ax_objs[-1].set_ylim(0,2.2)

        rect = ax_objs[-1].patch
        rect.set_alpha(0)

        # remove borders, axis ticks, and labels
        ax_objs[-1].set_yticklabels([])
        ax_objs[-1].tick_params(axis="both", which="both", length=0)

        if i == len(distributions) - 1:
            ax_objs[-1].set_xlabel(xlabel, fontsize=13)
        else:
            ax_objs[-1].set_xticklabels([])

        spines = ["top", "right", "left", "bottom"]
        for s in spines:
            ax_objs[-1].spines[s].set_visible(False)
        ax_objs[-1].text(-0.02, 0, transform(d_i + 1), fontsize=11, ha="right")
        d_i += every
    ax_objs[len(ax_objs) // 2].text(-0.17, 0, ylabel, fontsize=13, rotation=90)
    ax_objs[0].set_title(title)
    gs.update(hspace=-0.7)
    plt.tight_layout()


ridge_plot(
    bald_scores,
    title="BALD score density",
    xlabel="BALD score",
    ylabel="Acquired dataset size",
    every=15,
    transform=lambda x: 20 + (x - 1) * 10,
)
savefig("/Users/harry/Documents/workspace/thesis/figures/4/BALD_density.pdf")
