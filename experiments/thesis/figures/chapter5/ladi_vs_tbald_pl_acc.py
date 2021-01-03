import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import torch.utils.data as torchdata

from pathlib import Path
from torch import nn
import seaborn as sns
from torch.nn import functional as F

from alr.data.datasets import Dataset
from alr.utils import stratified_partition, manual_seed, savefig

VAL_SIZE = 5000

root = Path("/Users/harry/Documents/workspace/thesis/figures/4/data/ladi_vs_tbald")

all_accs = []
sizes = [20, 1500]

for size in sizes:
    manual_seed(42)
    train, test = Dataset.CIFAR10.get(raw=True)
    train, pool = stratified_partition(train, 10, size)
    pool, val = torchdata.random_split(pool, (len(pool) - VAL_SIZE, VAL_SIZE))
    true_labels = []
    for _, y in pool:
        true_labels.append(y)
    true_labels = torch.from_numpy(np.array(true_labels))
    with open(root / f"datum_{size}.pkl", "rb") as fp:
        data = pickle.load(fp)
    hist = torch.from_numpy(data["label_hist"])
    klass = hist.argmax(dim=-1)
    accs = []
    for h in [-2, -3, -4, -5, -6]:
        accs.append(torch.eq(klass[-1], klass[h]).float().mean().item())
    all_accs.append(accs)

for accs, size in zip(all_accs, sizes):
    plt.plot(accs, label=r"$|\mathcal{D}_{train}| = $" + str(size), marker="o")

plt.legend()
plt.ylabel("Proportion of samples with similar pseudo-label as history[T]")
plt.xticks(range(5), [f"history[T - {i}]" for i in range(1, 6)])
plt.title(
    "Comparison of final pseudo-label (history[T])\nagainst the last five pseudo-labels in history"
)
savefig("/Users/harry/Documents/workspace/thesis/figures/4/ladi_vs_tbald_pl_acc.pdf")
