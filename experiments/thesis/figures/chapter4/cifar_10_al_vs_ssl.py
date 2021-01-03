# TODO: accuracy plot - replot figure 5 from 02_SSAL
# TODO: reliability plot - /users/ms19jhf/Documents/workspace/experiments/vanilla_repeated_acquisition/mnist/permanent/pl_metrics/no_al_b=10_thresh=0.9/repeat_1
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import torch.utils.data as torchdata

from pathlib import Path
from torch import nn
from torch.nn import functional as F

from alr.utils import savefig

root = Path("/Users/harry/Documents/workspace/thesis/figures/3/data")
with open(root / "bald_10_combined_ens.pkl", "rb") as fp:
    bald10 = pickle.load(fp)
with open(root / "random_100_combined.pkl", "rb") as fp:
    rand100 = pickle.load(fp)


def plot(dic, label, dot=False):
    x = list(dic.keys())
    y = np.array(list(dic.values()))
    median = np.median(y, axis=-1)
    top = np.percentile(y, 75, axis=-1)
    btm = np.percentile(y, 25, axis=-1)
    if dot:
        (l,) = plt.plot(x, median, label=label, linestyle="dashdot")
    else:
        (l,) = plt.plot(x, median, label=label)
    plt.fill_between(x, btm, top, color=l.get_color(), alpha=0.2)
    # plt.xticks([20] + list(range(50, 300, 50)))
    # plt.xlim(left=20, right=250)


plot(rand100, "SSL (Random-100)", dot=True)
plot(bald10, "AL (BALD-10)")
plt.xlim(20, 2000)
plt.grid()
plt.title("CIFAR-10 test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Dataset size")
plt.axhline(y=0.94, color="k")

line = mpl.lines.Line2D([0], [0], color="k")
handles, labels = plt.gca().get_legend_handles_labels()
handles.insert(0, line)
labels.insert(0, "Accuracy on full dataset")
plt.legend(handles=handles, labels=labels)
savefig("/Users/harry/Documents/workspace/thesis/figures/3/cifar_10_al_vs_ssl.pdf")
