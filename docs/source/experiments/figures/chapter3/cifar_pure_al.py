import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import torch.utils.data as torchdata

from pathlib import Path
from torch import nn
from torch.nn import functional as F
from collections import defaultdict
from alr.utils import savefig

root = Path("/Users/harry/Documents/workspace/thesis/figures/4/data")

def plot(dic, label, colour=None, smooth=None, alpha=False):
    x = list(dic.keys())
    y = list(map(lambda x: np.median(x), dic.values()))
    top = list(map(lambda x: np.quantile(x, .75), dic.values()))
    btm = list(map(lambda x: np.quantile(x, .25), dic.values()))
    if smooth and len(x) > 15:
        x = x[::smooth]
        y = y[::smooth]
        top = top[::smooth]
        btm = btm[::smooth]
    if colour is None:
        if alpha:
            l, = plt.plot(x, y, label=label, alpha=alpha, linestyle='dashed')
        else:
            l, = plt.plot(x, y, label=label)
        plt.fill_between(x, top, btm, color=l.get_color(), alpha=.2)
    else:
        plt.plot(x, y, label=label, color=colour)
        plt.fill_between(x, top, btm, color=colour, alpha=.2)


with open(root / "im_random_50_combined.pkl", "rb") as fp:
    im_rand = pickle.load(fp)
with open(root / "bal_random_50_combined.pkl", "rb") as fp:
    bal_rand = pickle.load(fp)
with open(root / "im_bald_10_combined.pkl", "rb") as fp:
    im_bald = pickle.load(fp)


plot(im_bald, label="BALD 10 (imbalanced)")
plot(im_rand, label="Random 50 (imbalanced)")
plot(bal_rand, label="Random 50 (balanced)", alpha=1)

handles, labels = plt.gca().get_legend_handles_labels()
handles = handles[-1:] + handles[:-1]
labels = labels[-1:] + labels[:-1]

plt.legend(handles, labels)
plt.grid()
# plt.legend()
plt.xlim(800, 2000)
plt.xlabel("Acquired dataset size")
plt.ylabel("Accuracy")
plt.title("CIFAR-10 test accuracy")
savefig("/Users/harry/Documents/workspace/thesis/figures/4/cifar_pure_al.pdf")

