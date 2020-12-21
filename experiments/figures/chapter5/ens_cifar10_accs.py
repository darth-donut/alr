import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from alr.utils import savefig


def plot(dic, label, colour=None, smooth=None):
    x = list(dic.keys())
    y = list(map(lambda x: np.median(x), dic.values()))
    top = list(map(lambda x: np.quantile(x, 0.75), dic.values()))
    btm = list(map(lambda x: np.quantile(x, 0.25), dic.values()))
    if smooth and len(x) > 15:
        x = x[::smooth]
        y = y[::smooth]
        top = top[::smooth]
        btm = btm[::smooth]
    if colour is None:
        (l,) = plt.plot(x, y, label=label)
        plt.fill_between(x, top, btm, color=l.get_color(), alpha=0.2)
    else:
        plt.plot(x, y, label=label, color=colour)
        plt.fill_between(x, top, btm, color=colour, alpha=0.2)


root = Path("/Users/harry/Documents/workspace/thesis/figures/4/data")

files = [
    (root / "ens_random_50_combined.pkl", "Random 50"),
    (root / "ens_bald_10_combined.pkl", "BALD 10"),
]
for f, n in files:
    with open(f, "rb") as fp:
        # plot(pickle.load(fp), n, smooth=3)
        plot(pickle.load(fp), n)

plt.grid()
plt.legend()
plt.title("Test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Acquired dataset size")
plt.xlim(1000, 2000)
plt.ylim(0.64)
savefig("/Users/harry/Documents/workspace/thesis/figures/4/ens_cifar10_accs.pdf")
