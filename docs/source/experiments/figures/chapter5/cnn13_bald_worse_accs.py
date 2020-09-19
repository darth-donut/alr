import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from alr.utils import savefig


def plot(dic, label, colour=None, smooth=None):
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
        l, = plt.plot(x, y, label=label)
        plt.fill_between(x, top, btm, color=l.get_color(), alpha=.2)
    else:
        plt.plot(x, y, label=label, color=colour)
        plt.fill_between(x, top, btm, color=colour, alpha=.2)

root = Path("/Users/harry/Documents/workspace/thesis/figures/4/data")

files = [
    (root / "cnn13_rand_10_accs.pkl", "Random-10"),
    (root / "cnn13_bald_10_accs.pkl", "BALD-10"),
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
plt.xlim(20, 2000)
savefig("/Users/harry/Documents/workspace/thesis/figures/4/cnn13_bald_worse_accs.pdf")


