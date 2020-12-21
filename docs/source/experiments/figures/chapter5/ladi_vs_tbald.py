import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from alr.utils import savefig


def read_acc(metrics):
    accs = {}
    m = Path(metrics)
    for i in m.glob("*.pkl"):
        s = (int(i.name.split("_")[-1][:-4]) - 1) * 10 + 20
        with open(i, "rb") as fp:
            accs[s] = pickle.load(fp)["test_metrics"][0]
    accs = {k: accs[k] for k in sorted(accs)}
    return accs


def plot(dic, label, colour=None, smooth=None):
    keys = sorted(dic.keys())
    values = [dic[k] for k in keys]

    x = list(keys)
    y = list(map(lambda x: np.median(x), values))
    top = list(map(lambda x: np.quantile(x, 0.75), values))
    btm = list(map(lambda x: np.quantile(x, 0.25), values))
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


root = Path("/Users/harry/Documents/workspace/thesis/figures/4/data/ladi_vs_tbald/")

files = [
    (root / "lastKL_20_combined.pkl", "LADI-20"),
    (root / "tbald_20_combined.pkl", "TBALD-20"),
]

for f, n in files:
    with open(f, "rb") as fp:
        plot(pickle.load(fp), n)

plt.grid()
plt.legend()
plt.title("CIFAR-10 Test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Acquired dataset size")
plt.xlim(20, 1000)
savefig("ladi_vs_tbald.pdf")
plt.show()
