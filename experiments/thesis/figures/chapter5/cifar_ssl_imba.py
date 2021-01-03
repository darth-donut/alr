import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from alr.utils import savefig


def plot(dic, label, colour=None, smooth=None):
    x = sorted(list(dic.keys()))
    values = [dic[k] for k in x]
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


root = Path("/Users/harry/Documents/workspace/thesis/figures/4/data")

files = [
    # (root / "random_100_combined.pkl", "Random 100. (bal)"),
    # (root / "lastKL_20_combined.pkl", "Pre-acq KL-20 (bal)"),
    (root / "imba_last_kl_20_combined.pkl", "Pre-acq. LADI 20 (imbalanced)"),
    (root / "imba_bald_pre_20_combined.pkl", "Pre-acq. BALD 20 (imbalanced)"),
    (root / "imba_post_10_combined.pkl", "Pre-eval. BALD 10 (imbalanced)"),
    (root / "imba_kl_post_20_combined.pkl", "Pre-eval. LADI 20 (imbalanced)"),
    (root / "imba_rand_100_combined.pkl", "Random 100 (imbalanced)"),
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
plt.xlim(800, 2300)
savefig("/Users/harry/Documents/workspace/thesis/figures/4/cifar_ssl_imba.pdf")
