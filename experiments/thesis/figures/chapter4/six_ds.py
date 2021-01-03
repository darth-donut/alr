import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from typing import Optional
from matplotlib import cm
import numpy as np
from alr.utils import savefig

root = Path(
    "/Users/harry/Documents/workspace/thesis/experiments/vanilla_repeated_acquisition/mnist/reconsider/more_iters_diff_dataset"
)


def plot(dic, label):
    x = list(dic.keys())
    y = np.array(list(dic.values()))
    median = np.median(y, axis=-1)
    top = np.percentile(y, 75, axis=-1)
    btm = np.percentile(y, 25, axis=-1)
    (l,) = plt.plot(x, median, label=label)
    plt.fill_between(x, btm, top, color=l.get_color(), alpha=0.2)


with open(root / "no_al_b=10_thresh=0.9_accs.pkl", "rb") as fp:
    data = pickle.load(fp)

accs = defaultdict(list)
for trial in data:
    for e, v in enumerate(trial, 1):
        accs[e].append(v)

plot(accs, None)
plt.grid()
plt.xlim(0, 55)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MNIST test accuracy with six\ndifferent sets of 20 class-balanced points")
savefig("/Users/harry/Documents/workspace/thesis/figures/3/six_ds.pdf")
