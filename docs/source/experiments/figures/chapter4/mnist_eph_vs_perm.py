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



def plot(dic, label, dot=False):
    x = list(dic.keys())
    y = np.array(list(dic.values()))
    median = np.median(y, axis=-1)
    top = np.percentile(y, 75, axis=-1)
    btm = np.percentile(y, 25, axis=-1)
    if dot:
        l, = plt.plot(x, median, label=label, linestyle='dashdot')
    else:
        l, = plt.plot(x, median, label=label)
    plt.fill_between(x, btm, top, color=l.get_color(), alpha=.2)



root = Path("/Users/harry/Documents/workspace/thesis/experiments/vanilla_repeated_acquisition/mnist/permanent")
with open(root / "no_al_b=10_thresh=0.9_accs.pkl", "rb") as fp:
    data = pickle.load(fp)

perm = defaultdict(list)

for trial in data:
    for e, v in enumerate(trial.values(), 1):
        perm[e].append(v)

root = Path("/Users/harry/Documents/workspace/thesis/experiments/vanilla_repeated_acquisition/mnist/reconsider")
with open(root / "no_al_b=10_thresh=0.9_accs.pkl", "rb") as fp:
    data = pickle.load(fp)

eph = defaultdict(list)
for trial in data:
    for e, v in enumerate(trial.values(), 1):
        eph[e].append(v)

plot(perm, "Permanent")
plot(eph, "Ephemeral")
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("MNIST test accuracy")
plt.xlim(0, 24)
plt.legend()
savefig("/Users/harry/Documents/workspace/thesis/figures/3/mnist_eph_vs_perm.pdf")


