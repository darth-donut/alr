# plot sslb and bssl and SSL + random
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from typing import Optional
from matplotlib import cm
import numpy as np
from alr.utils import savefig

import os
os.chdir("/Users/harry/Documents/workspace/thesis/reports/03_MNIST_report")

root = Path("/Users/harry/Documents/workspace/thesis/experiments/ephemeral/mnist")
with open(root / "SSL_BALD" / "thresh_0.9_b_10_accs.pkl", "rb") as fp:
    spl_m = pickle.load(fp)
with open(root / "BALD_SSL" / "thresh_0.9_b_10_accs.pkl", "rb") as fp:
    al_ssl_m = pickle.load(fp)
baseline = Path("/Users/harry/Documents/workspace/thesis/experiments/al_baselines/mnist/bald1/bald_1_accs.pkl")
with open(baseline, "rb") as fp:
    bald = pickle.load(fp)
with open(root / "SSL_RA" / "thresh_0.9_b_10_accs.pkl", "rb") as fp:
    spl_m_ra = pickle.load(fp)
with open("/Users/harry/Documents/workspace/thesis/experiments/al_baselines/mnist/random10/random_10_accs.pkl", "rb") as fp:
    rand = pickle.load(fp)

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
    plt.xticks([20] + list(range(50, 300, 50)))
    plt.xlim(left=20, right=250)


def plot2(dic, label):
    x = list(dic.keys())
    y = list(dic.values())
    median = list(map(np.median, y))
    top = list(map(lambda x: np.percentile(x, 75), y))
    btm = list(map(lambda x: np.percentile(x, 25), y))
    l, = plt.plot(x, median, label=label)
    plt.fill_between(x, btm, top, color=l.get_color(), alpha=.2)
    plt.xticks([20] + list(range(50, 300, 50)))
    plt.xlim(left=20, right=250)

def plot3(arr, axis, label, colour=None, alpha=0.2):
    x = range(arr.shape[1])
    y = arr
    median = np.median(y, axis=0)
    topq = np.quantile(y, .75, axis=0)
    btmq = np.quantile(y, .25, axis=0)

    if colour:
        line, = axis.plot(x, median, color=colour, label=label)
    else:
        line, = axis.plot(x, median, label=label)
    axis.fill_between(
        x, btmq, topq,
        color=line.get_color(), alpha=alpha,
    )
    # axis.set_xticks([20] + list(range(50, 300, 50)))
    # axis.set_xlim(left=20, right=250)

plot(al_ssl_m, "Pre-evaluation (BALD-10)", dot=True)
plot(spl_m, "Pre-acquisition (BALD-10)", dot=True)
plot(spl_m_ra, "SSL (Random-10)", dot=True)
plot(bald, "AL (BALD-1)")
# plot(rand, "Random-10")
plt.hlines(0.99, xmin=20, xmax=250, colors='k')
# plt.text(97, .995, "Accuracy using full dataset", fontsize=8, color='k', fontweight='bold')
# plt.text(3, .980, "0.985", fontsize=8, color='grey')
plt.title("MNIST test accuracy")
plt.xlabel("Acquired dataset size")
plt.ylabel("Accuracy")
plt.xlim(left=20, right=250)

line = mpl.lines.Line2D([0], [0], color='k')
handles, labels = plt.gca().get_legend_handles_labels()
handles.insert(0, line); labels.insert(0, 'Accuracy on full dataset')
plt.legend(handles=handles, labels=labels)
plt.grid()

savefig("/Users/harry/Documents/workspace/thesis/figures/3/mnist_full_acc.pdf")

