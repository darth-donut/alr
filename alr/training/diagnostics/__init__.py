from pathlib import Path
import pickle
from typing import Optional
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


def jitter(x):
    return x + np.random.normal(scale=0.13, size=(len(x),))


def feature_scale(arr):
    mini, maxi = arr.min(), arr.max()
    return (arr - mini) / (maxi - mini)


def confidence_plot(
        confidences_E_N, proportions_E_N, axis,
        cmap: Optional[str] = 'viridis'
):
    assert confidences_E_N.shape == proportions_E_N.shape
    cmap = cm.get_cmap(cmap)
    E = confidences_E_N.shape[0]
    for idx, (x, y) in enumerate(zip(confidences_E_N, proportions_E_N)):
        axis.plot(x, y, label=f"epoch {idx + 1}", color=cmap(idx / E))
    axis.set_title("Pseudo-label confidence on pool set")
    axis.set_xlabel("Confidence threshold")
    axis.set_ylabel("Proportion of predictions that\npass the confidence threshold")


def reliability_plot(
        bins_E_M, accuracies_E_N, counts_E_N, axis,
        cmap: Optional[str] = 'viridis'
):
    assert accuracies_E_N.shape == counts_E_N.shape
    cmap = cm.get_cmap(cmap)
    E = bins_E_M.shape[0]
    for idx, (x, y, c) in enumerate(zip(bins_E_M, accuracies_E_N, counts_E_N)):
        y[c == 0] = np.nan
        axis.scatter(
            jitter(list(range(len(x) - 1))), y,
            label=f'epoch {idx + 1}', color=cmap(idx / E),
        )
    bins = bins_E_M[0]
    axis.set_xticklabels(
        [f"({bins[idx]:.1f},{b:.1f}]" for idx, b in enumerate(bins[1:])],
        rotation=45
    )
    axis.set_xticks(range(len(bins) - 1))
    axis.set_ylim(bottom=-0.05, top=1)
    axis.set_ylabel("Accuracy of pseudo-label")
    axis.set_xlabel("Confidence")
    axis.set_title(f"Reliability plot")
    axis.set_yticks(np.arange(0, 1.1, .1))
    axis.plot(range(len(bins) - 1), np.arange(.1, 1.1, .1) - .05, color='grey', alpha=.3, linestyle='-.')


def reliability_hist_plot(bins_E_M, counts_E_N, axis, cmap: Optional[str] = 'viridis'):
    cmap = cm.get_cmap(cmap)
    E = bins_E_M.shape[0]
    for idx, (x, y) in enumerate(zip(bins_E_M, counts_E_N)):
        axis.scatter(
            jitter(list(range(len(x) - 1))), y / y.sum(),
            label=f'epoch {idx + 1}', color=cmap(idx / E),
        )
    bins = bins_E_M[0]
    axis.set_xticklabels(
        [f"({bins[idx]:.1f},{b:.1f}]" for idx, b in enumerate(bins[1:])],
        rotation=45
    )
    axis.set_xticks(range(len(bins) - 1))
    axis.set_ylim(top=1)
    axis.set_ylabel("Proportion")
    axis.set_xlabel("Confidence")
    axis.set_title(f"Confidence histogram")


# todo(harry): can accommodate iterations too
def ece_plot(ece_E, axis, cmap: Optional[str] = 'viridis'):
    cmap = cm.get_cmap(cmap)
    E = ece_E.shape[0]
    axis.plot(range(1, E + 1), ece_E)
    axis.set_title("Expected Calibration Error (ECE)")
    axis.set_ylabel("ECE")
    axis.set_xlabel("Epoch")
    axis.set_xticks(range(1, E + 1))
    axis.set_xticklabels(range(1, E + 1), rotation=45)


def plot_entropy(ent_E_N, num_classes, axis, cmap: Optional[str] = 'viridis'):
    cmap = cm.get_cmap(cmap)
    bplot = axis.boxplot(ent_E_N.T, patch_artist=True, showfliers=False)
    E = ent_E_N.shape[0]
    max_ent = num_classes * ((-1 / num_classes) * np.log(1 / num_classes))
    for e, patch in enumerate(bplot['boxes']):
        patch.set_facecolor(cmap(e / E))
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Entropy")
    axis.set_ylim(bottom=-0.05, top=max_ent)
    axis.set_yticks(np.linspace(0, max_ent, 5))
    axis.set_title("Entropy")
    axis.set_xticklabels(range(1, E + 1), rotation=45)


# todo(harry): can accommodate iterations too
def plot_accuracy(pool_acc_E, val_acc_E, axis, cmap: Optional[str] = 'viridis'):
    cmap = cm.get_cmap(cmap)
    E = pool_acc_E.shape[0]
    assert val_acc_E.shape[0] == E
    axis.plot(range(1, E + 1), pool_acc_E, label="pool")
    axis.plot(range(1, E + 1), val_acc_E, label="val")
    axis.set_title("Accuracy")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy")
    axis.set_xticks(range(1, E + 1))
    axis.set_xticklabels(range(1, E + 1), rotation=45)
    axis.legend()


def plot_sample_size(metric: dict, axis):
    y = metric['history']['train_size']
    x = len(y)
    axis.plot(range(1, x + 1), y)
    axis.set_xticks(range(1, x + 1))
    axis.set_title("Training set size")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Training set size")
    axis.set_xticklabels(range(1, x + 1), rotation=45)


def plot_val_loss(metric: dict, axis):
    y = metric['history']['val_loss']
    x = len(y)
    axis.plot(range(1, x + 1), y)
    axis.set_xticks(range(1, x + 1))
    axis.set_title("Validation Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_xticklabels(range(1, x + 1), rotation=45)


def get_val_acc(metric: dict):
    return np.array(metric['history']['val_acc'])


def plot_labelled_classes(metric: dict, axis):
    x, y = np.unique(metric['labelled_classes'], return_counts=True)
    axis.bar(x, y)
    axis.set_xlabel("Class")
    axis.set_ylabel("Counts")
    axis.set_title("BALD-acquired classes (so far)")


def diagnostics(calib_metrics: str, metrics: str):
    calib_metrics = Path(calib_metrics)
    metrics = Path(metrics)
    pkls = list(calib_metrics.rglob("*.pkl"))
    with open(metrics, "rb") as fp:
        metrics = pickle.load(fp)

    def num_sort(fname: Path):
        basename = fname.name
        return int(basename[:basename.find("_")])

    pkls = sorted(pkls, key=num_sort)
    buffer = []
    for p in pkls:
        with open(p, "rb") as fp:
            buffer.append(pickle.load(fp))

    confidences, proportions, accuracies = [], [], []
    bins, bin_accuracy, counts, ece = [], [], [], []
    entropy = []
    for b in buffer:
        res = b['conf-thresh']
        confidences.append(res[0])
        proportions.append(res[1])

        accuracies.append(b['accuracy'])

        res = b['ece']
        bins.append(res[0])
        bin_accuracy.append(res[1])
        counts.append(res[2])
        # res[3] = mean confidence
        ece.append(res[4])

        entropy.append(b['entropy'])

    confidences_E_N = np.stack(confidences, axis=0)
    proportions_E_N = np.stack(proportions, axis=0)
    accuracies_E = np.stack(accuracies, axis=0)
    bins_E_M = np.stack(bins, axis=0)
    bin_accuracy_E_N = np.stack(bin_accuracy, axis=0)
    counts_E_N = np.stack(counts, axis=0)
    ece_E = np.stack(ece, axis=0)
    entropy_E_N = np.stack(entropy, axis=0)

    fig, axes = plt.subplots(3, 3, figsize=(3 * 5, 3 * 5))
    axes = axes.flatten()
    confidence_plot(confidences_E_N, proportions_E_N, axes[0])
    ece_plot(ece_E, axes[1])
    plot_val_loss(metrics, axes[2])
    reliability_hist_plot(bins_E_M, counts_E_N, axes[3])
    plot_entropy(entropy_E_N, num_classes=10, axis=axes[4])
    plot_labelled_classes(metrics, axis=axes[5])
    reliability_plot(bins_E_M, bin_accuracy_E_N, counts_E_N, axes[6])
    plot_accuracy(accuracies_E, get_val_acc(metrics), axis=axes[7])
    plot_sample_size(metrics, axes[8])
    plt.suptitle(f"Pool size = {entropy_E_N.shape[-1]:,}", y=1.)
    for i, ax in enumerate(axes):
        if i % 3 == 0:
            ax.grid()
    fig.tight_layout()
