from pathlib import Path
import pickle
from typing import Optional
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


def jitter(x):
    return x + np.random.normal(scale=0.13, size=(len(x),))


def feature_scale(arr):
    mini, maxi = arr.min(), arr.max()
    return (arr - mini) / (maxi - mini)


def confidence_plot(
    confidences_E_N, proportions_E_N, axis, cmap: Optional[str] = "viridis"
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
    bins_E_M,
    accuracies_E_N,
    counts_E_N,
    axis,
    title: Optional[str] = "Reliability plot",
    cmap: Optional[str] = "viridis",
):
    assert accuracies_E_N.shape == counts_E_N.shape
    cmap = cm.get_cmap(cmap)
    E = bins_E_M.shape[0]
    for idx, (x, y, c) in enumerate(zip(bins_E_M, accuracies_E_N, counts_E_N)):
        y[c == 0] = np.nan
        axis.scatter(
            jitter(list(range(len(x) - 1))),
            y,
            label=f"epoch {idx + 1}",
            color=cmap(idx / E),
        )
    bins = bins_E_M[0]
    axis.set_xticklabels(
        [f"({bins[idx]:.1f},{b:.1f}]" for idx, b in enumerate(bins[1:])], rotation=45
    )
    axis.set_xticks(range(len(bins) - 1))
    axis.set_ylim(bottom=-0.05, top=1)
    axis.set_ylabel("Accuracy of pseudo-label")
    axis.set_xlabel("Confidence")
    if title:
        axis.set_title(title)
    axis.set_yticks(np.arange(0, 1.1, 0.1))
    axis.plot(
        range(len(bins) - 1),
        np.arange(0.1, 1.1, 0.1) - 0.05,
        color="grey",
        alpha=0.3,
        linestyle="-.",
    )


def reliability_hist_plot(
    bins_E_M,
    counts_E_N,
    axis,
    cmap: Optional[str] = "viridis",
    xticklabels=True,
    title="Confidence histogram",
    bar=False,
):
    cmap = cm.get_cmap(cmap)
    E = bins_E_M.shape[0]
    for idx, (x, y) in enumerate(zip(bins_E_M, counts_E_N)):
        if bar:
            axis.bar(
                list(range(len(x) - 1)),
                y / y.sum(),
                label=f"epoch {idx + 1}",
                color=cmap(idx / E),
            )
        else:
            axis.scatter(
                jitter(list(range(len(x) - 1))),
                y / y.sum(),
                label=f"epoch {idx + 1}",
                color=cmap(idx / E),
            )
    bins = bins_E_M[0]
    axis.set_xticklabels(
        [f"({bins[idx]:.1f},{b:.1f}]" for idx, b in enumerate(bins[1:])], rotation=45
    )
    axis.set_ylim(top=1)
    axis.set_ylabel("Proportion")
    if xticklabels:
        axis.set_xticks(range(len(bins) - 1))
        axis.set_xlabel("Confidence")
    else:
        axis.set_xticks(())
    axis.set_title(title)


# todo(harry): can accommodate iterations too
def ece_plot(ece_E, axis, label: Optional[str] = None, cmap: Optional[str] = "viridis"):
    cmap = cm.get_cmap(cmap)
    E = ece_E.shape[0]
    if label:
        axis.plot(range(1, E + 1), ece_E, label=label)
    else:
        axis.plot(range(1, E + 1), ece_E)
    axis.set_title("Expected Calibration Error (ECE)")
    axis.set_ylabel("ECE")
    axis.set_xlabel("Epoch")
    axis.set_xticks(range(1, E + 1))
    axis.set_xticklabels(range(1, E + 1), rotation=45)


def plot_entropy(ent_E_N, num_classes, axis, cmap: Optional[str] = "viridis"):
    cmap = cm.get_cmap(cmap)
    bplot = axis.boxplot(ent_E_N.T, patch_artist=True, showfliers=False)
    E = ent_E_N.shape[0]
    max_ent = num_classes * ((-1 / num_classes) * np.log(1 / num_classes))
    for e, patch in enumerate(bplot["boxes"]):
        patch.set_facecolor(cmap(e / E))
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Entropy")
    axis.set_ylim(bottom=-0.05, top=max_ent)
    axis.set_yticks(np.linspace(0, max_ent, 5))
    axis.set_title("Entropy")
    axis.set_xticklabels(range(1, E + 1), rotation=45)


# todo(harry): can accommodate iterations too
def plot_accuracy(pool_acc_E, val_acc_E, axis, cmap: Optional[str] = "viridis"):
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
    y = metric["history"]["train_size"]
    x = len(y)
    axis.plot(range(1, x + 1), y)
    axis.set_xticks(range(1, x + 1))
    axis.set_title("Training set size")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Training set size")
    axis.set_xticklabels(range(1, x + 1), rotation=45)


def plot_val_loss(metric: dict, axis):
    y = metric["history"]["val_loss"]
    x = len(y)
    axis.plot(range(1, x + 1), y)
    axis.set_xticks(range(1, x + 1))
    axis.set_title("Validation Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_xticklabels(range(1, x + 1), rotation=45)


def get_val_acc(metric: dict):
    return np.array(metric["history"]["val_acc"])


def plot_labelled_classes(metric: dict, axis):
    x, y = np.unique(metric["labelled_classes"], return_counts=True)
    axis.bar(x, y)
    axis.set_xlabel("Class")
    axis.set_ylabel("Counts")
    axis.set_title("BALD-acquired classes (so far)")


def parse_calib_dir(calib_metrics: str):
    def num_sort(fname: Path):
        basename = fname.name
        return int(basename[: basename.find("_")])

    calib_metrics = Path(calib_metrics)
    pkls = list(calib_metrics.rglob("*.pkl"))
    pkls = sorted(pkls, key=num_sort)
    buffer = []
    for p in pkls:
        with open(p, "rb") as fp:
            buffer.append(pickle.load(fp))

    confidences, proportions, accuracies = [], [], []
    bins, bin_accuracy, counts, ece = [], [], [], []
    entropy = []
    per_acc = []
    for b in buffer:
        res = b["conf-thresh"]
        confidences.append(res[0])
        proportions.append(res[1])

        accuracies.append(b["accuracy"])

        res = b["ece"]
        bins.append(res[0])
        bin_accuracy.append(res[1])
        counts.append(res[2])
        # res[3] = mean confidence
        ece.append(res[4])

        entropy.append(b["entropy"])

        if "per-instance-accuracy" in b:
            per_acc.append(b["per-instance-accuracy"])

    confidences_E_N = np.stack(confidences, axis=0)
    proportions_E_N = np.stack(proportions, axis=0)
    accuracies_E = np.stack(accuracies, axis=0)
    bins_E_M = np.stack(bins, axis=0)
    bin_accuracy_E_N = np.stack(bin_accuracy, axis=0)
    counts_E_N = np.stack(counts, axis=0)
    ece_E = np.stack(ece, axis=0)
    try:
        # can only do so if entropy is a non-jagged matrix (non-pool set calib)
        entropy_E_N = np.stack(entropy, axis=0)
        if per_acc:
            per_acc_E_N = np.stack(per_acc, axis=0)
        else:
            per_acc_E_N = None
    except:
        entropy_E_N = None
        per_acc_E_N = None
    return (
        confidences_E_N,
        proportions_E_N,
        accuracies_E,
        bins_E_M,
        bin_accuracy_E_N,
        counts_E_N,
        ece_E,
        entropy_E_N,
        per_acc_E_N,
    )


def diagnostics(calib_metrics: str, metrics: str):
    metrics = Path(metrics)
    (
        confidences_E_N,
        proportions_E_N,
        accuracies_E,
        bins_E_M,
        bin_accuracy_E_N,
        counts_E_N,
        ece_E,
        entropy_E_N,
        _,
    ) = parse_calib_dir(calib_metrics)
    with open(metrics, "rb") as fp:
        metrics = pickle.load(fp)

    fig, axes = plt.subplots(3, 3, figsize=(3 * 5, 3 * 5))
    axes = axes.flatten()
    confidence_plot(confidences_E_N, proportions_E_N, axes[0])
    ece_plot(ece_E, axes[1])
    plot_val_loss(metrics, axes[2])
    reliability_hist_plot(bins_E_M, counts_E_N, axes[3])
    if entropy_E_N is not None:
        plot_entropy(entropy_E_N, num_classes=10, axis=axes[4])
    plot_labelled_classes(metrics, axis=axes[5])
    reliability_plot(bins_E_M, bin_accuracy_E_N, counts_E_N, axes[6])
    plot_accuracy(accuracies_E, get_val_acc(metrics), axis=axes[7])
    plot_sample_size(metrics, axes[8])
    plt.suptitle(f"Pool size = {entropy_E_N.shape[-1]:,}", y=1.0)
    for i, ax in enumerate(axes):
        if i % 3 == 0:
            ax.grid()
    fig.tight_layout()


def solo_reliability_plot(calib_metrics, title="Reliability plot", label="Iteration"):
    (
        confidences_E_N,
        proportions_E_N,
        accuracies_E,
        bins_E_M,
        bin_accuracy_E_N,
        counts_E_N,
        ece_E,
        entropy_E_N,
        _,
    ) = parse_calib_dir(calib_metrics)

    fig = plt.figure(constrained_layout=True, figsize=(8, 8))
    spec = fig.add_gridspec(
        ncols=2,
        nrows=2,
        width_ratios=[29, 1],
        height_ratios=[2, 7],
    )
    axes = [
        fig.add_subplot(spec[0, 0]),
        fig.add_subplot(spec[1, 0]),
        fig.add_subplot(spec[:, -1]),
    ]
    reliability_hist_plot(bins_E_M, counts_E_N, axes[0], xticklabels=False, title=title)
    reliability_plot(bins_E_M, bin_accuracy_E_N, counts_E_N, axes[1], title=None)
    norm = mpl.colors.Normalize(vmin=1, vmax=accuracies_E.shape[0])
    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cm.get_cmap("viridis")),
        orientation="vertical",
        label=label,
        cax=axes[2],
    )


def entropy_reliability_plot(calib_metrics, num_class=10):
    *_, entropy_E_N, per_acc_E_N = parse_calib_dir(calib_metrics)
    E = entropy_E_N.shape[0]

    max_ent = -np.log(1 / num_class)
    space = np.linspace(0, max_ent, 11)

    fig = plt.figure(constrained_layout=True, figsize=(8, 8))
    if E > 1:
        spec = fig.add_gridspec(
            ncols=2,
            nrows=2,
            width_ratios=[29, 1],
            height_ratios=[2, 7],
        )
        axes = [
            fig.add_subplot(spec[0, 0]),
            fig.add_subplot(spec[1, 0]),
            fig.add_subplot(spec[:, -1]),
        ]
    else:
        spec = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[2, 7])
        axes = [fig.add_subplot(spec[0, 0]), fig.add_subplot(spec[1, 0])]

    for ent, acc in zip(entropy_E_N, per_acc_E_N):
        y = []
        x = []
        p = []
        for i, upper in enumerate(space[1:]):
            lower = space[i]
            mask = (ent > lower) & (ent <= upper)
            mean_acc = acc[mask].mean()
            prop = mask.mean()
            y.append(mean_acc)
            # (lower, upper]
            x.append(f"({lower:.2f}, {upper:.2f}]")
            p.append(prop)
        if E == 1:
            axes[1].bar(range(len(y)), y)
            axes[0].bar(range(len(p)), p)
        else:
            raise NotImplementedError
        axes[1].set_xticklabels(x, rotation=45, ha="right")
        axes[1].set_xticks(range(len(y)))
    axes[0].set_xticks(())
    axes[0].set_xticklabels(())
    axes[0].set_title("Reliability plot")
    axes[0].set_ylabel("Proportion")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xlabel("Entropy")

    # norm = mpl.colors.Normalize(vmin=1, vmax=accuracies_E.shape[0])
    # fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('viridis')),
    #              orientation='vertical', label=label, cax=axes[2])
