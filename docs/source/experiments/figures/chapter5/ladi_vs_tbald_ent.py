import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import torch.utils.data as torchdata

from pathlib import Path
from torch import nn
import seaborn as sns
from torch.nn import functional as F

from alr.data.datasets import Dataset
from alr.utils import stratified_partition, manual_seed, savefig

manual_seed(42)
size = 200
VAL_SIZE = 5000
train, test = Dataset.CIFAR10.get(raw=True)
train, pool = stratified_partition(train, 10, size)
pool, val = torchdata.random_split(pool, (len(pool) - VAL_SIZE, VAL_SIZE))

true_labels = []
for _, y in pool:
    true_labels.append(y)
true_labels = torch.from_numpy(np.array(true_labels))

root = Path("/Users/harry/Documents/workspace/thesis/figures/4/data/ladi_vs_tbald/")

with open(root / "datum_200.pkl", "rb") as fp:
    data = pickle.load(fp)


def _xlogy(x, y):
    res = x * torch.log(y)
    res[y == 0] = .0
    assert torch.isfinite(res).all()
    return res


def tbald(history):
    def truncate(hist):
        E = hist.shape[0]
        # have at least 10 even if self._last percent of E is
        # less than 10. If E is less than 10, then take everything (E)
        e = max(min(10, E), int(E * 0.2))
        return hist[-e:].double()

    history = torch.from_numpy(history)
    mc_preds = truncate(history)
    mean_mc_preds = mc_preds.mean(dim=0)
    H = -(_xlogy(mean_mc_preds, mean_mc_preds)).sum(dim=1)
    E = (_xlogy(mc_preds, mc_preds)).sum(dim=2).mean(dim=0)
    I = H + E
    assert torch.isfinite(I).all()
    # result = torch.argsort(I, descending=True).numpy()
    return I, mean_mc_preds

def get_last_pl(history):
    return torch.from_numpy(history[-1])

score, mean_hist = tbald(data['label_hist'])
last = get_last_pl(data['label_hist'])
last_ent = -_xlogy(last, last).sum(1)
avg_ent = -_xlogy(mean_hist, mean_hist).sum(1)
sns.kdeplot(last_ent.numpy(), label=r'LADI ($history_{i}[T]$)')
sns.kdeplot(avg_ent.numpy(), label=r'TBALD ($\overline{history_{i}}$)')
plt.title("Entropy of average vs. last pseudo-label prediction")
plt.ylabel("Density")
plt.xlabel("Entropy")
plt.legend()
savefig("/Users/harry/Documents/workspace/thesis/figures/4/ladi_vs_tbald_ent.pdf")

# torch.eq(mean_hist.argmax(dim=1), true_labels).float().mean()
# torch.eq(get_last_pl(data['label_hist']).argmax(dim=1), true_labels).float().mean()

