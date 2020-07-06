import copy
from typing import Callable, Optional, Tuple
from contextlib import contextmanager

import torch.utils.data as torchdata
import torch
import torchvision as tv
from alr.utils import stratified_partition, _map_device
from alr.utils._type_aliases import _DeviceType
from alr.training.samplers import RandomFixedLengthSampler, MinLabelledSampler
import numpy as np
from torch.nn import functional as F
from torch import nn
from torch.nn.utils import weight_norm

# constants
ALPHA = 1
BATCH_SIZE = 100
MIN_LABEL_PROP = 16
RFLS_LEN = 20_000
VAL_SIZE = 5_000
OPTIM = 'SGD'
OPTIM_KWARGS = dict(
    lr=0.1, momentum=0.9, weight_decay=1e-4
)
EPOCHS_INITIAL = 10
EPOCHS_FULL = 400
# reset optimiser between stage 1 and 2
RESET_OPTIM = True
# log error every 20 iterations
LOG_EVERY = 20
DROPOUT = 0.1

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    kwargs = dict(num_workers=4, pin_memory=True)
else:
    device = torch.device("cpu")
    kwargs = {}


def evaluate(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for x, y in loader:
            x, y = _map_device([x, y], device)
            preds = model(x)
            correct += torch.eq(
                preds.argmax(dim=-1), y.argmax(dim=-1)
            ).float().sum()
            total += y.size(0)
    return correct / total


# from https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py#L119
def mixup(x: torch.Tensor,
          y: torch.Tensor,
          alpha: float = 1.0,
          device: _DeviceType = None):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    if device:
        index = index.to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def reg_nll_loss(coef: Optional[Tuple[float, float]] = (.8, .4)):
    def _reg_nll_loss(pred: torch.Tensor,
                      target: torch.Tensor):
        C = target.size(-1)
        prob = pred.exp()
        # heuristic: empirical mean of mini-batch
        prob_avg = prob.mean(dim=0)
        # uniform prior
        prior = target.new_ones(C) / C

        # standard cross entropy loss: H[target, pred]
        ce_loss = -torch.mean(
            torch.sum(target * pred, dim=1)
        )
        # prior loss: KL(prior || empirical mean) = sum c=1..C of prior * log[prior/emp. mean]
        # note, this is simplified, the full prior loss is:
        #  sum(prior * log[prior] - prior * log[prob_avg])
        # but since the first term is a constant, we drop it.
        prior_loss = -torch.sum(
            prior * torch.log(prob_avg)
        )
        # entropy loss: neg. mean of sum c=1..C of p(y=c|x)log[p(y=c|x)]
        entropy_loss = -torch.mean(
            torch.sum(prob * pred, dim=1)
        )
        return ce_loss + coef[0] * prior_loss + coef[1] * entropy_loss

    return _reg_nll_loss


def reg_mixup_loss(coef: Optional[Tuple[float, float]] = (.8, .4)):
    def _reg_mixup_loss(pred: torch.Tensor,
                        y1: torch.Tensor,
                        y2: torch.Tensor,
                        lamb: int):
        """
        pred is log_softmax,
        y1 and y2 are softmax probabilities
        """
        C = y1.size(-1)
        assert y2.size(-1) == C
        # NxC
        prob = pred.exp()
        # C
        prob_avg = prob.mean(dim=0)
        prior = y2.new_ones(C) / C

        # term1, term2, [1,]
        term1 = -torch.mean(torch.sum(y1 * pred, dim=1))
        term2 = -torch.mean(torch.sum(y2 * pred, dim=1))
        mixup_loss = lamb * term1 + (1 - lamb) * term2

        prior_loss = -torch.sum(
            prior * torch.log(prob_avg)
        )
        entropy_loss = -torch.mean(
            torch.sum(prob * pred, dim=1)
        )

        return mixup_loss + coef[0] * prior_loss + coef[1] * entropy_loss

    return _reg_mixup_loss


# from: https://github.com/EricArazo/PseudoLabeling/blob/2fbbbd3ca648cae453e3659e2e2ed44f71be5906/utils_pseudoLab/ssl_networks.py
class Net(nn.Module):
    """
    CNN from Mean Teacher paper
    """

    def __init__(self, num_classes=10, dropRatio=.5):
        super(Net, self).__init__()

        self.activation = nn.LeakyReLU(0.1)
        self.conv1a = weight_norm(nn.Conv2d(3, 128, 3, padding=1))
        self.bn1a = nn.BatchNorm2d(128)
        self.conv1b = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1b = nn.BatchNorm2d(128)
        self.conv1c = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1c = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop = nn.Dropout(dropRatio)

        self.conv2a = weight_norm(nn.Conv2d(128, 256, 3, padding=1))
        self.bn2a = nn.BatchNorm2d(256)
        self.conv2b = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2b = nn.BatchNorm2d(256)
        self.conv2c = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2c = nn.BatchNorm2d(256)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)

        self.conv3a = weight_norm(nn.Conv2d(256, 512, 3, padding=0))
        self.bn3a = nn.BatchNorm2d(512)
        self.conv3b = weight_norm(nn.Conv2d(512, 256, 1, padding=0))
        self.bn3b = nn.BatchNorm2d(256)
        self.conv3c = weight_norm(nn.Conv2d(256, 128, 1, padding=0))
        self.bn3c = nn.BatchNorm2d(128)
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)

        self.fc1 = weight_norm(nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.activation(self.bn1a(self.conv1a(x)))
        x = self.activation(self.bn1b(self.conv1b(x)))
        x = self.activation(self.bn1c(self.conv1c(x)))
        x = self.mp1(x)
        x = self.drop(x)

        x = self.activation(self.bn2a(self.conv2a(x)))
        x = self.activation(self.bn2b(self.conv2b(x)))
        x = self.activation(self.bn2c(self.conv2c(x)))
        x = self.mp2(x)
        x = self.drop(x)

        x = self.activation(self.bn3a(self.conv3a(x)))
        x = self.activation(self.bn3b(self.conv3b(x)))
        x = self.activation(self.bn3c(self.conv3c(x)))
        x = self.ap3(x)
        x = x.view(-1, 128)
        return F.log_softmax(self.fc1(x), dim=-1)


train_transform = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010)),
])
test_transform = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010)),
])
data_augmentation = tv.transforms.Compose([
    tv.transforms.Pad(2, padding_mode='reflect'),
    tv.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    tv.transforms.RandomCrop(32),
    tv.transforms.RandomHorizontalFlip(),
])


def onehot_transform(n):
    def _onehot_transform(x):
        return torch.eye(n)[x]

    return _onehot_transform


cifar_train = tv.datasets.CIFAR10(
    root="data", train=True,  # leave transform for PDS
    download=True, target_transform=onehot_transform(10),
)
cifar_test = tv.datasets.CIFAR10(
    root="data", train=False, transform=test_transform,
    download=True, target_transform=onehot_transform(10),
)
test_loader = torchdata.DataLoader(
    cifar_test, shuffle=False, batch_size=512, **kwargs,
)


class IndexMarker(torchdata.Dataset):
    PSEUDO_LABELLED = True
    LABELLED = False

    def __init__(self, dataset: torchdata.Dataset, mark):
        self.dataset = dataset
        self.mark = mark

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # returns (x, y), idx, mark
        return self.dataset[idx], idx, self.mark


class PDS(torchdata.Dataset):
    def __init__(self,
                 dataset: IndexMarker,
                 transform: Callable[[torch.Tensor], torch.Tensor],
                 augmentation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        self.dataset = dataset
        self._augmentation = augmentation
        self._transform = transform
        self._with_metadata = True
        self._new_targets = None

    def __getitem__(self, idx):
        (img_raw, target), idx, mark = self.dataset[idx]

        # override target
        if self._new_targets is not None:
            target = self._new_targets[idx]

        if self._augmentation:
            img_aug = self._augmentation(img_raw)
            img_raw, img_aug = map(self._transform, [img_raw, img_aug])
        else:
            img_raw = self._transform(img_raw)
            img_aug = img_raw
        if self._with_metadata:
            return img_raw, img_aug, target, idx, mark
        return img_aug, target

    def __len__(self):
        return len(self.dataset)

    @contextmanager
    def no_fluff(self):
        if self._with_metadata:
            self._with_metadata = False
            yield self
            self._with_metadata = True

    @contextmanager
    def no_augmentation(self):
        if self._augmentation:
            store = self._augmentation
            self._augmentation = None
            yield self
            self._augmentation = store

    def override_targets(self, new_targets: torch.Tensor):
        assert new_targets.size(0) == len(self.dataset)
        self._new_targets = new_targets


train, pool = stratified_partition(cifar_train, 10, size=500)
pool, val = torchdata.random_split(pool, (len(pool) - VAL_SIZE, VAL_SIZE))
train = IndexMarker(train, mark=IndexMarker.LABELLED)
train = PDS(train, transform=train_transform, augmentation=data_augmentation)
model = Net(dropRatio=DROPOUT).to(device)
optimiser = getattr(torch.optim, OPTIM)(model.parameters(), **OPTIM_KWARGS)
optimiser_init_state = copy.deepcopy(optimiser.state_dict())
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimiser, milestones=[250, 350], gamma=.1
)
val = PDS(IndexMarker(val, mark=None), transform=test_transform, augmentation=None)
val._with_metadata = False

print("Stage 1")
with train.no_fluff():
    train_loader = torchdata.DataLoader(
        train, batch_size=BATCH_SIZE,
        sampler=RandomFixedLengthSampler(train, RFLS_LEN, shuffle=True),
        **kwargs
    )
    val_loader = torchdata.DataLoader(
        val, batch_size=512,
        shuffle=False, **kwargs,
    )
    for e in range(1, EPOCHS_INITIAL + 1):
        print(f"Epoch {e}/{EPOCHS_INITIAL}")
        for i, (x, y) in enumerate(train_loader, 1):
            model.train()
            x, y = _map_device([x, y], device)
            pred = model(x)
            loss = reg_nll_loss()(pred, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            if i % LOG_EVERY == 0:
                print(f"\tIteration {i}/{len(train_loader)}:"
                      f" loss = {loss.item():.4f}")
        print(f"\tval_acc = {evaluate(model, val_loader)}")

print(f"Stage 1 over, test acc = {evaluate(model, test_loader)}")

if RESET_OPTIM:
    optimiser.load_state_dict(optimiser_init_state)

# pseudo-label pool
pool = PDS(
    IndexMarker(pool,
                mark=IndexMarker.PSEUDO_LABELLED),
    transform=train_transform,
    augmentation=data_augmentation,
)
# DO NOT use augmentation when obtaining pseudo-labels
with pool.no_augmentation():
    with pool.no_fluff():
        pool_loader = torchdata.DataLoader(
            pool, batch_size=512, shuffle=False,
            **kwargs,
        )
        pseudo_labels = []
        with torch.no_grad():
            model.eval()
            for x, _ in pool_loader:
                x = x.to(device)
                # note: add probability, NOT log_softmax!
                pseudo_labels.append(model(x).exp().detach().cpu())

# update pseudo-labels
pool.override_targets(
    torch.cat(pseudo_labels)
)

# get full dataset
full_dataset = torchdata.ConcatDataset((train, pool))

fds_loader = torchdata.DataLoader(
    full_dataset, batch_sampler=MinLabelledSampler(
        train, pool, batch_size=BATCH_SIZE, min_labelled=MIN_LABEL_PROP,
    ), **kwargs,
)

print("Stage 2")
pseudo_labels = torch.empty(size=(len(pool), 10))
for e in range(1, EPOCHS_FULL + 1):
    print(f"Epoch {e}/{EPOCHS_FULL}")
    for i, (img_raw, img_aug, target, idx, mark) in enumerate(fds_loader, 1):
        img_raw, img_aug, target, idx, mark = \
            _map_device(
                [img_raw, img_aug, target, idx, mark],
                device
            )
        # train
        model.train()
        xp, y1, y2, lamb = mixup(img_aug, target, alpha=ALPHA)
        preds = model(xp)
        loss = reg_mixup_loss()(preds, y1, y2, lamb)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # update pseudo-labels
        with torch.no_grad():
            model.eval()
            pld_mask = mark == IndexMarker.PSEUDO_LABELLED
            # unaugmented, raw, pseudo-labelled images
            pld_img = img_raw[pld_mask]
            # get *softmax* predictions -- exponentiate the output!
            new_pld = model(pld_img).exp().detach().cpu()
            pseudo_labels[idx[pld_mask]] = new_pld

        if i % LOG_EVERY == 0:
            print(f"\tIteration {i}/{len(fds_loader)}:"
                  f" loss = {loss.item():.4f}")
    pool.override_targets(pseudo_labels)
    scheduler.step()
    print(f"\tval_acc = {evaluate(model, val_loader)}")

print(f"Stage 2 over, test acc = {evaluate(model, test_loader)}")
torch.save(model.state_dict(), "weights.pt")
