# alr: Active Learning Research Library

A library for your daily active learning needs (WIP). This library also offers semi-supervised learning trainers
and acquisition functions that complement these learning algorithms.

# Example

## BALD-10 on MNIST

Here's an example of greedily-acquiring top BALD-scoring points on the MNIST dataset:

```python
import torch
import torch.utils.data as torchdata
from torch.nn import functional as F

from alr import MCDropout
from alr.acquisition import BALD
from alr.data import DataManager, UnlabelledDataset
from alr.data.datasets import Dataset
from alr.training import Trainer
from alr.utils import manual_seed, eval_fwd_exp, timeop

# reproducibility
manual_seed(42, det_cudnn=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
kwargs = dict(num_workers=4, pin_memory=True)

# ========= CONSTANTS ===========
BATCH_SIZE = 64
EPOCHS = 50
VAL_SIZE = 5_000
ITERS = 21
BATCH_ACQUIRE_SIZE = 10

# ========= SETUP ===========
train, pool, test = Dataset.MNIST.get_fixed()
pool, val = torchdata.random_split(pool, (len(pool) - VAL_SIZE, VAL_SIZE))
pool = UnlabelledDataset(pool)

pool_loader = torchdata.DataLoader(
    pool, batch_size=1024, shuffle=False, **kwargs,
)
val_loader = torchdata.DataLoader(
    val, batch_size=1024, shuffle=False, **kwargs,
)
test_loader = torchdata.DataLoader(
    test, batch_size=1024, shuffle=False, **kwargs,
)
accs = []

model = MCDropout(Dataset.MNIST.model, forward=20, fast=True).to(device)
acq_fn = BALD(eval_fwd_exp(model), device=device, batch_size=1024, **kwargs)
dm = DataManager(train, pool, acq_fn)

# =========== RUN =============
for i in range(1, ITERS + 1):
    model.reset_weights()
    trainer = Trainer(
        model, F.nll_loss, optimiser='Adam',
        patience=3, reload_best=True, device=device
    )
    train_loader = torchdata.DataLoader(
        dm.labelled, batch_size=BATCH_SIZE, **kwargs
    )
    with timeop() as t:
        history = trainer.fit(train_loader, val_loader, epochs=EPOCHS)

    test_metrics = trainer.evaluate(test_loader)

    print(f"=== Iteration {i} of {ITERS} ({i / ITERS:.2%}) ===")
    print(f"\ttrain: {dm.n_labelled}; val: {len(val)}; "
          f"pool: {dm.n_unlabelled}; test: {len(test)}")
    print(f"\t[test] acc: {test_metrics['acc']:.4f}, time: {t}")
    accs.append(test_metrics['acc'])
    dm.acquire(b=BATCH_ACQUIRE_SIZE)

print(accs)
```

# Todo

[ ] Add example of SSL + AL
[ ] Generate Sphinx doc
