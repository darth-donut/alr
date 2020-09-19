import numpy as np
import torch
import torch.utils.data as torchdata
import pickle

from ignite.engine import create_supervised_evaluator
from torch.nn import functional as F
from torch import nn

from alr import MCDropout
from alr.utils import stratified_partition, manual_seed
from alr.data.datasets import Dataset
from alr.training.samplers import EpochExtender
from alr.training import Trainer
from alr.training.pseudo_label_trainer import PLPredictionSaver

from pathlib import Path

manual_seed(42)
kwargs = dict(pin_memory=True, num_workers=4)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# instantiate a regular model and an optimiser.
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        # 32 24 24
        self.dropout1 = nn.Dropout2d()
        # maxpool --
        # 32 12 12
        self.conv2 = nn.Conv2d(32, 64, 5)
        # 64 8 8
        self.dropout2 = nn.Dropout2d()
        # maxpool --
        # 64 4 4
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.dropout3 = nn.Dropout()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(self.dropout1(F.relu(self.conv1(x))), 2)
        x = F.max_pool2d(self.dropout2(F.relu(self.conv2(x))), 2)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc2(self.dropout3(F.relu(self.fc1(x))))
        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    ITERS = 100
    EPOCHS = 50
    train, test = Dataset.MNIST.get()
    seeds = np.random.choice(int(1e5), replace=False, size=ITERS)
    test_loader = torchdata.DataLoader(
        test, batch_size=1024, shuffle=False, **kwargs,
    )
    buffer = []
    model_output = Path("saved_models")
    model_output.mkdir()
    pl_metric_output = Path("pl_testset_metrics")
    pl_metric_output.mkdir()
    indices = set()
    dup_indices = 0

    for i in range(ITERS):
        print(f"==== Iteration {i + 1} of {ITERS} ({(i + 1) / ITERS:.2%}) ====")
        manual_seed(seeds[i], det_cudnn=False)   # already did that det_cudnn thing
        sub_train, pool = stratified_partition(train, Dataset.MNIST.about.n_class, size=20)
        pool, val = torchdata.random_split(pool, [len(pool) - 5_000, 5_000])

        idxs = tuple(sub_train.indices)
        if idxs in indices:
            dup_indices += 1
        else:
            indices.add(idxs)

        payload = {
            'seed': seeds[i],
            'indices': idxs,
        }

        train_loader = torchdata.DataLoader(
            sub_train, batch_size=64,
            sampler=EpochExtender(sub_train, EPOCHS),
            **kwargs,
        )
        val_loader = torchdata.DataLoader(
            val, batch_size=1024, shuffle=False, **kwargs,
        )

        model = MCDropout(Net(), forward=20, fast=True).to(device)

        trainer = Trainer(
            model, F.nll_loss, 'Adam',
            patience=3, reload_best=True, device=device
        )

        history = trainer.fit(train_loader, val_loader, EPOCHS)

        payload['history'] = history
        test_metrics = trainer.evaluate(test_loader)
        payload['test_acc'] = test_metrics['acc']
        payload['test_loss'] = test_metrics['loss']

        save_pl_metrics = create_supervised_evaluator(
            model, metrics=None, device=device
        )
        PLPredictionSaver(pl_metric_output / f"{seeds[i]}").attach(save_pl_metrics)
        save_pl_metrics.run(test_loader)

        torch.save(model.state_dict(), model_output / f"{seeds[i]}.pth")
        buffer.append(payload)


    print(f"There were {dup_indices} duplicated set of indices.")
    
    with open("data.pkl", "wb") as fp:
        pickle.dump(buffer, fp)
