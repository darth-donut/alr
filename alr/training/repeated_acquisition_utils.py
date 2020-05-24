import pickle
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as torchdata
from ignite.engine import create_supervised_evaluator, Events, Engine


class RelabelledDataset(torchdata.Dataset):
    def __init__(self, dataset: torchdata.Dataset, labels: np.ndarray):
        assert labels.shape[0] == len(dataset)
        self._dataset = dataset
        self._labels = labels

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx][0], self._labels[idx]


class PseudoLabelAcquirer:
    def __init__(self,
                 save_to: str,
                 step: int,
                 threshold=0.9,
                 pred_transform=lambda x: x.exp()):
        self._indices = []
        self._plabs = []
        self._pred_transform = pred_transform
        self._thresh = threshold
        self._targets = []
        self._preds = []
        if save_to is not None:
            self._save_to = Path(save_to)
            self._save_to.mkdir(parents=True, exist_ok=True)
        else:
            self._save_to = None
        self._step = step
        self._batch_size = None

    def _parse(self, engine: Engine):
        preds, targets = engine.state.output
        # state.iteration starts with 1
        iteration = engine.state.iteration - 1
        offset = iteration * self._batch_size

        self._targets.append(targets.cpu().numpy())
        self._preds.append(preds.cpu().numpy())
        with torch.no_grad():
            preds = self._pred_transform(preds)
            preds_max, plabs = torch.max(preds, dim=-1)
            mask = torch.nonzero(preds_max >= self._thresh).flatten()
            if mask.shape[0]:
                # plabs = [N,]
                self._plabs.append(plabs[mask])
                self._indices.append(mask + offset)

    def _flush(self, engine: Engine):
        # save to memory
        if self._indices and self._plabs:
            engine.state.pl_indices = torch.cat(self._indices)
            engine.state.pl_plabs = torch.cat(self._plabs)
        else:
            engine.state.pl_indices = torch.Tensor([])
            engine.state.pl_plabs = torch.Tensor([])

        # save to file
        if self._save_to is not None:
            fname = self._save_to / f"{self._step}_pl_predictions.pkl"
            assert not fname.exists(), "You've done goofed."
            with open(fname, "wb") as fp:
                payload = {
                    'preds': np.concatenate(self._preds, axis=0),
                    'targets': np.concatenate(self._targets, axis=0),
                }
                pickle.dump(payload, fp)

    def attach(self, engine: Engine, batch_size: int):
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._parse)
        engine.add_event_handler(Events.COMPLETED, self._flush)
        self._batch_size = batch_size


def get_confident_indices(model: torch.nn.Module,
                          dataset: torchdata.Dataset,
                          threshold: float,
                          root: str,
                          step: int,
                          device,
                          **kwargs):
    # mustn't shuffle to get the right indices!
    bs = 1024
    loader = torchdata.DataLoader(dataset, batch_size=bs, shuffle=False, **kwargs)
    evaluator = create_supervised_evaluator(model, metrics=None, device=device)
    PseudoLabelAcquirer(root, step, threshold).attach(evaluator, batch_size=bs)
    evaluator.run(loader)
    return evaluator.state.pl_indices.cpu().numpy(), evaluator.state.pl_plabs.cpu().numpy()

