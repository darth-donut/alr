import numpy as np
import torch
from torch.nn import functional as F
import torch.utils.data as torchdata
from alr.training.pseudo_label_trainer import PLTracker, soft_nll_loss, soft_cross_entropy
from alr.data.datasets import Dataset
from alr.utils.math import entropy
from ignite.engine import Engine, Events


def test_pl_tracker_metrics():
    tracker = PLTracker(entropy_fn=entropy, device='cpu')

    train, test = Dataset.MNIST.get()
    train, _ = torchdata.random_split(train, [32, 59_968])
    train_loader = torchdata.DataLoader(train, batch_size=32, shuffle=False)

    all_probs = []

    def work(e, batch):
        in_x, in_y = next(iter(train_loader))
        x = tracker.process_batch(batch)
        assert torch.eq(x, in_x).all()
        logits = torch.randn(size=(32, 10))
        logits[torch.arange(logits.size(0)), in_y] = 5
        probs = F.log_softmax(logits, dim=-1)
        assert torch.eq(torch.argmax(probs, dim=-1), in_y).all()
        if e.state.epoch == 1:
            # acc = 30/32
            logits[31, (in_y[-1] + 1) % 10] = 6
            logits[30, (in_y[-2] + 1) % 10] = 7
            probs = F.log_softmax(logits, dim=-1)
            acc = torch.eq(torch.argmax(probs, dim=-1), in_y).float().mean()
            assert acc == 30/32
            tracker.record_predictions(probs)
        else:
            # acc = 31/32
            logits[31, (in_y[-1] + 1) % 10] = 8
            probs = F.log_softmax(logits, dim=-1)
            acc = torch.eq(torch.argmax(probs, dim=-1), in_y).float().mean()
            assert acc == 31/32
            tracker.record_predictions(probs)
        all_probs.append(probs)

    accs = []
    confs = []
    entropies = []
    engine = Engine(work)
    tracker.attach(engine)

    @engine.on(Events.EPOCH_COMPLETED)
    def record(e):
        accs.append(e.state.pl_tracker['acc'])
        confs.append(e.state.pl_tracker['confidence'])
        entropies.append(e.state.pl_tracker['entropy'])

    engine.run(train_loader, max_epochs=2, epoch_length=1)

    for i, acc in enumerate([30/32, 31/32]):
        assert accs[i] == acc
        assert len(confs[i]) == 32
        assert np.equal(np.array(confs[i]), np.amax(all_probs[i].numpy(), axis=-1)).all()
        assert torch.isfinite(entropy(all_probs[i]).sum(dim=-1)).all()
        assert len(entropies[i]) == 32
        assert torch.allclose(torch.Tensor(entropies[i]), entropy(all_probs[i]).sum(dim=-1))


def test_soft_nll_loss():
    F = torch.nn.functional
    target_logits = torch.randn(size=(100, 10))
    logits = torch.randn(size=(100, 10))

    # calculated expected cross entropy
    target_dist = F.softmax(target_logits, dim=-1)
    probs = F.softmax(logits, dim=-1)
    cross_entropy = -(target_dist * torch.log(probs)).sum(dim=-1).mean()

    # now, test nll
    res = soft_nll_loss(preds=F.log_softmax(logits, dim=-1), target=F.log_softmax(target_logits, dim=-1))
    assert torch.isclose(res, cross_entropy)


def test_soft_nll_loss_trivial():
    N = 1000
    target_logits = torch.randperm(N) % 10
    one_hot = torch.eye(10)[target_logits]
    assert torch.eq(torch.argmax(one_hot, dim=-1), target_logits).all()

    logits = torch.randn(size=(N, 10))
    preds = F.log_softmax(logits, dim=-1)
    a = F.nll_loss(preds, target_logits)
    b = soft_nll_loss(preds, target=(one_hot + 1e-8).log())
    assert torch.isclose(a, b)


def test_soft_cross_entropy():
    target_logits = torch.randn(size=(100, 10))
    logits = torch.randn(size=(100, 10))

    # calculated expected cross entropy
    target_dist = F.softmax(target_logits, dim=-1)
    probs = F.softmax(logits, dim=-1)
    cross_entropy = -(target_dist * torch.log(probs)).sum(dim=-1).mean()

    # now, test cross entropy
    res = soft_cross_entropy(logits, target=target_logits)
    assert torch.isclose(res, cross_entropy)


def test_soft_cross_entropy_trivial():
    N = 1000
    target_logits = torch.randperm(N) % 10
    target_logits_r = torch.randn(size=(N, 10))
    # make the argmax value so high that it will cause the softmax dist to be essentially one-hot
    target_logits_r[torch.arange(N), target_logits] = 1e10

    logits = torch.randn(size=(N, 10))
    a = F.cross_entropy(logits, target_logits)
    b = soft_cross_entropy(logits, target=target_logits_r)
    assert torch.isclose(a, b)
