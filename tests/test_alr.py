import torch

from alr import MCDropout
from torch import nn
from torch.nn import functional as F


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc = nn.Linear(10, 10)
        self.drop = nn.Dropout()

    def forward(self, x):
        return self.drop(self.fc(x))


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc = nn.Linear(10, 10)
        self.drop = nn.Dropout()

    def forward(self, x):
        return F.log_softmax(self.drop(self.fc(x)), dim=-1)


def test_mcd_logsoft_consistency():
    # apply_softmax should be consistent with actually using F.log_softmax
    # in the model definition itself.
    model1 = MCDropout(Net1(), output_transform=lambda x: F.log_softmax(x, dim=-1), forward=10)
    model2 = MCDropout(Net2(), forward=10)
    model2.load_state_dict(model1.state_dict(), strict=True)
    model1.train()
    model2.train()
    tensor = torch.randn(size=(5, 10))
    torch.manual_seed(42)
    logsoft1 = model1(tensor)
    torch.manual_seed(42)
    logsoft2 = model2(tensor)
    assert torch.allclose(logsoft1, logsoft2)


def test_mcd_with_logsoft():
    # model's forward pass should sum to one
    model = MCDropout(Net1(), output_transform=lambda x: F.log_softmax(x, dim=-1), forward=10)
    model.train()
    output = model(torch.randn(size=(5, 10))).exp_().sum(dim=-1)
    assert torch.allclose(output, torch.ones_like(output))


def test_mcd_stochastic_fwd():
    # stochastic_forward's individual forward passes should sum to one
    model = MCDropout(Net1(), output_transform=lambda x: F.log_softmax(x, dim=-1), forward=10)
    model.eval()
    size = (12301, 10)
    output = model.stochastic_forward(torch.randn(size=size)).exp_()
    assert output.shape == (10, size[0], 10)
    output = output.sum(dim=-1)
    assert torch.allclose(output, torch.ones_like(output))


def test_mcd_stochastic_fwd_wo_logsoft():
    # stochastic_forward's individual forward passes should sum to one
    model = MCDropout(Net2(), forward=10)
    model.eval()
    size = (12301, 10)
    output = model.stochastic_forward(torch.randn(size=size)).exp_()
    assert output.shape == (10, size[0], 10)
    output = output.sum(dim=-1)
    assert torch.allclose(output, torch.ones_like(output))


def test_mcd_eval_forward():
    # since eval implies that forward will return averaged scores, it's
    # not going to sum to one.
    model = MCDropout(Net1(), output_transform=lambda x: F.log_softmax(x, dim=-1), forward=10)
    model.eval()
    output = model(torch.randn(size=(12309, 10))).exp_().sum(dim=-1)
    # if we're taking the mean of 10 forward passes, it shouldn't sum to one
    # of course, it could, but given an untrained Net1 with high dropout
    # probability and 10 hidden units, it's extremely unlikely
    assert not torch.allclose(output, torch.ones_like(output))


def test_mcd_eval_forward_consistent_with_predict():
    # regardless of model's mode, predict should have the same behaviour as forward
    model = MCDropout(Net1(), output_transform=lambda x: F.log_softmax(x, dim=-1), forward=10)
    model.eval()
    input = torch.randn(size=(12309, 10))
    torch.manual_seed(42)
    output = model(input).exp_()
    torch.manual_seed(42)
    output2 = model.predict(input).exp_()
    assert torch.allclose(output, output2)


def test_mcd_train_forward_consistent_with_predict():
    # regardless of model's mode, predict should have the same behaviour as forward
    model = MCDropout(Net1(), output_transform=lambda x: F.log_softmax(x, dim=-1), forward=10)
    model.train()
    input = torch.randn(size=(12309, 10))
    torch.manual_seed(42)
    output = model(input).exp_()
    torch.manual_seed(42)
    output2 = model.predict(input).exp_()
    assert torch.allclose(output, output2)
