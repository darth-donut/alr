import torch
import numpy as np
import copy

from alr import MCDropout, ALRModel
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
    model1 = MCDropout(
        Net1(), output_transform=lambda x: F.log_softmax(x, dim=-1), forward=10
    )
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
    model = MCDropout(
        Net1(), output_transform=lambda x: F.log_softmax(x, dim=-1), forward=10
    )
    model.train()
    output = model(torch.randn(size=(5, 10))).exp_().sum(dim=-1)
    assert torch.allclose(output, torch.ones_like(output))


def test_mcd_stochastic_fwd():
    # stochastic_forward's individual forward passes should sum to one
    model = MCDropout(
        Net1(), output_transform=lambda x: F.log_softmax(x, dim=-1), forward=10
    )
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


def test_mcd_eval_forward_logsumexp():
    # using log_softmax
    model = MCDropout(
        Net1(),
        reduce="logsumexp",
        output_transform=lambda x: F.log_softmax(x, dim=-1),
        forward=10,
    )
    model.eval()
    output = model(torch.randn(size=(12309, 10))).exp_().sum(dim=-1)
    assert torch.allclose(output, torch.ones_like(output))


def test_mcd_eval_forward_mean():
    # using softmax
    model = MCDropout(
        Net1(),
        reduce="mean",
        output_transform=lambda x: F.softmax(x, dim=-1),
        forward=10,
    )
    model.eval()
    output = model(torch.randn(size=(12309, 10))).sum(dim=-1)
    assert torch.allclose(output, torch.ones_like(output))


def test_mcd_eval_forward_consistent_with_predict():
    # when model's in eval, predict should have the same behaviour as forward
    model = MCDropout(
        Net1(), output_transform=lambda x: F.log_softmax(x, dim=-1), forward=10
    )
    model.eval()
    input = torch.randn(size=(12309, 10))
    torch.manual_seed(42)
    output = model(input).exp_()
    torch.manual_seed(42)
    # model.predict overrides model.train() with .eval()
    model.train()
    output2 = model.predict(input).exp_()
    assert torch.allclose(output, output2)


def test_mcd_fast_stochastic_fwd_flat_data():
    data = torch.from_numpy(np.random.normal(size=(1, 10))).float()
    net = MCDropout(Net2(), forward=50, fast=True)
    with torch.no_grad():
        preds = net.stochastic_forward(data)
        assert preds.size() == (50, 1, 10)
    # all n_forward instances of the data are identical,
    # but we assert that the output is stochastic, as required.
    # if the same dropout2d mask was used for each item in the batch, then
    # the variance wouldn't be 0
    assert (preds.var(dim=0) > 1e-3).all()


def test_mcd_fast_stochastic_fwd_img_data():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 5)
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
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = F.max_pool2d(self.dropout1(F.relu(self.conv1(x))), 2)
            x = F.max_pool2d(self.dropout2(F.relu(self.conv2(x))), 2)
            x = x.view(-1, 64 * 4 * 4)
            x = self.fc2(F.relu(self.fc1(x)))
            return F.log_softmax(x, dim=1)

    img = torch.from_numpy(np.random.normal(size=(1, 3, 28, 28))).float()
    net = MCDropout(Net(), forward=20, fast=True)
    with torch.no_grad():
        preds = net.stochastic_forward(img)
        assert preds.size() == (20, 1, 10)
    # all n_forward instances of the img are identical,
    # but we assert that the output is stochastic, as required.
    # if the same dropout2d mask was used for each item in the batch, then
    # the variance wouldn't be 0
    assert (preds.var(dim=0) > 1e-3).all()


def test_ALRModel_reset_weights_param():
    class A(ALRModel):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(10, 2)
            b = torch.ones(size=(2,))
            # this should be tracked and reset properly too!
            self.b = nn.Parameter(b)
            self.snap()

        def forward(self, x):
            return F.log_softmax(self.w(x) + self.b, dim=-1)

    data = torch.from_numpy(np.random.normal(size=(32, 10))).float()
    targets = torch.from_numpy(np.random.randint(0, 2, size=(32,)))
    net = A()

    optim = torch.optim.Adam(net.parameters())
    store = copy.deepcopy(net.state_dict())

    # train model to change weights
    for _ in range(50):
        preds = net(data)
        loss = F.nll_loss(preds, targets)
        optim.zero_grad()
        loss.backward()
        optim.step()

    # weights shouldn't be the same after they're trained
    with torch.no_grad():
        for k, v in net.state_dict().items():
            assert isinstance(v, torch.Tensor)
            assert (torch.abs(v - store[k]) > 1e-4).all()

    # reset weights
    net.reset_weights()

    # make sure weights are reset
    for k, v in net.state_dict().items():
        assert isinstance(v, torch.Tensor)
        assert torch.allclose(v, store[k])


def test_ALRModel_reset_weights():
    class A(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(10, 2)

        def forward(self, x):
            return F.log_softmax(self.w(x), dim=-1)

    data = torch.from_numpy(np.random.normal(size=(32, 10))).float()
    targets = torch.from_numpy(np.random.randint(0, 2, size=(32,)))
    net = MCDropout(A())

    optim = torch.optim.Adam(net.parameters())
    store = copy.deepcopy(net.state_dict())

    # train model to change weights
    for _ in range(50):
        preds = net(data)
        loss = F.nll_loss(preds, targets)
        optim.zero_grad()
        loss.backward()
        optim.step()

    # weights shouldn't be the same after they're trained
    with torch.no_grad():
        for k, v in net.state_dict().items():
            assert isinstance(v, torch.Tensor)
            assert (torch.abs(v - store[k]) > 1e-4).all()

    # reset weights
    net.reset_weights()

    # make sure weights are reset
    for k, v in net.state_dict().items():
        assert isinstance(v, torch.Tensor)
        assert torch.allclose(v, store[k])


def test_mc_dropout_fast_img_data(benchmark):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 5)
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
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = F.max_pool2d(self.dropout1(F.relu(self.conv1(x))), 2)
            x = F.max_pool2d(self.dropout2(F.relu(self.conv2(x))), 2)
            x = x.view(-1, 64 * 4 * 4)
            x = self.fc2(F.relu(self.fc1(x)))
            return F.log_softmax(x, dim=1)

    img = torch.from_numpy(np.random.normal(size=(32, 3, 28, 28))).float()

    def fast():
        net = MCDropout(Net(), forward=20, fast=True)
        with torch.no_grad():
            net.stochastic_forward(img)

    benchmark(fast)


def test_mc_dropout_regular_img_data(benchmark):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 5)
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
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = F.max_pool2d(self.dropout1(F.relu(self.conv1(x))), 2)
            x = F.max_pool2d(self.dropout2(F.relu(self.conv2(x))), 2)
            x = x.view(-1, 64 * 4 * 4)
            x = self.fc2(F.relu(self.fc1(x)))
            return F.log_softmax(x, dim=1)

    img = torch.from_numpy(np.random.normal(size=(32, 3, 28, 28))).float()

    def regular():
        net = MCDropout(Net(), forward=20, fast=False)
        with torch.no_grad():
            net.stochastic_forward(img)

    benchmark(regular)


def test_mc_dropout_fast_flat_data(benchmark):
    data = torch.from_numpy(np.random.normal(size=(32, 10))).float()

    def fast():
        net = MCDropout(Net2(), forward=50, fast=True)
        with torch.no_grad():
            net.stochastic_forward(data)

    benchmark(fast)


def test_mc_dropout_regular_flat_data(benchmark):
    data = torch.from_numpy(np.random.normal(size=(32, 10))).float()

    def regular():
        net = MCDropout(Net2(), forward=50, fast=False)
        with torch.no_grad():
            net.stochastic_forward(data)

    benchmark(regular)
