import pytest

from torch import nn
from torch.nn.modules.dropout import _DropoutNd
from torch.nn import functional as F
from alr.modules.dropout import replace_dropout


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc = nn.Linear(10, 10)
        self.drop = nn.Dropout()

    def forward(self, x):
        return self.drop(self.fc(x))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(10, 10)
        self.drop = nn.Dropout(p=0.3)
        self.nn = Net1()

    def forward(self, x):
        return self.drop1(self.fc1(x))


def _is_persistent(mod):
    if isinstance(mod, _DropoutNd):
        assert type(mod).__name__.startswith("Persistent")


def _is_not_persistent(mod):
    if isinstance(mod, _DropoutNd):
        assert not type(mod).__name__.startswith("Persistent")


def test_dropout_replacement_clone():
    model = Net()
    model2 = replace_dropout(model, clone=True)
    model2.apply(_is_persistent)
    model.apply(_is_not_persistent)


def test_dropout_replacement_no_clone():
    model = Net()
    model2 = replace_dropout(model)
    model2.apply(_is_persistent)
    model.apply(_is_persistent)


def test_functional_dropout_warn():
    class WarnNet(nn.Module):
        def forward(self, x):
            F.dropout(x, .5, True)

    class WarnNet2(nn.Module):
        def forward(self, x):
            F.dropout2d(x, .5, True)
    with pytest.warns(UserWarning):
        replace_dropout(WarnNet())
    with pytest.warns(UserWarning):
        replace_dropout(WarnNet2())
