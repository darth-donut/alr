from efficientnet_pytorch import EfficientNet as EN
from torch import nn
from torch.nn import functional as F


class EfficientNet(nn.Module):
    def __init__(self, version=3, dropout_rate=0.5, num_classes=10):
        super(EfficientNet, self).__init__()
        params = dict(
            image_size=[32, 32], dropout_rate=dropout_rate, num_classes=num_classes
        )
        self.module = EN.from_name(f"efficientnet-b{version}", override_params=params)

    def forward(self, x):
        out = self.module(x)
        return F.log_softmax(out, dim=1)
