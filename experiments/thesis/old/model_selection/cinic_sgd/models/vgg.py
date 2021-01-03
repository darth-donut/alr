from torch import nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F

# vgg16_cinic10_bn(pretrained=True, num_classes=10)

model_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-c768596a.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg16_cinic10_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}


class VGG(nn.Module):
    """VGG with BatchNorm performs best.
    We only add MCDropout in the classifier head (where VGG used dropout before, too)."""

    def __init__(self, features, num_classes=10, init_weights=True, smaller_head=False):
        super().__init__()

        self.features = features
        if smaller_head:
            # self.avgpool = nn.Identity()
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512 * 1 * 1, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(True),
                nn.Linear(512, num_classes),
            )
        else:
            # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                # nn.Linear(512 * 7 * 7, 4096),
                nn.Linear(512 * 1 * 1, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )

        if init_weights:
            self.apply(self.initialize_weights)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x

    @staticmethod
    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "D2D_03":
            pass
        elif v == "D2D_04":
            pass  # layers += [mc_dropout.MCDropout2d(0.4)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}

# def _vgg(cfg, batch_norm, **kwargs):
#     return VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
def _vgg(
    arch,
    cfg,
    batch_norm,
    pretrained,
    progress,
    pretrained_features_only=False,
    **kwargs
):
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    if pretrained_features_only:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        fixed_state_dict = {
            path[len("features.") :]: state
            for path, state in state_dict.items()
            if "features." in path
        }

        model.features.load_state_dict(fixed_state_dict)
    return model


# def vgg16_cinic10_bn(**kwargs):
def vgg16_cinic10_bn(pretrained=False, progress=True, **kwargs):
    """
    VGG 16-layer model (configuration "D") with batch normalization
    Inspired by: https://github.com/geifmany/cifar-vgg/blob/master/cifar100vgg.py to follow
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7486599 and then gave up on Dropout in Conv layers
    and just used the smaller classifier head and reduced dropout.
    """
    # return _vgg(
    #     cfg="D",
    #     batch_norm=True,
    #     smaller_head=True,
    #     **kwargs,
    # )
    return _vgg(
        "vgg16_cinic10_bn",
        "D",
        True,
        progress=progress,
        pretrained_features_only=pretrained,
        pretrained=False,
        smaller_head=True,
        **kwargs,
    )
