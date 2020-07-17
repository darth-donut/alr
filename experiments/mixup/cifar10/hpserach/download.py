import torchvision as tv
tv.datasets.CIFAR10(
    root="data", train=True,
    download=True
)
tv.datasets.CIFAR10(
    root="data", train=False,
    download=True
)

