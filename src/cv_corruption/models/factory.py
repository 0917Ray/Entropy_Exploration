"""Model factory for experiment commands."""

from .resnet import make_model


def build_model(name="resnet50_cifar10"):
    if name != "resnet50_cifar10":
        raise ValueError(f"Unknown model architecture: {name}")
    return make_model()
