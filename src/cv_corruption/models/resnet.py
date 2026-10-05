"""CIFAR-10 ResNet-50 definition and input normalization."""

from torch import nn
from torchvision import models


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def make_model() -> nn.Module:
    model = models.resnet50(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
