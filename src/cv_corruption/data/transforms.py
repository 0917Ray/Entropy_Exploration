"""Shared transform construction hooks."""

from torchvision import transforms


def normalize(mean, std):
    return transforms.Normalize(mean, std)
