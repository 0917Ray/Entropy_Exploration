"""CIFAR-10 pickle loader used by clean training and evaluation."""

import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from cv_corruption.models.resnet import MEAN, STD


class CIFAR10Pickles(Dataset):
    def __init__(self, root: str | Path, train: bool, augment: bool | None = None):
        root = Path(root)
        names = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
        arrays, labels = [], []
        for name in names:
            path = root / name
            if not path.is_file():
                raise FileNotFoundError(f"Missing CIFAR-10 file: {path}")
            with path.open("rb") as stream:
                batch = pickle.load(stream, encoding="bytes")
            images = batch[b"data"]
            targets = np.asarray(batch[b"labels"])
            if images.shape != (10000, 3072) or images.dtype != np.uint8:
                raise ValueError(f"Invalid image data in {path}: {images.shape}, {images.dtype}")
            if targets.shape != (10000,) or targets.min() < 0 or targets.max() > 9:
                raise ValueError(f"Invalid labels in {path}")
            arrays.append(images.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1))
            labels.append(targets)
        self.images = np.concatenate(arrays)
        self.labels = np.concatenate(labels)
        if augment is None:
            augment = train
        self.transform = transforms.Compose(
            ([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if augment else [])
            + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.transform(Image.fromarray(self.images[index])), int(self.labels[index])


def class_names(root: str | Path) -> list[str]:
    path = Path(root) / "batches.meta"
    if not path.is_file():
        raise FileNotFoundError(f"Missing CIFAR-10 metadata: {path}")
    with path.open("rb") as stream:
        meta = pickle.load(stream, encoding="bytes")
    names = [name.decode("ascii") for name in meta[b"label_names"]]
    if len(names) != 10 or len(set(names)) != 10:
        raise ValueError(f"Expected 10 distinct classes in {path}")
    return names
