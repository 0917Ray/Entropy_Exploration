"""CIFAR-10-C dataset primitives used by evaluation commands."""

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CIFAR10CReader:
    """Memory-map each selected corruption once and serve indexed images."""

    def __init__(self, root):
        self.root = Path(root)
        self._arrays = {}

    def _load(self, corruption):
        if corruption not in self._arrays:
            path = self.root / f"{corruption}.npy"
            images = np.load(path, mmap_mode="r", allow_pickle=False)
            if images.shape != (50000, 32, 32, 3) or images.dtype != np.uint8:
                raise ValueError(f"Invalid CIFAR-10-C image array in {path}: {images.shape}, {images.dtype}")
            self._arrays[corruption] = images
        return self._arrays[corruption]

    def get(self, corruption, severity, index):
        if severity not in range(1, 6):
            raise ValueError("severity must be in 1..5")
        if not 0 <= index < 10000:
            raise IndexError("CIFAR-10-C sample index must be in 0..9999")
        return self._load(corruption)[(severity - 1) * 10000 + index]


class CorruptionSlice(Dataset):
    def __init__(self, path, labels, severity, normalization, limit=None):
        images = np.load(Path(path), mmap_mode="r", allow_pickle=False)
        if images.shape != (50000, 32, 32, 3) or images.dtype != np.uint8:
            raise ValueError(f"Invalid CIFAR-10-C image array: {images.shape}, {images.dtype}")
        start = (severity - 1) * 10000
        count = min(limit or 10000, 10000)
        self.images = images[start:start + count]
        self.labels = labels[:count]
        self.transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(normalization["mean"], normalization["std"])
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.transform(Image.fromarray(self.images[index])), int(self.labels[index])
