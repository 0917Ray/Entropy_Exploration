"""Atomic checkpoint persistence."""

import os
from pathlib import Path

import torch


def save_checkpoint(path: Path, checkpoint: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)
