"""Lightweight configuration helpers.

The command-specific argparse definitions remain close to their commands;
this module provides a stable place for future typed schemas.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    data: Path
    output: Path
