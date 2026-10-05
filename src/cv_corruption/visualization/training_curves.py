"""Training-metric conversion and plotting integration.

The chart-ready CSV deliberately contains one row per recorded epoch.  The
actual rendering is delegated to the reusable ``plot-training-curves`` skill
script when it is available in the Codex environment.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import shutil
from pathlib import Path

import yaml


SKILL_SCRIPT = Path(
    os.environ.get(
        "PLOT_TRAINING_CURVES_SCRIPT",
        "/home/fcr/.codex/skills/plot-training-curves/scripts/plot_training_curves.py",
    )
)
PROJECT_CONFIG = Path(__file__).resolve().parents[3] / "configs/visualization/training_curves.yaml"


def _paths(run_dir: Path) -> dict[str, Path]:
    return {
        "data": run_dir / "data",
        "configs": run_dir / "configs",
        "scripts": run_dir / "scripts",
        "logs": run_dir / "logs",
    }


def _organize_images(run_dir: Path) -> None:
    """Move renderer outputs into format folders and remove redundant prefixes."""
    outputs = run_dir / "outputs"
    sources = list(run_dir.glob("training_curves_figure_*"))
    sources.extend((run_dir / "configs").glob("training_curves_figure_*"))
    for source in sources:
        if source.suffix.lower() not in {".png", ".pdf"}:
            continue
        stem = source.stem.removeprefix("training_curves_figure_")
        target = outputs / source.suffix.lower().lstrip(".") / "training_curves" / f"{stem}{source.suffix.lower()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        source.replace(target)

    # The skill writes a self-contained script beside the output prefix. Keep
    # that useful artifact under scripts, while the canonical CSV/config stay
    # in data/ and configs/ respectively.
    generated_script = run_dir / "configs/training_curves_figure_plot.py"
    if generated_script.is_file():
        target = run_dir / "scripts/training_curves_plot.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        generated_script.replace(target)
    for redundant in (
        run_dir / "configs/training_curves_figure_data.csv",
        run_dir / "configs/training_curves_figure_config.json",
    ):
        if redundant.is_file():
            redundant.unlink()


def _merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def metrics_to_rows(metrics_path: Path) -> list[dict[str, float | int]]:
    """Read training JSONL and return stable, plot-ready epoch records."""
    rows: list[dict[str, float | int]] = []
    with Path(metrics_path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                row = {
                    "epoch": int(record["epoch"]),
                    "train_loss": float(record["train"]["loss"]),
                    "val_loss": float(record["test"]["loss"]),
                    "train_accuracy": float(record["train"]["accuracy"]),
                    "val_accuracy": float(record["test"]["accuracy"]),
                    "learning_rate": float(record["lr"]),
                }
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid metrics record at {metrics_path}:{line_number}") from exc
            rows.append(row)
    if not rows:
        raise ValueError(f"No metric records found in {metrics_path}")
    return rows


def write_bundle(run_dir: Path, *, smooth: int | None = None, render: bool = True,
                 template: Path | None = None) -> Path:
    """Create CSV/config/plot script and optionally render the standard figures."""
    run_dir = Path(run_dir)
    paths = _paths(run_dir)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    rows = metrics_to_rows(metrics_path)
    data_path = paths["data"] / "training_curves.csv"
    with data_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    template_path = Path(template) if template else PROJECT_CONFIG
    with template_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    config = _merge(config, {
        "data": {
            "input": "../data/training_curves.csv",
            "source": str(metrics_path.resolve()),
            "x": "epoch",
            "train_loss_key": "train_loss",
            "val_loss_key": "val_loss",
            "lr_key": "learning_rate",
        },
        "plot": {"smooth": int(smooth) if smooth is not None else config.get("plot", {}).get("smooth", 1)},
        "series": {
            "train_accuracy": {"color": "#4F7C65", "linewidth": 2.0,
                                "line_alpha": 0.92, "marker": "o", "marker_size": 4.5},
            "val_accuracy": {"color": "#A75B73", "linewidth": 2.0,
                              "line_alpha": 0.92, "marker": "o", "marker_size": 4.5},
        },
    })
    config_path = paths["configs"] / "training_curves.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    accuracy_script = paths["scripts"] / "accuracy_curves.py"
    shutil.copyfile(Path(__file__).with_name("accuracy_curves.py"), accuracy_script)
    if render:
        render_bundle(run_dir, config_path)
        render_accuracy_bundle(run_dir, config_path)
    return data_path


def render_accuracy_bundle(run_dir: Path, config_path: Path | None = None) -> None:
    """Render train/validation accuracy curves from the saved chart data."""
    config_path = config_path or Path(run_dir) / "configs/training_curves.json"
    log_path = Path(run_dir) / "logs/training_curves.log"
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, str(Path(run_dir) / "scripts/accuracy_curves.py"),
             "--config", str(config_path)],
            cwd=Path(run_dir), check=True, stdout=log, stderr=log,
        )


def render_bundle(run_dir: Path, config_path: Path | None = None) -> None:
    """Render the bundle with the skill's standalone plotting implementation."""
    config_path = config_path or Path(run_dir) / "configs/training_curves.json"
    if not SKILL_SCRIPT.is_file():
        raise FileNotFoundError(
            f"plot-training-curves script not found: {SKILL_SCRIPT}; "
            "set PLOT_TRAINING_CURVES_SCRIPT to its path"
        )
    log_path = Path(run_dir) / "logs/training_curves.log"
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, str(SKILL_SCRIPT), "--config", str(config_path)],
            cwd=Path(run_dir),
            check=True,
            stdout=log,
            stderr=log,
        )
    _organize_images(Path(run_dir))
