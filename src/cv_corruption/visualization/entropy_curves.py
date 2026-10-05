"""Plot entropy experiment JSONL metrics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


METRICS = ("entropy", "normalized_entropy", "confidence", "accuracy", "loss", "ece")
TRAIN_COLOR = "#4F7C65"
TEST_COLOR = "#A75B73"
BLUE = "#516480"
AXIS_COLOR = "#27313d"
CLASS_COLORS = ["#516480", "#4F7C65", "#A75B73", "#B07A3A", "#6B5B95",
                "#3F7F82", "#8C6D5A", "#7A8B4A", "#9C5875", "#5F7186"]


def _load(run_dir: Path):
    with (run_dir / "metrics/epoch_metrics.jsonl").open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _save(fig, output: Path, dpi=300):
    root = output.parents[1]
    category = output.parent.name
    for extension in ("png", "pdf"):
        target = root / extension / category / f"{output.name}.{extension}"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)


def _style_axis(ax, *, y_grid=True):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(2.5)
    ax.tick_params(width=1.8, length=5, color=AXIS_COLOR)
    ax.set_axisbelow(True)
    ax.grid(axis="y" if y_grid else "both", linestyle="--", linewidth=.8, alpha=.28)


def _moving_average(values, window=49):
    values = np.asarray(values, dtype=float)
    if values.size < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _plot_sparse_observations(ax, x, values, color, label):
    stride = max(1, len(values) // 400)
    sample_x = x[::stride]
    sample_values = np.asarray(values)[::stride]
    ax.plot(sample_x, sample_values, color=color, alpha=.18, linewidth=.65,
            marker=".", markersize=1.8, label=label)


def _line_plot(rows, key, output, ylabel=None):
    epochs = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for split, color in (("train", "#4F7C65"), ("test", "#A75B73")):
        ax.plot(epochs, [r[split][key] for r in rows], label=split, color=color, linewidth=2)
    ax.set(xlabel="Epoch", ylabel=ylabel or key.replace("_", " ").title())
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    ax.grid(True, linestyle="--", alpha=.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, output)


def _triad(rows, output, delta=False, rate=False):
    epochs = np.asarray([r["epoch"] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)
    for ax, key in zip(axes, ("entropy", "confidence", "accuracy")):
        for split, color in (("train", "#4F7C65"), ("test", "#A75B73")):
            values = np.asarray([r[split][key] for r in rows], dtype=float)
            if delta:
                values = np.diff(values, prepend=values[0])
            if rate:
                previous = np.maximum(np.abs(values[:-1]), 1e-12)
                values = np.r_[0.0, np.diff(values) / previous]
            ax.plot(epochs, values, label=f"{split} observed", color=color, linewidth=1.0, alpha=.18)
            ax.plot(epochs, _moving_average(values, window=11), label=f"{split} smoothed",
                    color=color, linewidth=2.2, alpha=.96)
        ax.set_title(key.title())
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        ax.grid(True, linestyle="--", alpha=.3)
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Relative change" if rate else ("Change" if delta else "Value"))
    axes[-1].legend()
    fig.tight_layout()
    _save(fig, output)


def _classwise(rows, output_dir: Path, config: dict):
    selected = [0, 10, 50, 100]
    available = {r["epoch"]: r for r in rows}
    selected = [epoch for epoch in selected if epoch in available]
    for epoch in selected:
        row = available[epoch]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
        for ax, key in zip(axes, ("entropy", "accuracy", "confidence")):
            values = [row["test"]["classwise"].get(str(c), {}).get(key, np.nan) for c in range(10)]
            ax.bar(range(10), values, width=.68, color=BLUE, alpha=0.50,
                   edgecolor=AXIS_COLOR, linewidth=2.0)
            ax.set_title(key.title())
            ax.set_xlabel("Class")
            ax.set_xticks(range(10))
            if key in {"accuracy", "confidence"}:
                ax.set_ylim(0, 1.0)
            else:
                ax.set_ylim(0, max(values) * 1.12)
            _style_axis(ax)
        fig.suptitle(f"Test class-wise metrics, epoch {epoch}")
        fig.tight_layout()
        _save(fig, output_dir / f"classwise_epoch_{epoch:03d}")

    class_groups = config.get("class_groups", [[1, 10], [2, 3, 4]])
    index_base = int(config.get("class_index_base", 1))
    if index_base not in (0, 1):
        raise ValueError("class_index_base must be 0 or 1")
    normalized_groups = []
    for group in class_groups:
        ids = [int(value) - index_base for value in group]
        if not ids or any(class_id < 0 or class_id >= 10 for class_id in ids):
            raise ValueError("class_groups must contain valid CIFAR-10 class ids")
        normalized_groups.append(ids)

    # One file per class and metric keeps comparisons uncluttered.
    epochs = [r["epoch"] for r in rows]
    for class_id in range(10):
        for key in ("entropy", "accuracy"):
            values = [r["test"]["classwise"].get(str(class_id), {}).get(key, np.nan) for r in rows]
            fig, ax = plt.subplots(figsize=(7.2, 4.6))
            ax.plot(epochs, values, color=BLUE, linewidth=2.1, alpha=.95)
            ax.set_title(f"Class {class_id + index_base}: {key.title()}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(key.title())
            if key == "accuracy":
                ax.set_ylim(0, 1.0)
            _style_axis(ax)
            fig.tight_layout()
            _save(fig, output_dir / f"class_{class_id + index_base:02d}_{key}_over_epoch")

    # Configurable groups, with separate entropy and accuracy figures.
    for group in normalized_groups:
        labels = [str(class_id + index_base) for class_id in group]
        suffix = "_".join(f"{class_id + index_base:02d}" for class_id in group)
        for key in ("entropy", "accuracy"):
            fig, ax = plt.subplots(figsize=(7.4, 4.8))
            for class_id, label in zip(group, labels):
                values = [r["test"]["classwise"].get(str(class_id), {}).get(key, np.nan) for r in rows]
                ax.plot(epochs, values, color=CLASS_COLORS[class_id], linewidth=2.0,
                        alpha=.95, label=f"Class {label}")
            ax.set_title("Classes " + ", ".join(labels) + f": {key.title()}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(key.title())
            if key == "accuracy":
                ax.set_ylim(0, 1.0)
            _style_axis(ax)
            ax.legend(frameon=True, loc="best")
            fig.tight_layout()
            _save(fig, output_dir / f"classes_{suffix}_{key}_over_epoch")

    # Preserve a compact aggregate CSV-style figure only through the explicit config switch.
    if not config.get("include_aggregate_classwise", False):
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True)
    epochs = [r["epoch"] for r in rows]
    for class_id in range(10):
        for ax, key in zip(axes, ("entropy", "accuracy")):
            values = [r["test"]["classwise"].get(str(class_id), {}).get(key, np.nan) for r in rows]
            ax.plot(epochs, values, linewidth=1.4, label=str(class_id))
            ax.set_title(f"Class-wise {key}")
            ax.set_xlabel("Epoch")
            ax.grid(True, linestyle="--", alpha=.3)
            for spine in ax.spines.values():
                spine.set_linewidth(2.0)
    axes[1].legend(title="Class", ncol=2, fontsize=8)
    fig.tight_layout()
    _save(fig, output_dir / "classwise_over_epoch")


def _reliability(rows, output_dir: Path):
    for epoch in (0, 10, 50, rows[-1]["epoch"]):
        row = next((r for r in rows if r["epoch"] == epoch), None)
        if row is None:
            continue
        # ECE is already aggregated; draw the available scalar as a compact summary.
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        confidence = row["test"]["confidence"]
        accuracy = row["test"]["accuracy"]
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.scatter([confidence], [accuracy], s=80, color="#A75B73", label=f"ECE={row['test']['ece']:.3f}")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Accuracy",
               title=f"Reliability summary, epoch {epoch}")
        ax.grid(True, linestyle="--", alpha=.3)
        ax.legend()
        fig.tight_layout()
        _save(fig, output_dir / f"reliability_epoch_{epoch:03d}")


def _batch_plot(run_dir: Path, output_dir: Path):
    path = run_dir / "metrics/batch_metrics.jsonl"
    if not path.is_file():
        return
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    x = np.arange(len(rows))
    for key in ("entropy", "loss"):
        fig, ax = plt.subplots(figsize=(9.2, 4.8))
        before = np.asarray([r["before"][key] for r in rows])
        after = np.asarray([r["after"][key] for r in rows])
        _plot_sparse_observations(ax, x, before, TRAIN_COLOR, "Before (observed)")
        _plot_sparse_observations(ax, x, after, TEST_COLOR, "After (observed)")
        ax.plot(x, _moving_average(before, window=101), color=TRAIN_COLOR, linewidth=2.2, alpha=.98, label="Before (smoothed)")
        ax.plot(x, _moving_average(after, window=101), color=TEST_COLOR, linewidth=2.2, alpha=.98, label="After (smoothed)")
        ax.set_ylabel(key.title())
        _style_axis(ax)
        ax.legend(frameon=True, loc="best")
        ax.set_xlabel("Recorded batch")
        fig.tight_layout(); _save(fig, output_dir / f"batch_{key}")

    for key, color in (("entropy", TRAIN_COLOR), ("loss", TEST_COLOR)):
        fig, ax = plt.subplots(figsize=(9.2, 4.8))
        values = np.asarray([r["delta"][key] for r in rows])
        _plot_sparse_observations(ax, x, values, color, "Observed")
        ax.plot(x, _moving_average(values, window=101), color=color, alpha=.98, linewidth=2.2,
                label=f"Delta {key} (smoothed)")
        ax.axhline(0, color=AXIS_COLOR, linewidth=1.1)
        lower, upper = np.quantile(values, [.01, .99])
        margin = max((upper - lower) * .12, 1e-4)
        ax.set_ylim(lower - margin, upper + margin)
        ax.set_ylabel("After - before")
        ax.set_title(key.title())
        _style_axis(ax)
        ax.legend(frameon=True, loc="best")
        ax.set_xlabel("Recorded batch")
        fig.tight_layout(); _save(fig, output_dir / f"batch_transfer_{key}")


def render_entropy_bundle(run_dir: Path, config_path: Path | None = None) -> Path:
    """Render all entropy figures and return the output directory."""
    run_dir = Path(run_dir)
    rows = _load(run_dir)
    config = {}
    if config_path:
        with Path(config_path).open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
    output_dir = run_dir / "outputs/entropy"
    for key in METRICS:
        _line_plot(rows, key, output_dir / f"epoch_{key}")
    _triad(rows, output_dir / "epoch_entropy_confidence_accuracy")
    _triad(rows, output_dir / "epoch_entropy_confidence_accuracy_delta", delta=True)
    _triad(rows, output_dir / "epoch_entropy_confidence_accuracy_rate", rate=True)
    _classwise(rows, output_dir, config)
    _reliability(rows, output_dir)
    _batch_plot(run_dir, output_dir)
    return output_dir
