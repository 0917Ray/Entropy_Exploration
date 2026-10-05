"""Plot entropy experiment JSONL metrics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = ("entropy", "normalized_entropy", "confidence", "accuracy", "loss", "ece")


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
            ax.plot(epochs, values, label=split, color=color, linewidth=1.8)
        ax.set_title(key.title())
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        ax.grid(True, linestyle="--", alpha=.3)
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Relative change" if rate else ("Change" if delta else "Value"))
    axes[-1].legend()
    fig.tight_layout()
    _save(fig, output)


def _classwise(rows, output_dir: Path):
    selected = [0, 10, 50, 100]
    available = {r["epoch"]: r for r in rows}
    selected = [epoch for epoch in selected if epoch in available]
    for epoch in selected:
        row = available[epoch]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
        for ax, key in zip(axes, ("entropy", "accuracy", "confidence")):
            values = [row["test"]["classwise"].get(str(c), {}).get(key, np.nan) for c in range(10)]
            ax.bar(range(10), values, color="#516480", alpha=0.58,
                   edgecolor="#27313d", linewidth=1.5)
            ax.set_title(key.title())
            ax.set_xlabel("Class")
            ax.grid(axis="y", linestyle="--", alpha=.3)
            ax.set_axisbelow(True)
        fig.suptitle(f"Test class-wise metrics, epoch {epoch}")
        fig.tight_layout()
        _save(fig, output_dir / f"classwise_epoch_{epoch:03d}")

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
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, key in zip(axes, ("entropy", "loss")):
        ax.plot(x, [r["before"][key] for r in rows], alpha=.55, label="before")
        ax.plot(x, [r["after"][key] for r in rows], alpha=.8, label="after")
        ax.set_ylabel(key.title()); ax.grid(True, linestyle="--", alpha=.3); ax.legend()
    axes[-1].set_xlabel("Recorded batch")
    fig.tight_layout(); _save(fig, output_dir / "batch_entropy_loss")

    fig, ax = plt.subplots(figsize=(9, 4))
    for key, color in (("entropy", "#4F7C65"), ("loss", "#A75B73")):
        ax.plot(x, [r["delta"][key] for r in rows], label=f"Delta {key}", color=color, linewidth=1)
    ax.axhline(0, color="black", linewidth=.8); ax.set(xlabel="Recorded batch", ylabel="After - before")
    ax.grid(True, linestyle="--", alpha=.3); ax.legend(); fig.tight_layout(); _save(fig, output_dir / "batch_transfer_effect")


def render_entropy_bundle(run_dir: Path) -> Path:
    """Render all entropy figures and return the output directory."""
    run_dir = Path(run_dir)
    rows = _load(run_dir)
    output_dir = run_dir / "outputs/entropy"
    for key in METRICS:
        _line_plot(rows, key, output_dir / f"epoch_{key}")
    _triad(rows, output_dir / "epoch_entropy_confidence_accuracy")
    _triad(rows, output_dir / "epoch_entropy_confidence_accuracy_delta", delta=True)
    _triad(rows, output_dir / "epoch_entropy_confidence_accuracy_rate", rate=True)
    _classwise(rows, output_dir)
    _reliability(rows, output_dir)
    _batch_plot(run_dir, output_dir)
    return output_dir
