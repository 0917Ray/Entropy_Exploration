"""Plot entropy experiment JSONL metrics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import yaml
from matplotlib.colors import to_rgba


METRICS = ("entropy", "normalized_entropy", "confidence", "accuracy", "loss", "ece")
TRAIN_COLOR = "#5E887E"
TEST_COLOR = "#BA6580"
BLUE = "#5B7CA7"
AXIS_COLOR = "#3A3D42"
CLASSWISE_COLOR = TEST_COLOR


def _load(run_dir: Path):
    with (run_dir / "metrics/epoch_metrics.jsonl").open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _plot_config(config: dict | None) -> dict:
    config = config or {}
    plot = config.get("plot", {})
    markers = plot.get("markers", {})
    return {
        "smooth_enabled": bool(plot.get("smooth_enabled", True)),
        "smooth": max(1, int(plot.get("smooth", 5))),
        "show_original": bool(plot.get("show_original", True)),
        "markers_enabled": bool(markers.get("enabled", True)),
        "marker_count": markers.get("count", 10),
    }


def _marker_indices(size: int, count: int | str) -> np.ndarray:
    if isinstance(count, str) and count.upper() == "MAX":
        return np.arange(size)
    count = max(1, int(count))
    return np.arange(size) if size <= count else np.arange(0, size, int(np.ceil(size / count)))


def _save(fig, output: Path, config: dict | None = None):
    root = output.parents[2]
    category = f"{output.parent.parent.name}/{output.parent.name}"
    figure = (config or {}).get("figure", {})
    dpi = int(figure.get("dpi", 300))
    transparent = bool(figure.get("transparent", True))
    for extension in ("png", "pdf"):
        target = root / extension / category / f"{output.name}.{extension}"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=dpi, bbox_inches="tight", transparent=transparent)
    plt.close(fig)


def _style_axis(ax, *, y_grid=True, config=None):
    config = config or {}
    spine_width = float(config.get("figure", {}).get("spine_width", 2.5))
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(spine_width)
    ax.tick_params(width=1.1, length=4.2, color=AXIS_COLOR, direction="in",
                   top=True, right=True)
    ax.set_axisbelow(True)
    if not config.get("axis", {}).get("grid", True):
        ax.grid(False)
    else:
        ax.grid(True, which="major", axis="both", linestyle="--", linewidth=.75, alpha=.80)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.grid(False, which="minor", axis="x")
        ax.grid(True, which="minor", axis="y", linestyle="--", linewidth=.50, alpha=.34)
    axis = config.get("axis", {})
    ax.xaxis.set_major_locator(mticker.MaxNLocator(
        nbins=int(axis.get("target_x_ticks", 9)), integer=True,
        steps=[1, 2, 2.5, 5, 10], min_n_ticks=4))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(
        nbins=int(axis.get("target_y_ticks", 7)), integer=False,
        steps=[1, 2, 2.5, 5, 10], min_n_ticks=4))


def _moving_average(values, window=49):
    values = np.asarray(values, dtype=float)
    if window <= 1 or values.size <= 2:
        return values.copy()
    window = min(int(window), values.size)
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _series_style(config, style_key, split):
    series = (config or {}).get("series", {})
    base = dict(series.get(split, {}) or {})
    specific = series.get(style_key, {})
    if isinstance(specific, dict) and split in specific:
        base.update(specific[split] or {})
    return base


def _plot_series(ax, x, values, color, label, config=None, split=None, style_key=None):
    settings = _plot_config(config)
    values = np.asarray(values, dtype=float)
    smooth = _moving_average(values, settings["smooth"])
    if settings["show_original"] and settings["smooth_enabled"] and settings["smooth"] > 1:
        ax.plot(x, values, color=to_rgba(color, .35), linewidth=1.0, linestyle=":", zorder=1)
    selected = _series_style(config, style_key, split or "")
    style = {
        "color": color,
        "linewidth": float(selected.get("linewidth", 2.0)),
        "alpha": float(selected.get("line_alpha", .92)),
        "label": label, "zorder": 3,
    }
    if settings["markers_enabled"]:
        style.update(
            marker=selected.get("marker", "o"),
            markersize=float(selected.get("marker_size", 5.0)),
            markevery=_marker_indices(len(values), settings["marker_count"]),
            markerfacecolor=to_rgba(color, float(selected.get("marker_face_alpha", .6))),
            markeredgecolor=to_rgba(color, float(selected.get("marker_edge_alpha", .95))),
            markeredgewidth=float(selected.get("marker_edge_width", 1.35)),
        )
    ax.plot(x, smooth if settings["smooth_enabled"] else values, **style)


def _series_color(config, split, fallback, style_key=None):
    return _series_style(config, style_key, split).get("color", fallback)


def _line_plot(rows, key, output, config=None, ylabel=None, splits=("train", "test")):
    epochs = [r["epoch"] for r in rows]
    figsize = tuple((config or {}).get("figure", {}).get("figsize", [12, 4.5]))
    fig, ax = plt.subplots(figsize=figsize)
    for split, fallback in (("train", TRAIN_COLOR), ("test", TEST_COLOR)):
        if split not in splits:
            continue
        color = _series_color(config, split, fallback, f"epoch_{key}")
        values = np.asarray([r[split][key] for r in rows], dtype=float)
        _plot_series(ax, epochs, values, color, split, config, split=split, style_key=f"epoch_{key}")
        std_key = f"{key}_std"
        if all(std_key in r[split] for r in rows):
            spread = np.asarray([r[split][std_key] for r in rows], dtype=float)
            ax.fill_between(epochs, values - spread, values + spread, color=color, alpha=.16,
                            linewidth=0, label=f"{split} ± std")
    ax.set(xlabel=(config or {}).get("axis", {}).get("xlabel", "Epoch"),
           ylabel=ylabel or key.replace("_", " ").title())
    _style_axis(ax, config=config)
    if (config or {}).get("legend", {}).get("enabled", True):
        ax.legend(loc=(config or {}).get("legend", {}).get("loc", "best"), frameon=False)
    fig.tight_layout()
    _save(fig, output, config)


def _triad(rows, output, config=None, delta=False, rate=False, splits=("train", "test")):
    epochs = np.asarray([r["epoch"] for r in rows])
    figsize = tuple((config or {}).get("figure", {}).get("panel_figsize", [12, 4.5]))
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True)
    for ax, key in zip(axes, ("entropy", "confidence", "accuracy")):
        for split, fallback in (("train", TRAIN_COLOR), ("test", TEST_COLOR)):
            if split not in splits:
                continue
            color = _series_color(config, split, fallback, f"epoch_{key}")
            values = np.asarray([r[split][key] for r in rows], dtype=float)
            if delta:
                values = np.diff(values, prepend=values[0])
            if rate:
                previous = np.maximum(np.abs(values[:-1]), 1e-12)
                values = np.r_[0.0, np.diff(values) / previous]
            _plot_series(ax, epochs, values, color, split, config, split=split, style_key=f"epoch_{key}")
        ax.set_title(key.title())
        _style_axis(ax, config=config)
        ax.set_xlabel((config or {}).get("axis", {}).get("xlabel", "Epoch"))
    axes[0].set_ylabel("Relative change" if rate else ("Change" if delta else "Value"))
    if (config or {}).get("legend", {}).get("enabled", True):
        axes[-1].legend(loc=(config or {}).get("legend", {}).get("loc", "best"), frameon=False)
    fig.tight_layout()
    _save(fig, output, config)


def _classwise(rows, output_dir: Path, config: dict):
    selected = [0, 10, 50, 100]
    available = {r["epoch"]: r for r in rows}
    selected = [epoch for epoch in selected if epoch in available]
    for epoch in selected:
        row = available[epoch]
        figsize = tuple(config.get("figure", {}).get("figsize", [12, 4.5]))
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        for ax, key in zip(axes, ("entropy", "accuracy", "confidence")):
            values = [row["test"]["classwise"].get(str(c), {}).get(key, np.nan) for c in range(10)]
            ax.bar(range(10), values, width=.68, color=CLASSWISE_COLOR, alpha=.50,
                   edgecolor=AXIS_COLOR, linewidth=2.0)
            ax.set_title(key.title())
            ax.set_xlabel("Class")
            ax.set_xticks(range(10))
            if key in {"accuracy", "confidence"}:
                ax.set_ylim(0, 1.0)
            else:
                ax.set_ylim(0, max(values) * 1.12)
            _style_axis(ax, config=config)
        fig.suptitle(f"Test class-wise metrics, epoch {epoch}")
        fig.tight_layout()
        _save(fig, output_dir / f"classwise_epoch_{epoch:03d}", config)

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
            figsize = tuple(config.get("figure", {}).get("figsize", [12, 4.5]))
            fig, ax = plt.subplots(figsize=figsize)
            _plot_series(ax, epochs, values, CLASSWISE_COLOR, "Test", config, split="test", style_key="classwise")
            ax.set_title(f"Class {class_id + index_base}: {key.title()}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(key.title())
            if key == "accuracy":
                ax.set_ylim(0, 1.0)
            _style_axis(ax, config=config)
            fig.tight_layout()
            _save(fig, output_dir / f"class_{class_id + index_base:02d}_{key}_over_epoch", config)

    # Configurable groups, with separate entropy and accuracy figures.
    for group in normalized_groups:
        labels = [str(class_id + index_base) for class_id in group]
        suffix = "_".join(f"{class_id + index_base:02d}" for class_id in group)
        for key in ("entropy", "accuracy"):
            figsize = tuple(config.get("figure", {}).get("figsize", [12, 4.5]))
            fig, ax = plt.subplots(figsize=figsize)
            for class_id, label in zip(group, labels):
                values = [r["test"]["classwise"].get(str(class_id), {}).get(key, np.nan) for r in rows]
                _plot_series(ax, epochs, values, CLASSWISE_COLOR, f"Class {label}", config,
                             split="test", style_key="classwise")
            ax.set_title("Classes " + ", ".join(labels) + f": {key.title()}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(key.title())
            if key == "accuracy":
                ax.set_ylim(0, 1.0)
            _style_axis(ax, config=config)
            ax.legend(frameon=False, loc="best")
            fig.tight_layout()
            _save(fig, output_dir / f"classes_{suffix}_{key}_over_epoch", config)

    # Preserve a compact aggregate CSV-style figure only through the explicit config switch.
    if not config.get("include_aggregate_classwise", False):
        return
    figsize = tuple(config.get("figure", {}).get("panel_figsize", [12, 4.5]))
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    epochs = [r["epoch"] for r in rows]
    for class_id in range(10):
        for ax, key in zip(axes, ("entropy", "accuracy")):
            values = [r["test"]["classwise"].get(str(class_id), {}).get(key, np.nan) for r in rows]
            _plot_series(ax, epochs, values, CLASSWISE_COLOR, str(class_id), config,
                         split="test", style_key="classwise")
            ax.set_title(f"Class-wise {key}")
            ax.set_xlabel("Epoch")
            _style_axis(ax, config=config)
    axes[1].legend(title="Class", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    _save(fig, output_dir / "classwise_over_epoch", config)


def _reliability(rows, output_dir: Path, config=None):
    for epoch in (0, 10, 50, int(rows[-1]["epoch"])):
        row = next((r for r in rows if r["epoch"] == epoch), None)
        if row is None:
            continue
        # ECE is already aggregated; draw the available scalar as a compact summary.
        figsize = tuple((config or {}).get("figure", {}).get("figsize", [12, 4.5]))
        fig, ax = plt.subplots(figsize=figsize)
        confidence = row["test"]["confidence"]
        accuracy = row["test"]["accuracy"]
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.scatter([confidence], [accuracy], s=80, color=TEST_COLOR, label=f"ECE={row['test']['ece']:.3f}")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Accuracy",
               title=f"Reliability summary, epoch {epoch}")
        ax.legend(frameon=False)
        fig.tight_layout()
        _style_axis(ax, config=config)
        _save(fig, output_dir / f"reliability_epoch_{int(epoch):03d}", config)


def _batch_plot(run_dir: Path, output_dir: Path, config=None):
    path = run_dir / "metrics/batch_metrics.jsonl"
    if not path.is_file():
        return
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    x = np.arange(len(rows))
    for key in ("entropy", "loss"):
        figsize = tuple((config or {}).get("figure", {}).get("figsize", [12, 4.5]))
        fig, ax = plt.subplots(figsize=figsize)
        before = np.asarray([r["before"][key] for r in rows])
        after = np.asarray([r["after"][key] for r in rows])
        _plot_series(ax, x, before, TRAIN_COLOR, "Before", config, split="train", style_key=f"batch_{key}")
        _plot_series(ax, x, after, TEST_COLOR, "After", config, split="test", style_key=f"batch_{key}")
        ax.set_ylabel(key.title())
        _style_axis(ax, config=config)
        ax.legend(frameon=False, loc="best")
        ax.set_xlabel("Recorded batch")
        fig.tight_layout(); _save(fig, output_dir / f"batch_{key}", config)

    for key, color in (("entropy", TRAIN_COLOR), ("loss", TEST_COLOR)):
        figsize = tuple((config or {}).get("figure", {}).get("figsize", [12, 4.5]))
        fig, ax = plt.subplots(figsize=figsize)
        values = np.asarray([r["delta"][key] for r in rows])
        _plot_series(ax, x, values, color, f"Delta {key}", config,
                     split="train" if key == "entropy" else "test", style_key=f"batch_transfer_{key}")
        ax.axhline(0, color=AXIS_COLOR, linewidth=1.1)
        lower, upper = np.quantile(values, [.01, .99])
        margin = max((upper - lower) * .12, 1e-4)
        ax.set_ylim(lower - margin, upper + margin)
        ax.set_ylabel("After - before")
        ax.set_title(key.title())
        _style_axis(ax, config=config)
        ax.legend(frameon=False, loc="best")
        ax.set_xlabel("Recorded batch")
        fig.tight_layout(); _save(fig, output_dir / f"batch_transfer_{key}", config)


def render_entropy_bundle(run_dir: Path, config_path: Path | None = None, *,
                          smooth: int | None = None, smooth_enabled: bool | None = None,
                          show_original: bool | None = None,
                          markers_enabled: bool | None = None,
                          marker_count: str | int | None = None) -> Path:
    """Render all entropy figures and return the output directory."""
    run_dir = Path(run_dir)
    rows = _load(run_dir)
    config = {}
    if config_path:
        with Path(config_path).open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
    plot = config.setdefault("plot", {})
    markers = plot.setdefault("markers", {})
    if smooth is not None:
        plot["smooth"] = int(smooth)
    if smooth_enabled is not None:
        plot["smooth_enabled"] = bool(smooth_enabled)
    if show_original is not None:
        plot["show_original"] = bool(show_original)
    if markers_enabled is not None:
        markers["enabled"] = bool(markers_enabled)
    if marker_count is not None:
        markers["count"] = marker_count
    output_dir = run_dir / "outputs/entropy"
    for key in METRICS:
        for suffix, splits in (("", ("train", "test")), ("_train", ("train",)), ("_test", ("test",))):
            _line_plot(rows, key, output_dir / "epoch_metrics" / f"epoch_{key}{suffix}", config,
                       splits=splits)
    for mode in ("", "_train", "_test"):
        splits = ("train", "test") if mode == "" else (mode[1:],)
        _triad(rows, output_dir / "relationships" / f"epoch_entropy_confidence_accuracy{mode}",
               config, splits=splits)
        _triad(rows, output_dir / "relationships" / f"epoch_entropy_confidence_accuracy_delta{mode}",
               config, delta=True, splits=splits)
        _triad(rows, output_dir / "relationships" / f"epoch_entropy_confidence_accuracy_rate{mode}",
               config, rate=True, splits=splits)
    _classwise(rows, output_dir / "classwise", config)
    _reliability(rows, output_dir / "reliability", config)
    _batch_plot(run_dir, output_dir / "batch", config)
    return output_dir
