"""Standalone accuracy-curve renderer for a training run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_rows(config_path: Path) -> list[dict[str, float]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_path = config_path.parent / config["data"]["input"]
    with data_path.open(encoding="utf-8", newline="") as stream:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def render(config_path: Path) -> list[Path]:
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = load_rows(config_path)
    if not rows:
        raise ValueError("accuracy curve data is empty")
    x = [row["epoch"] for row in rows]
    # Accuracy uses the same series interface as loss and learning rate.
    series_config = config.get("series", {})
    legacy = config.get("accuracy", {})
    legacy_colors = legacy.get("colors", {"train": "#4F7C65", "val": "#A75B73"})
    train_style = series_config.get("train_accuracy", {
        "color": legacy_colors["train"], "linewidth": legacy.get("linewidth", 2.0),
        "line_alpha": legacy.get("line_alpha", 0.92), "marker": legacy.get("marker", "o"),
        "marker_size": legacy.get("marker_size", 4.5),
    })
    val_style = series_config.get("val_accuracy", {
        "color": legacy_colors["val"], "linewidth": legacy.get("linewidth", 2.0),
        "line_alpha": legacy.get("line_alpha", 0.92), "marker": legacy.get("marker", "o"),
        "marker_size": legacy.get("marker_size", 4.5),
    })
    # Configs live at run/configs; rendered figures belong in run/outputs.
    output_dir = config_path.parent.parent / "outputs"
    dpi = config["figure"]["dpi"]
    transparent = config["figure"]["transparent"]
    outputs = []
    series = [("train_accuracy", "Training accuracy", train_style),
              ("val_accuracy", "Validation accuracy", val_style)]
    for suffix, title, selected in (
        ("a_train_accuracy", "(e) Training Accuracy", [series[0]]),
        ("b_val_accuracy", "(f) Validation Accuracy", [series[1]]),
        ("c_train_vs_val_accuracy", "(g) Training vs. Validation Accuracy", series),
    ):
        fig, ax = plt.subplots(figsize=config["figure"]["figsize"])
        for key, label, color in selected:
            ax.plot(x, [row[key] * 100 for row in rows], label=label,
                    color=color["color"], linewidth=color["linewidth"], alpha=color["line_alpha"],
                    marker=color["marker"], markersize=color["marker_size"])
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
        fig.tight_layout()
        for extension in ("png", "pdf") if config["output"]["pdf"] else ("png",):
            path = output_dir / extension / f"{suffix}.{extension}"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=dpi, transparent=transparent, bbox_inches="tight")
            outputs.append(path)
        plt.close(fig)
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    for path in render(args.config.resolve()):
        print(path)


if __name__ == "__main__":
    main()
