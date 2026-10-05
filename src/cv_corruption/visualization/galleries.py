"""Shared CIFAR gallery I/O and axis styling."""

import pickle

import matplotlib.pyplot as plt
import numpy as np


LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def load_clean(path):
    path = path / "test_batch" if path.is_dir() else path
    with path.open("rb") as stream:
        batch = pickle.load(stream, encoding="bytes")
    images = np.asarray(batch[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(batch[b"labels"], dtype=np.int64)
    return images, labels


def class_indices(labels, sample_index=0):
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    selected = []
    for label in range(10):
        matches = np.flatnonzero(labels == label)
        if len(matches) <= sample_index:
            raise ValueError(f"No sample {sample_index} for label {label}")
        selected.append(int(matches[sample_index]))
    return selected


def style_axis(ax, image, args):
    ax.imshow(image, interpolation=args.interpolation)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(args.border_width > 0)
        spine.set_color(args.border_color)
        spine.set_linewidth(args.border_width)


def save_figure(fig, path, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight", pad_inches=args.pad_inches,
                transparent=getattr(args, "transparent", True))
    plt.close(fig)


def validate_plot_args(args):
    if args.dpi <= 0 or args.font_size <= 0 or args.border_width < 0 or args.pad_inches < 0:
        raise ValueError("DPI/font size must be positive; border width/padding non-negative")
    for name in ("figsize", "row_figsize", "overview_figsize", "level5_figsize", "classes_figsize"):
        value = getattr(args, name, None)
        if value is not None and (len(value) != 2 or any(v <= 0 for v in value)):
            raise ValueError(f"{name} must contain two positive dimensions in inches")


def validate_groups(groups, severities):
    if not isinstance(groups, dict) or not groups:
        raise ValueError("groups must be a nonempty mapping")
    names = []
    for group, values in groups.items():
        if not isinstance(group, str) or not group or "/" in group or "\\" in group or group in {".", ".."}:
            raise ValueError("Group names must be nonempty filename components")
        if not isinstance(values, list) or not values or any(not isinstance(name, str) or not name for name in values):
            raise ValueError(f"groups.{group} must be a nonempty list of corruption names")
        names.extend(values)
    if len(names) != len(set(names)):
        raise ValueError("Corruption names must not be duplicated across groups")
    if not severities or len(set(severities)) != len(severities) or any(s not in range(1, 6) for s in severities):
        raise ValueError("severities must be distinct integers in 1..5")
    return names
