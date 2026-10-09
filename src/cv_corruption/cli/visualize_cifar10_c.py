"""Generate Tiny ImageNet-C-style CIFAR-10-C galleries in PNG and PDF."""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from cv_corruption.config.loader import parse_config, save_config
from cv_corruption.data.cifar10_c import CIFAR10CReader
from cv_corruption.visualization.galleries import (
    LABELS, class_indices, load_clean, style_axis, validate_groups, validate_plot_args,
)

CV_ROOT = PROJECT_ROOT / "CV_Corruption"
DATA = CV_ROOT / "data/raw/cifar10_c/CIFAR-10-C"
CLEAN = CV_ROOT / "data/raw/cifar10/cifar-10-batches-py/test_batch"
OUT = CV_ROOT / "runs/dataset_visualizations/cifar10_c"
GROUPS = {
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"],
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "gaussian_blur"],
    "weather": ["snow", "frost", "fog", "brightness", "spatter", "saturate"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}
CORRUPTIONS = [name for names in GROUPS.values() for name in names]
STANDARD_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]
EXTENDED_CORRUPTIONS = ["speckle_noise", "gaussian_blur", "spatter", "saturate"]
LEVEL5_CORRUPTIONS = STANDARD_CORRUPTIONS + EXTENDED_CORRUPTIONS


def ordered_level5_corruptions(corruptions):
    """Return the selected corruptions in canonical CIFAR-10-C order."""
    selected = set(corruptions)
    return [name for name in LEVEL5_CORRUPTIONS if name in selected]


def save(fig, stem, args):
    stem = Path(stem)
    for extension, enabled in (("png", args.save_png), ("pdf", args.save_pdf)):
        if enabled:
            path = args.output / "images" / extension / stem.with_suffix(f".{extension}")
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=args.dpi,
                        bbox_inches="tight", pad_inches=args.pad_inches,
                        transparent=args.transparent)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", "--data-dir", type=Path, default=DATA)
    parser.add_argument("--clean", "--clean-data", type=Path, default=CLEAN)
    parser.add_argument("--output", "--output-dir", type=Path, default=OUT)
    parser.add_argument("--corruptions", type=str, nargs="+", choices=CORRUPTIONS)
    parser.add_argument("--severities", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--selected-label", type=int, default=0)
    parser.add_argument("--selected-index", type=int, help="Absolute index in the clean test batch")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--overview-figsize", type=float, nargs=2, default=[10.5, 31])
    parser.add_argument("--level5-figsize", type=float, nargs=2, default=[16, 5.2])
    parser.add_argument("--classes-figsize", type=float, nargs=2, default=[10, 5.2])
    parser.add_argument("--group-width", type=float, default=10.5)
    parser.add_argument("--row-height", type=float, default=1.7)
    parser.add_argument("--font-size", type=float, default=10.0)
    parser.add_argument("--title-font-size", type=float, default=15.0)
    parser.add_argument("--row-font-size", type=float, default=8.0)
    parser.add_argument("--class-font-size", type=float, default=13.0)
    parser.add_argument("--label-pad", type=float, default=34.0)
    parser.add_argument("--interpolation", default="nearest")
    parser.add_argument("--border-width", type=float, default=1.8)
    parser.add_argument("--border-color", default="#3A3D42")
    parser.add_argument("--level5-border-color", default="#D55E00")
    parser.add_argument("--pad-inches", type=float, default=0.08)
    parser.add_argument("--transparent", action=argparse.BooleanOptionalAction, default=True,
                        help="Use a transparent figure background for PNG and PDF output")
    parser.add_argument("--title-y", type=float, default=0.998)
    parser.add_argument("--column-title-pad", type=float, default=8.0)
    parser.add_argument("--level5-title-y", type=float, default=0.98)
    parser.add_argument("--level5-legend-anchor", type=float, nargs=2, default=[0.5, 0.005])
    parser.add_argument("--class-title-pad", type=float, default=10.0)
    for name in ("save-png", "save-pdf", "show-title", "generate-overview", "generate-level5-overview",
                 "generate-groups", "generate-classes"):
        parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=True)
    parser.set_defaults(groups=GROUPS, labels=LABELS)
    args = parse_config(parser, argv, default_config=CV_ROOT / "configs/visualization/cifar10_c.yaml",
                        path_fields=("data", "clean", "output"))
    names = validate_groups(args.groups, args.severities)
    if any(name not in CORRUPTIONS for name in names):
        parser.error("groups contains an unsupported CIFAR-10-C corruption")
    if args.corruptions is None:
        args.corruptions = names
    if any(name not in names for name in args.corruptions) or len(set(args.corruptions)) != len(args.corruptions):
        parser.error("corruptions must be distinct members of groups")
    args.groups = {group: [name for name in values if name in args.corruptions]
                   for group, values in args.groups.items() if any(name in args.corruptions for name in values)}
    validate_plot_args(args)
    if not 0 <= args.selected_label < 10 or args.sample_index < 0 or len(args.labels) != 10:
        parser.error("selected_label must be in 0..9; sample_index non-negative; labels must have ten names")
    if not (args.save_png or args.save_pdf):
        parser.error("Enable save_png or save_pdf")
    if any(v <= 0 for v in (args.group_width, args.row_height, args.title_font_size, args.row_font_size,
                            args.class_font_size, *args.level5_figsize)):
        parser.error("Figure dimensions and font sizes must be positive")
    return args


def gallery(clean, index, corruptions, title, figsize, args, reader):
    fig, axes = plt.subplots(len(corruptions), len(args.severities) + 1, figsize=figsize, squeeze=False)
    if args.show_title:
        fig.suptitle(title, fontsize=args.title_font_size, fontweight="bold", y=args.title_y)
    for column, header in enumerate(["Clean"] + [f"Severity {s}" for s in args.severities]):
        axes[0, column].set_title(header, fontsize=args.font_size, pad=args.column_title_pad)
    for row, corruption in enumerate(corruptions):
        style_axis(axes[row, 0], clean[index], args)
        axes[row, 0].set_ylabel(corruption.replace("_", "\n"), rotation=0,
                               labelpad=args.label_pad, va="center", fontsize=args.row_font_size)
        for column, severity in enumerate(args.severities, 1):
            style_axis(axes[row, column], reader.get(corruption, severity, index), args)
    fig.tight_layout(rect=(0.04, 0, 1, 0.99), h_pad=0.7, w_pad=0.35)
    return fig


def level5_gallery(index, corruptions, args, reader):
    """Build a two-row gallery containing one level-5 image per corruption."""
    fig, axes = plt.subplots(2, 10, figsize=args.level5_figsize, squeeze=False)
    if args.show_title:
        fig.suptitle("CIFAR-10-C corruptions | Severity 5",
                     fontsize=args.title_font_size, fontweight="bold", y=args.level5_title_y)

    extended = set(EXTENDED_CORRUPTIONS)
    for axis, corruption in zip(axes.flat, corruptions):
        style_axis(axis, reader.get(corruption, 5, index), args)
        border_color = args.level5_border_color if corruption in extended else args.border_color
        for spine in axis.spines.values():
            spine.set_color(border_color)
        axis.set_title(corruption.replace("_", " "), fontsize=args.row_font_size, pad=6)

    for axis in axes.flat[len(corruptions):]:
        axis.set_visible(False)

    legend_handles = [
        Patch(facecolor="white", edgecolor=args.border_color, linewidth=args.border_width,
              label="Original 15 corruptions"),
        Patch(facecolor="white", edgecolor=args.level5_border_color, linewidth=args.border_width,
              label="Added 4 corruptions"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=args.font_size, frameon=False,
               bbox_to_anchor=tuple(args.level5_legend_anchor))
    fig.tight_layout(rect=(0.01, 0.09, 0.99, 0.91), w_pad=0.35, h_pad=0.8)
    return fig


def main():
    args = parse_args()
    clean, labels = load_clean(args.clean)
    selected = class_indices(labels, args.sample_index)
    index = args.selected_index if args.selected_index is not None else selected[args.selected_label]
    if not 0 <= index < len(clean) or labels[index] != args.selected_label:
        raise ValueError("selected_index must be a clean test index belonging to selected_label")
    args.selected_index = index
    reader = CIFAR10CReader(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.generate_overview:
        fig = gallery(clean, index, args.corruptions, f"CIFAR-10-C: {len(args.corruptions)} corruptions",
                      args.overview_figsize, args, reader)
        save(fig, "overview/all", args)
    if args.generate_level5_overview:
        level5_corruptions = ordered_level5_corruptions(args.corruptions)
        fig = level5_gallery(index, level5_corruptions, args, reader)
        save(fig, "overview/level5", args)
    if args.generate_groups:
        for group, corruptions in args.groups.items():
            fig = gallery(clean, index, corruptions, f"CIFAR-10-C | {group}",
                          [args.group_width, args.row_height * len(corruptions) + 0.6], args, reader)
            save(fig, f"groups/{group}", args)
    if args.generate_classes:
        fig, axes = plt.subplots(2, 5, figsize=args.classes_figsize, squeeze=False)
        if args.show_title:
            fig.suptitle("CIFAR-10-C: clean examples by label", fontsize=args.title_font_size,
                         y=args.title_y)
        for ax, label, idx in zip(axes.flat, range(10), selected):
            style_axis(ax, clean[idx], args)
            ax.set_title(f"{label}: {args.labels[label]}", fontsize=args.class_font_size,
                         pad=args.class_title_pad)
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.86, hspace=0.62, wspace=0.22)
        save(fig, "classes/labels", args)
    (args.output / "configs").mkdir(parents=True, exist_ok=True)
    save_config(args.output / "configs" / "resolved_config.json", args)
    print(f"Generated {len(args.corruptions)} corruption types in {args.output}")


if __name__ == "__main__":
    main()
