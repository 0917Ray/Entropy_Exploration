"""Create one labeled CIFAR-10 example per class as a PDF figure."""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from cv_corruption.config.loader import parse_config, save_config
from cv_corruption.visualization.galleries import (
    LABELS, class_indices, load_clean, save_figure, style_axis, validate_plot_args,
)


CV_ROOT = PROJECT_ROOT / "CV_Corruption"
DATA = CV_ROOT / "data/raw/cifar10/cifar-10-batches-py"
OUTPUT = CV_ROOT / "runs/dataset_visualizations/cifar10/cifar10_label_examples_two_rows.pdf"
OUTPUT_ROW = CV_ROOT / "runs/dataset_visualizations/cifar10/cifar10_label_examples_single_row.pdf"
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", "--data-dir", type=Path, default=DATA)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-row", type=Path)
    parser.add_argument("--labels", type=str, nargs=10, default=LABELS)
    parser.add_argument("--sample-index", type=int, default=0, help="Nth test image within each class")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--figsize", type=float, nargs=2, default=[11, 5.6])
    parser.add_argument("--row-figsize", type=float, nargs=2, default=[20, 3.1])
    parser.add_argument("--font-size", type=float, default=18.0)
    parser.add_argument("--title-font-size", type=float, default=20.0)
    parser.add_argument("--interpolation", default="nearest")
    parser.add_argument("--show-title", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--border-width", type=float, default=2.2)
    parser.add_argument("--border-color", default="#3A3D42")
    parser.add_argument("--pad-inches", type=float, default=0.12)
    parser.add_argument("--transparent", action=argparse.BooleanOptionalAction, default=True,
                        help="Use a transparent figure background for PNG and PDF output")
    parser.add_argument("--title-y", type=float, default=0.98)
    parser.add_argument("--label-pad", type=float, default=8.0)
    args = parse_config(parser, argv, default_config=CV_ROOT / "configs/visualization/cifar10_examples.yaml",
                        path_fields=("data", "output", "output_row"))
    validate_plot_args(args)
    if len(args.labels) != 10 or args.sample_index < 0 or args.title_font_size <= 0:
        parser.error("labels must contain ten names; sample index non-negative; title font size positive")
    if args.output is None and args.output_row is None:
        parser.error("At least one output must be enabled")
    if args.output == args.output_row:
        parser.error("output and output_row must be different files")
    return args


def main():
    args = parse_args()
    images, labels = load_clean(args.data)
    selected = class_indices(labels, args.sample_index)
    for path, shape, figsize in ((args.output, (2, 5), args.figsize),
                                (args.output_row, (1, 10), args.row_figsize)):
        if path is None:
            continue
        fig, axes = plt.subplots(*shape, figsize=figsize, squeeze=False, constrained_layout=True)
        if args.show_title:
            fig.suptitle("CIFAR-10: one test example per label", fontsize=args.title_font_size, y=args.title_y)
        for ax, label, index in zip(axes.flat, range(10), selected):
            style_axis(ax, images[index], args)
            if args.show_labels:
                ax.set_title(f"{label}: {args.labels[label]}", fontsize=args.font_size, pad=args.label_pad)
        name = path.stem.replace("cifar10_label_examples_", "label_examples_").replace("two_rows", "overview").replace("single_row", "row")
        root = path.parent
        for extension in ("png", "pdf"):
            output = root / "images" / extension / f"{name}.{extension}"
            save_figure(fig, output, args)
        save_config(root / "configs" / "resolved_config.json", args)
        print(name)


if __name__ == "__main__":
    main()
