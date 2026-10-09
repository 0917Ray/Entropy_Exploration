"""Generate publication-ready training curves for a training run."""

import argparse
import logging
from pathlib import Path

from cv_corruption.visualization.training_curves import (
    render_accuracy_bundle, render_bundle, render_seed_std_bundle, write_bundle,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smooth", "--smooth-window", dest="smooth", type=int, default=None,
                        help="Moving-average window for the displayed curves")
    parser.add_argument("--no-smooth", action="store_true", help="Disable curve smoothing")
    parser.add_argument("--no-original", action="store_true",
                        help="Hide the original low-alpha curve when smoothing is enabled")
    parser.add_argument("--no-markers", action="store_true", help="Hide curve markers")
    parser.add_argument("--markers", choices=["5", "10", "15", "20", "MAX"], default=None,
                        help="Maximum number of markers per curve")
    args = parser.parse_args(argv)
    if args.smooth is not None and args.smooth < 1:
        parser.error("--smooth must be >= 1")
    return args


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    data_path = write_bundle(run_dir, smooth=args.smooth, smooth_enabled=not args.no_smooth,
                             show_original=not args.no_original,
                             markers_enabled=not args.no_markers, marker_count=args.markers,
                             render=False)
    render_bundle(run_dir)
    render_accuracy_bundle(run_dir)
    render_seed_std_bundle(run_dir)
    logging.getLogger(__name__).info("Saved chart data and figures in %s (source: %s)", run_dir, data_path.name)


if __name__ == "__main__":
    main()
