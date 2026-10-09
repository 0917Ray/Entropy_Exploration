"""Render entropy experiment figures from an existing run."""

import argparse
import logging
from pathlib import Path

from cv_corruption.config.loader import config_path
from cv_corruption.visualization.entropy_curves import render_entropy_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/visualization/entropy_curves.yaml"))
    parser.add_argument("--smooth", "--smooth-window", dest="smooth", type=int, default=None)
    parser.add_argument("--no-smooth", action="store_true")
    parser.add_argument("--no-original", action="store_true")
    parser.add_argument("--no-markers", action="store_true")
    parser.add_argument("--markers", choices=["5", "10", "15", "20", "MAX"], default=None)
    args = parser.parse_args(argv)
    if args.smooth is not None and args.smooth < 1:
        parser.error("--smooth must be >= 1")
    config = config_path(args.config)
    output = render_entropy_bundle(
        args.run_dir.expanduser().resolve(), config,
        smooth=args.smooth, smooth_enabled=not args.no_smooth,
        show_original=not args.no_original, markers_enabled=not args.no_markers,
        marker_count=args.markers,
    )
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.info("Saved entropy figures in %s", output)


if __name__ == "__main__":
    main()
