"""Render entropy experiment figures from an existing run."""

import argparse
import logging
from pathlib import Path

from cv_corruption.visualization.entropy_curves import render_entropy_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = render_entropy_bundle(args.run_dir.expanduser().resolve())
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.info("Saved entropy figures in %s", output)


if __name__ == "__main__":
    main()
