from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PACKAGE_ROOT.parent
CV_ROOT = PROJECT_ROOT
DATA_ROOT = CV_ROOT / "data"
RUNS_ROOT = CV_ROOT / "runs"
