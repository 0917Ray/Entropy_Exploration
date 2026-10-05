"""Evaluate a clean CIFAR-10 checkpoint on the 15 standard CIFAR-10-C corruptions."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cv_corruption.data.cifar10 import CIFAR10Pickles, class_names
from cv_corruption.data.cifar10_c import CorruptionSlice
from cv_corruption.evaluation.corruption import CORRUPTIONS
from cv_corruption.models.resnet import make_model
from cv_corruption.config.loader import parse_config, save_config

CV_ROOT = PROJECT_ROOT / "CV_Corruption"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=CV_ROOT / "data/raw/cifar10_c/CIFAR-10-C")
    parser.add_argument("--clean-data-dir", type=Path, default=CV_ROOT / "data/raw/cifar10/cifar-10-batches-py")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--corruptions", type=str, nargs="+", choices=CORRUPTIONS, default=list(CORRUPTIONS))
    parser.add_argument("--severities", nargs="+", type=int, default=list(range(1, 6)))
    parser.add_argument("--limit", type=int, help="Evaluate only the first N images per severity (quick check)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--cpu", action=argparse.BooleanOptionalAction, default=False)
    args = parse_config(parser, argv, path_fields=("checkpoint", "data_dir", "clean_data_dir", "output"))
    if not args.checkpoint:
        parser.error("checkpoint is required (set it in YAML or pass --checkpoint)")
    if args.batch_size < 1 or args.workers < 0 or args.limit is not None and not 1 <= args.limit <= 10000:
        parser.error("batch-size must be positive, workers non-negative, and limit in 1..10000")
    if any(severity not in range(1, 6) for severity in args.severities):
        parser.error("severities must be in 1..5")
    if len(set(args.severities)) != len(args.severities) or len(set(args.corruptions)) != len(args.corruptions):
        parser.error("severities and corruptions must not contain duplicates")
    return args


def main():
    args = parse_args()
    if not args.cpu and (not torch.cuda.is_available() or args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count()):
        raise ValueError("requested GPU unavailable; use --cpu to evaluate on CPU")
    device = torch.device("cpu" if args.cpu else f"cuda:{args.gpu_id}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["config"]["architecture"] != "resnet50_cifar10" or checkpoint["class_names"] != class_names(args.clean_data_dir):
        raise ValueError("Checkpoint architecture or class order does not match local CIFAR-10")
    labels = np.load(args.data_dir / "labels.npy", allow_pickle=False)
    clean_labels = CIFAR10Pickles(args.clean_data_dir, train=False).labels
    if labels.shape == (50000,):
        labels = labels[:10000]
    if labels.shape != (10000,) or not np.array_equal(labels, clean_labels):
        raise ValueError("CIFAR-10-C labels do not match the clean CIFAR-10 test set")
    model = make_model().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    results = []
    for corruption in args.corruptions:
        for severity in args.severities:
            dataset = CorruptionSlice(args.data_dir / f"{corruption}.npy", labels, severity,
                                      checkpoint["normalization"], args.limit)
            loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                                pin_memory=device.type == "cuda")
            correct = 0
            with torch.inference_mode():
                for images, targets in loader:
                    logits = model(images.to(device, non_blocking=True))
                    correct += (logits.argmax(1).cpu() == targets).sum().item()
            result = {"corruption": corruption, "severity": severity, "accuracy": correct / len(dataset),
                      "samples": len(dataset)}
            results.append(result)
            print(f"{corruption} severity={severity}: {result['accuracy']:.2%} ({len(dataset)} images)", flush=True)
    output = args.output or args.checkpoint.parent.parent / "cifar10_c_metrics.json"
    args.output = output
    save_config(output.with_name(output.stem + "_config.json"), args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump({"checkpoint": str(args.checkpoint.resolve()), "results": results,
                   "mean_accuracy": sum(row["accuracy"] for row in results) / len(results)}, stream, indent=2)
    print(f"Mean accuracy: {sum(row['accuracy'] for row in results) / len(results):.2%}; saved: {output}")


if __name__ == "__main__":
    main()
