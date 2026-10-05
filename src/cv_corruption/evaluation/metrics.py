"""Classification metrics used by clean and corruption experiments."""

from __future__ import annotations

import math

import torch


def accuracy(correct: int, samples: int) -> float:
    if samples <= 0:
        raise ValueError("samples must be positive")
    return correct / samples


def batch_metrics(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 10,
                  ece_bins: int = 10) -> dict:
    """Return differentiable-free scalar and class-wise metrics for one batch."""
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("logits must be [N,C] and targets must be [N]")
    if logits.shape[1] != num_classes or ece_bins < 1:
        raise ValueError("invalid class count or ECE bin count")
    probabilities = logits.detach().float().softmax(dim=1)
    targets = targets.detach().long()
    confidence, predictions = probabilities.max(dim=1)
    entropy = -(probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log() * probabilities).sum(dim=1)
    correct = predictions.eq(targets)
    ece = expected_calibration_error(confidence, correct, ece_bins)
    class_values = {}
    for class_id in range(num_classes):
        mask = targets.eq(class_id)
        count = int(mask.sum())
        if count:
            class_values[str(class_id)] = {
                "samples": count,
                "entropy": float(entropy[mask].mean()),
                "normalized_entropy": float(entropy[mask].mean() / math.log(num_classes)),
                "confidence": float(confidence[mask].mean()),
                "accuracy": float(correct[mask].float().mean()),
                "loss": float(torch.nn.functional.cross_entropy(logits[mask], targets[mask])),
            }
        else:
            class_values[str(class_id)] = {"samples": 0, "entropy": None,
                                           "normalized_entropy": None, "confidence": None,
                                           "accuracy": None, "loss": None}
    return {
        "samples": int(targets.numel()),
        "entropy": float(entropy.mean()),
        "normalized_entropy": float(entropy.mean() / math.log(num_classes)),
        "confidence": float(confidence.mean()),
        "accuracy": float(correct.float().mean()),
        "loss": float(torch.nn.functional.cross_entropy(logits, targets)),
        "ece": float(ece),
        "classwise": class_values,
    }


def expected_calibration_error(confidence: torch.Tensor, correct: torch.Tensor, bins: int = 10) -> torch.Tensor:
    if bins < 1:
        raise ValueError("bins must be positive")
    confidence, correct = confidence.float(), correct.float()
    edges = torch.linspace(0, 1, bins + 1, device=confidence.device)
    total = confidence.numel()
    result = confidence.new_zeros(())
    for index in range(bins):
        mask = (confidence >= edges[index]) & (confidence <= edges[index + 1] if index == bins - 1 else confidence < edges[index + 1])
        if mask.any():
            result = result + mask.float().mean() * (correct[mask].mean() - confidence[mask].mean()).abs()
    return result


def weighted_average(records: list[dict]) -> dict:
    """Aggregate scalar metrics by sample count; class-wise values likewise."""
    if not records or sum(r.get("samples", 0) for r in records) <= 0:
        raise ValueError("records must contain samples")
    total = sum(r["samples"] for r in records)
    scalar_keys = ("entropy", "normalized_entropy", "confidence", "accuracy", "loss", "ece")
    result = {"samples": total}
    for key in scalar_keys:
        values = [r for r in records if key in r and r[key] is not None]
        if values:
            value_total = sum(r["samples"] for r in values)
            result[key] = sum(r[key] * r["samples"] for r in values) / value_total
    classes = {}
    for class_id in records[0].get("classwise", {}):
        rows = [r["classwise"][class_id] for r in records if r.get("classwise", {}).get(class_id, {}).get("samples", 0)]
        classes[class_id] = weighted_average(rows) if rows else {"samples": 0}
    result["classwise"] = classes
    return result
