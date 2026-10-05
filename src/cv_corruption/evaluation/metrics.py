"""Metric functions independent of a particular dataset."""


def accuracy(correct: int, samples: int) -> float:
    if samples <= 0:
        raise ValueError("samples must be positive")
    return correct / samples
