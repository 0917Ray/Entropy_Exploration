import unittest

import torch

from cv_corruption.evaluation.metrics import batch_metrics, expected_calibration_error, weighted_average


class MetricsTests(unittest.TestCase):
    def test_batch_metrics_and_ranges(self):
        logits = torch.tensor([[8.0, 0.0], [0.0, 8.0]])
        result = batch_metrics(logits, torch.tensor([0, 1]), num_classes=2)
        self.assertAlmostEqual(result["accuracy"], 1.0)
        self.assertGreaterEqual(result["normalized_entropy"], 0.0)
        self.assertLessEqual(result["normalized_entropy"], 1.0)
        self.assertAlmostEqual(result["ece"], 1 - result["confidence"], places=5)

    def test_weighted_average_uses_samples(self):
        rows = [
            {"samples": 1, "entropy": 0.0, "normalized_entropy": 0.0, "confidence": 1.0,
             "accuracy": 1.0, "loss": 0.0, "ece": 0.0, "classwise": {}},
            {"samples": 3, "entropy": 1.0, "normalized_entropy": 1.0, "confidence": 0.0,
             "accuracy": 0.0, "loss": 2.0, "ece": 1.0, "classwise": {}},
        ]
        self.assertAlmostEqual(weighted_average(rows)["entropy"], 0.75)

    def test_ece_empty_bins(self):
        value = expected_calibration_error(torch.tensor([0.1]), torch.tensor([True]), bins=10)
        self.assertGreaterEqual(float(value), 0.0)


if __name__ == "__main__":
    unittest.main()
