"""Validate sample configs without loading datasets or starting training."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from cv_corruption.cli import train
from cv_corruption.cli import evaluate as evaluation
from cv_corruption.cli import visualize_cifar10_c as show_cifar10_c
from cv_corruption.cli import visualize_examples as show_cifar10_examples
from cv_corruption.cli import visualize_tiny_imagenet_c as show_tiny_imagenet_c
from cv_corruption.config.loader import CV_ROOT, resolved_config, save_config
from cv_corruption.cli.train import timestamped_run_dir
from cv_corruption.visualization.training_curves import metrics_to_rows, write_bundle


class EntryConfigTests(unittest.TestCase):
    def test_train_file_and_cli_overrides(self):
        args = train.parse_args(["--config", str(CV_ROOT / "configs/training/cifar10_resnet50.yaml"),
                                 "--epochs", "20", "--batch-size", "256", "--gpu-ids", "1", "--no-amp"])
        self.assertEqual((args.epochs, args.batch_size, args.gpu_ids, args.amp), (20, 256, [1], False))

    def test_legacy_resume_leaves_recipe_unset(self):
        args = train.parse_args(["--resume", "CV_Corruption/runs/test/checkpoints/last.pt", "--cpu"])
        self.assertIsNone(args.batch_size)
        self.assertIsNone(args.lr)
        self.assertIsNone(args.amp)
        self.assertIsNone(args.output_dir)

    def test_smoke_caps_epochs(self):
        args = train.parse_args(["--config", str(CV_ROOT / "configs/training/cifar10_resnet50.yaml"), "--smoke"])
        self.assertEqual(args.epochs, 1)

    def test_eval_file_and_legacy_checkpoint(self):
        args = evaluation.parse_args(["--config", str(CV_ROOT / "configs/evaluation/cifar10_c.yaml"),
                                      "--corruptions", "gaussian_noise", "--severities", "1", "--limit", "16"])
        self.assertEqual((args.corruptions, args.severities, args.limit), (["gaussian_noise"], [1], 16))
        args = evaluation.parse_args(["--checkpoint", "CV_Corruption/runs/other/checkpoints/best.pt"])
        self.assertIsNone(args.output)

    def test_all_plot_defaults_parse(self):
        examples = show_cifar10_examples.parse_args([])
        self.assertEqual(examples.figsize, [11, 5.6])
        self.assertEqual(examples.data.name, "test_batch")
        self.assertTrue(examples.transparent)
        cifar = show_cifar10_c.parse_args([])
        self.assertEqual(len(cifar.corruptions), 19)
        self.assertTrue(cifar.generate_level5_overview)
        self.assertEqual(cifar.level5_figsize, [16, 5.2])
        self.assertTrue(cifar.transparent)
        self.assertEqual(show_cifar10_c.ordered_level5_corruptions(cifar.corruptions),
                         show_cifar10_c.LEVEL5_CORRUPTIONS)
        tiny = show_tiny_imagenet_c.parse_args([])
        self.assertEqual(tiny.tile_size, 160)
        self.assertTrue(tiny.transparent)
        self.assertEqual(tiny.report_path, tiny.output / "reports/report.md")

    def test_gallery_config_round_trips(self):
        with tempfile.TemporaryDirectory(prefix="cv-gallery-config-") as directory:
            path = Path(directory) / "resolved_config.json"
            for entry in (show_cifar10_examples, show_cifar10_c, show_tiny_imagenet_c):
                with self.subTest(entry=entry.__name__):
                    original = entry.parse_args([])
                    save_config(path, original)
                    loaded = entry.parse_args(["--config", str(path)])
                    self.assertEqual(resolved_config(loaded), resolved_config(original))

    def test_subset_plot_config(self):
        args = show_cifar10_c.parse_args(["--set", "groups={noise: [gaussian_noise]}", "--severities", "1", "3"])
        self.assertEqual(args.corruptions, ["gaussian_noise"])
        self.assertEqual(args.severities, [1, 3])
        args = show_tiny_imagenet_c.parse_args(["--set", "groups={digital: [jpeg_compression]}",
                                               "--no-save-pdf", "--class-count", "3"])
        self.assertFalse(args.save_pdf)
        self.assertEqual(args.class_count, 3)
        args = show_cifar10_c.parse_args(["--no-generate-level5-overview"])
        self.assertFalse(args.generate_level5_overview)

    def test_eval_rejects_invalid_values_from_config_overrides(self):
        for assignment in ("corruptions=[typo]", "severities=[]", "severities=[6]", "batch_size=0", "amp=false"):
            with self.subTest(assignment=assignment), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    evaluation.parse_args(["--config", str(CV_ROOT / "configs/evaluation/cifar10_c.yaml"), "--set", assignment])

    def test_training_curves_bundle_and_timestamp(self):
        with tempfile.TemporaryDirectory(prefix="cv-curves-") as directory:
            run = Path(directory)
            (run / "metrics.jsonl").write_text(
                '{"epoch": 1, "train": {"loss": 2.0, "accuracy": 0.1}, '
                '"test": {"loss": 1.9, "accuracy": 0.2}, "lr": 0.1}\n',
                encoding="utf-8",
            )
            self.assertEqual(metrics_to_rows(run / "metrics.jsonl")[0]["epoch"], 1)
            write_bundle(run, render=False)
            self.assertTrue((run / "data/training_curves.csv").is_file())
            self.assertTrue((run / "configs/training_curves.json").is_file())
            self.assertFalse((run / "configs/images").exists())
            fresh = timestamped_run_dir(run / "experiment")
            self.assertRegex(fresh.name, r"^experiment_\d{8}-\d{6}$")


if __name__ == "__main__":
    unittest.main()
