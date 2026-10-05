"""Regression coverage for configurable bar colors and portable replotting."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cv_corruption.cli import plot_corruption_counts as plot


class BarConfigTests(unittest.TestCase):
    def colors(self, mode, highlight=None):
        config = plot.deep_merge(plot.DEFAULT_CONFIG, {
            "font": {"sans_serif": ["DejaVu Sans"]},
            "bars": {"color_mode": mode},
        })
        fig = plot.plot_single_bar(["Baseline", "Ours"], plot.np.array([1.0, 2.0]), None,
                                   config, highlight)
        self.addCleanup(plot.plt.close, fig)
        return config, [bar.get_facecolor() for bar in fig.axes[0].patches], [
            bar.get_edgecolor() for bar in fig.axes[0].patches]

    def test_gray_mode(self):
        config, faces, edges = self.colors("gray", "Ours")
        self.assertEqual(faces, [plot.rgba(config["colors"]["bar_gray_face"], 0.5)] * 2)
        self.assertEqual(edges, [plot.rgba(config["colors"]["bar_gray_edge"], 0.95)] * 2)

    def test_highlight_mode(self):
        config, faces, _ = self.colors("highlight", "Ours")
        self.assertEqual(faces, [plot.rgba(config["colors"]["bar_gray_face"], 0.5),
                                 plot.rgba(config["bars"]["highlight_color"], 0.5)])

    def test_palette_mode(self):
        config, faces, _ = self.colors("palette")
        self.assertEqual(faces, [plot.rgba(color, 0.5) for color in config["bars"]["palette"][:2]])

    def test_invalid_color_mode(self):
        config = plot.deep_merge(plot.DEFAULT_CONFIG, {"bars": {"color_mode": "typo"}})
        with self.assertRaisesRegex(ValueError, "color_mode"):
            plot.plot_single_bar(["A"], plot.np.array([1.0]), None, config)

    def test_empty_palette(self):
        config = plot.deep_merge(plot.DEFAULT_CONFIG, {
            "font": {"sans_serif": ["DejaVu Sans"]},
            "bars": {"color_mode": "palette", "palette": []},
        })
        with self.assertRaisesRegex(ValueError, "palette"):
            plot.plot_single_bar(["A"], plot.np.array([1.0]), None, config)

    def test_portable_bundle(self):
        with tempfile.TemporaryDirectory(prefix="cv-bar-bundle-") as directory:
            root = Path(directory)
            output = root / "counts.png"
            config = plot.deep_merge(plot.DEFAULT_CONFIG, {
                "data": {"mode": "single", "label_column": "category", "value_column": "value"},
                "bars": {"color_mode": "gray"},
                "font": {"sans_serif": ["DejaVu Sans"]},
                "output": {"dpi": 50, "pdf": True},
            })
            plot.write_bundle(output, config, [{"category": "A", "value": "1.23456789"}],
                              ["category", "value"])
            saved = json.loads((root / "counts_config.json").read_text())
            self.assertEqual(saved["data"]["input"], "counts_data.csv")
            self.assertEqual(saved["output"]["filename"], "counts.png")
            self.assertTrue(saved["output"]["transparent"])
            result = subprocess.run([sys.executable, str(root / "counts_plot.py"),
                                     "--config", str(root / "counts_config.json")],
                                    cwd="/tmp", capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".pdf").is_file())
            self.assertIn("1.23456789", (root / "counts_data.csv").read_text())


if __name__ == "__main__":
    unittest.main()
