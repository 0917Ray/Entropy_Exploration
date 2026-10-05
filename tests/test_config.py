"""Run with python -m unittest discover -s CV_Corruption/tests."""

import argparse
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from cv_corruption.config.loader import PROJECT_ROOT, parse_config, save_config


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cv-config-test-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "experiment.yaml"

    def parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--epochs", type=int, default=200)
        parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0])
        parser.add_argument("--figsize", type=float, nargs=2, default=[6, 4])
        parser.add_argument("--output", type=Path)
        parser.set_defaults(style={"font_size": 12, "color": "red"}, groups={"noise": ["shot_noise"]})
        return parser

    def config(self, content):
        self.path.write_text(content, encoding="utf-8")
        return ["--config", str(self.path)]

    def test_priority_and_nested_merge(self):
        argv = self.config("epochs: 20\nstyle:\n  font_size: 16\ngpu_ids: [1, 2]\n")
        args = parse_config(self.parser(), argv + ["--epochs", "10", "--no-amp", "--set", "style.color=blue"])
        self.assertEqual(args.epochs, 10)
        self.assertFalse(args.amp)
        self.assertEqual(args.style, {"font_size": 16, "color": "blue"})
        self.assertEqual(args.gpu_ids, [1, 2])
        self.assertIn("epochs", args._provided)

    def test_set_wins_over_flags(self):
        args = parse_config(self.parser(), ["--epochs", "10", "--set", "epochs=5"])
        self.assertEqual(args.epochs, 5)

    def test_help_does_not_require_config(self):
        missing = str(Path(self.temp.name) / "missing.yaml")
        with contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as result:
                parse_config(self.parser(), ["--config", missing, "--help"])
        self.assertEqual(result.exception.code, 0)
        self.assertIn("--config", output.getvalue())

    def test_fixed_length_config_sequences(self):
        for values in ("[6]", "[6, 4, 2]"):
            with self.subTest(values=values), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_config(self.parser(), self.config(f"figsize: {values}\n"))

    def test_groups_select_instead_of_merge(self):
        args = parse_config(self.parser(), self.config("groups:\n  digital: [contrast]\n"))
        self.assertEqual(args.groups, {"digital": ["contrast"]})

    def test_paths_independent_of_cwd(self):
        argv = self.config("output: CV_Corruption/runs/example\n")
        old = Path.cwd()
        try:
            os.chdir(self.temp.name)
            args = parse_config(self.parser(), argv, path_fields=("output",))
        finally:
            os.chdir(old)
        self.assertEqual(args.output, PROJECT_ROOT / "CV_Corruption/runs/example")

    def test_json_round_trip(self):
        args = parse_config(self.parser(), ["--epochs", "7", "--output", "CV_Corruption/runs/example"],
                            path_fields=("output",))
        path = Path(self.temp.name) / "resolved.json"
        save_config(path, args)
        loaded = parse_config(self.parser(), ["--config", str(path)], path_fields=("output",))
        self.assertEqual(loaded.epochs, 7)
        self.assertEqual(loaded.output, args.output)
        self.assertNotIn("_provided", json.loads(path.read_text()))

    def test_bad_config_fails(self):
        invalid = ["epochs: 1.5\n", "amp: 'false'\n", "gpu_ids: [false]\n",
                   "gpu_ids: []\n", "style:\n  typo: 2\n", "unknown: 2\n", "- not-a-mapping\n"]
        for content in invalid:
            with self.subTest(content=content), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_config(self.parser(), self.config(content))


if __name__ == "__main__":
    unittest.main()
