"""YAML/JSON configuration, CLI overrides and reproducibility helpers."""

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CV_ROOT = PROJECT_ROOT / "CV_Corruption"


def project_path(path):
    """Resolve a repository-relative path (kept for public compatibility)."""
    path = Path(path).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def experiment_path(path):
    """Resolve data/output paths relative to the CV_Corruption project root.

    Legacy values beginning with ``CV_Corruption/`` are accepted so old
    configuration files continue to work while new files stay portable.
    """
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    parts = path.parts
    if parts and parts[0] == CV_ROOT.name:
        path = Path(*parts[1:])
    return (CV_ROOT / path).resolve()


def config_path(path):
    """Find a config from either the repository root or CV_Corruption root."""
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = (Path.cwd() / path, PROJECT_ROOT / path, CV_ROOT / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return (PROJECT_ROOT / path).resolve()


def load_file(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream) if Path(path).suffix.lower() == ".json" else yaml.safe_load(stream)
    if value is None:
        return {}
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("Configuration must contain a mapping with string keys")
    return value


def deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _validate_structure(value, default, name):
    if default is None:
        return
    if isinstance(default, dict):
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a mapping")
        for key, item in value.items():
            if key not in default:
                raise ValueError(f"Unknown configuration field: {name}.{key}")
            _validate_structure(item, default[key], f"{name}.{key}")
    elif isinstance(default, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be true or false (not a string)")
    elif isinstance(default, (list, tuple)):
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{name} must be a list")
    elif isinstance(default, (int, float)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        if isinstance(default, int) and not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    elif isinstance(default, str) and not isinstance(value, str):
        raise ValueError(f"{name} must be a string")


def parse_config(parser, argv=None, *, default_config=None, path_fields=()):
    """Defaults < file < CLI flags < --set; experiment paths use CV_ROOT."""
    parser.add_argument("--config", type=Path, default=default_config, help="YAML/JSON configuration file")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a config field (YAML value); nested fields use dots")
    argv = list(sys.argv[1:] if argv is None else argv)
    if "-h" in argv or "--help" in argv:
        parser.parse_args(argv)
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--config", type=Path, default=default_config)
    selected, _ = selector.parse_known_args(argv)
    actions = {action.dest: action for action in parser._actions if action.dest not in {"help", "config", "set"}}
    defaults = {name: action.default for name, action in actions.items()}
    defaults.update({name: value for name, value in parser._defaults.items() if name not in {"config", "set"}})
    try:
        config = load_file(config_path(selected.config)) if selected.config else {}
        unknown = set(config) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown configuration fields: {', '.join(sorted(unknown))}")
        # A groups mapping selects groups; other nested style mappings merge.
        merged = deep_merge(defaults, {key: value for key, value in config.items() if key != "groups"})
        if "groups" in config:
            merged["groups"] = config["groups"]
        parser.set_defaults(**merged)
        args = parser.parse_args(argv)
        explicit = {name for name, action in actions.items()
                    if any(token.split("=", 1)[0] in action.option_strings for token in argv)}
        provided = set(config) | explicit
        for assignment in args.set:
            if "=" not in assignment:
                raise ValueError("--set expects KEY=VALUE")
            key, raw = assignment.split("=", 1)
            parts = key.split(".")
            if parts[0] not in defaults:
                raise ValueError(f"Unknown configuration field: {key}")
            value = yaml.safe_load(raw)
            if len(parts) == 1:
                setattr(args, key, value)
            else:
                target = getattr(args, parts[0])
                for part in parts[1:-1]:
                    if not isinstance(target, dict) or part not in target:
                        raise ValueError(f"Unknown configuration field: {key}")
                    target = target[part]
                if not isinstance(target, dict) or parts[-1] not in target:
                    raise ValueError(f"Unknown configuration field: {key}")
                target[parts[-1]] = value
            provided.add(parts[0])
        for name, default in defaults.items():
            value = getattr(args, name)
            if name != "groups":
                _validate_structure(value, default, name)
            action = actions.get(name)
            if isinstance(action, argparse.BooleanOptionalAction) and value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be true or false (not a string)")
            if value is None:
                if default is not None:
                    raise ValueError(f"{name} may not be null")
                continue
            if action and action.type:
                sequence = isinstance(action.nargs, int) or action.nargs in {"+", "*"}
                items = value if sequence else [value]
                if sequence and (not isinstance(value, (list, tuple)) or not value):
                    raise ValueError(f"{name} must be a nonempty list")
                if isinstance(action.nargs, int) and len(value) != action.nargs:
                    raise ValueError(f"{name} must contain exactly {action.nargs} values")
                converted = []
                for item in items:
                    if action.type is int and (isinstance(item, bool) or not isinstance(item, int)):
                        raise ValueError(f"{name} must contain integers")
                    if action.type is float and isinstance(item, bool):
                        raise ValueError(f"{name} must contain numbers")
                    result = action.type(item)
                    if isinstance(result, float) and not math.isfinite(result):
                        raise ValueError(f"{name} must be finite")
                    if action.choices and result not in action.choices:
                        raise ValueError(f"Invalid {name}: {result}; choose from {action.choices}")
                    converted.append(result)
                setattr(args, name, converted if sequence else converted[0])
        for name in path_fields:
            value = getattr(args, name)
            if value is not None:
                setattr(args, name, experiment_path(value))
        args.config = config_path(args.config) if args.config else None
        args._provided = provided
        return args
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        parser.error(str(exc))


def resolved_config(args):
    return {key: value for key, value in vars(args).items()
            if not key.startswith("_") and key not in {"config", "set"}}


def save_config(path, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(resolved_config(args), stream, indent=2, default=str)
        stream.write("\n")
