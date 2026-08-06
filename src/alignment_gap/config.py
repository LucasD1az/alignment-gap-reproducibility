"""Configuration and repository path helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "analysis.yml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge a YAML override into a base configuration."""
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml_with_extends(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    seen = set() if seen is None else set(seen)
    if resolved in seen:
        chain = " -> ".join(str(item) for item in [*seen, resolved])
        raise ValueError(f"Circular configuration inheritance: {chain}")
    seen.add(resolved)

    with resolved.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {resolved}")

    parent = config.pop("extends", None)
    if parent is None:
        return config
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    base = _load_yaml_with_extends(parent_path, seen)
    return _deep_merge(base, config)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = _load_yaml_with_extends(config_path)
    config["_config_path"] = str(config_path.resolve())
    config["_repo_root"] = str(REPO_ROOT.resolve())
    return config


def repo_path(config: dict[str, Any], key: str) -> Path:
    raw = config["paths"][key]
    path = Path(raw)
    if not path.is_absolute():
        path = Path(config["_repo_root"]) / path
    return path


def year_period(config: dict[str, Any], year: int, end_key: str = "campaign_end") -> tuple[str, str]:
    entry = config["periods"][int(year)] if int(year) in config["periods"] else config["periods"][str(year)]
    return entry["campaign_start"], entry[end_key]
