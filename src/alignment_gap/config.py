"""Configuration and repository path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "analysis.yml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
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
