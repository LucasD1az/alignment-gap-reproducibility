"""Preflight inventory for the private files required by the public pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import repo_path
from .io import find_existing


def check_private_inputs(config: dict[str, Any]) -> list[dict[str, Any]]:
    temp = repo_path(config, "temp")
    checks: list[dict[str, Any]] = []

    def add(name: str, patterns: list[str], *, year: int | None = None, required: bool = True) -> None:
        values = {"year": year} if year is not None else {}
        path = find_existing(temp, patterns, **values)
        checks.append(
            {
                "input": name,
                "year": year,
                "required": required,
                "status": "found" if path else "missing",
                "path": str(path.relative_to(temp)) if path else None,
                "expected": [pattern.format(**values) for pattern in patterns],
            }
        )

    patterns = config["input_patterns"]
    for year in (2016, 2020, 2024):
        add("raw_posts", patterns["raw_posts"], year=year)
        add("first_pass", patterns["first_pass"], year=year)
        add("stance", patterns["stance"], year=year)
        add("democracy_subtopics", patterns["democracy_subtopics"], year=year)
        add("democratic_concerns_stance", patterns["democratic_concerns_stance"], year=year)
        if year in (2020, 2024):
            add("candidate_stance", patterns["candidate_stance"], year=year)
            add("page_mapping", patterns["page_mapping"], year=year, required=False)
            add("region_exposure", patterns["region_exposure"], year=year)
            add("election_results", patterns["election_results"], year=year)

    for key, year, candidate in (
        ("biden_2020", 2020, "Joe Biden"),
        ("trump_2020", 2020, "Donald Trump"),
        ("harris_2024", 2024, "Kamala Harris"),
        ("trump_2024", 2024, "Donald Trump"),
    ):
        add(f"speech:{candidate}", patterns["speeches"][key], year=year)
        if "speech_stance" in patterns:
            add(
                f"speech_stance:{candidate}",
                patterns["speech_stance"][key],
                year=year,
                required=False,
            )
    return checks


def missing_required_inputs(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in check_private_inputs(config) if row["required"] and row["status"] == "missing"]
