"""Integrity checks for the public release datasets."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import repo_path
from .constants import YEARS
from .io import read_jsonl


FORBIDDEN_PUBLIC_COLUMNS = {
    "text",
    "post_owner.name",
    "post_owner.username",
    "link",
    "url",
    "caption",
    "description",
    "username",
    "name",
}


def validate_public_data(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for year in YEARS:
        posts_path = repo_path(config, "posts") / f"posts_{year}.csv.gz"
        labels_path = repo_path(config, "labels") / f"labels_{year}.jsonl.gz"
        if not posts_path.exists():
            errors.append(f"Missing {posts_path}")
            continue
        if not labels_path.exists():
            errors.append(f"Missing {labels_path}")
            continue
        posts = pd.read_csv(posts_path, dtype="string")
        labels = pd.DataFrame(read_jsonl(labels_path))
        required_posts = {"post_id", "page_id", "creation_time", "like_count", "reaction_count"}
        required_labels = {"post_id", "topic", "stance", "candidate_stance"}
        if set(posts.columns) != required_posts:
            errors.append(f"{posts_path.name}: columns are {posts.columns.tolist()}, expected {sorted(required_posts)}")
        if set(labels.columns) != required_labels:
            errors.append(f"{labels_path.name}: columns are {labels.columns.tolist()}, expected {sorted(required_labels)}")
        if posts["post_id"].duplicated().any():
            errors.append(f"{posts_path.name}: duplicate post_id values")
        if not labels.empty and labels["post_id"].duplicated().any():
            errors.append(f"{labels_path.name}: duplicate post_id values")
        if set(posts["post_id"].dropna()) != set(labels["post_id"].dropna()):
            errors.append(f"{year}: posts and labels do not contain the same post IDs")
        forbidden = FORBIDDEN_PUBLIC_COLUMNS & set(posts.columns) | FORBIDDEN_PUBLIC_COLUMNS & set(labels.columns)
        if forbidden:
            errors.append(f"{year}: forbidden public fields found: {sorted(forbidden)}")
        if not posts["post_id"].str.startswith(f"post{year}_", na=False).all():
            errors.append(f"{posts_path.name}: post IDs do not look anonymized")
        if not posts["page_id"].str.startswith("page_", na=False).all():
            errors.append(f"{posts_path.name}: page IDs do not look anonymized")

    for year, candidates in (
        (2020, ("Joe Biden", "Donald Trump")),
        (2024, ("Kamala Harris", "Donald Trump")),
    ):
        for candidate in candidates:
            stance_path = repo_path(config, "speeches") / f"stance_counts_{candidate.casefold().replace(' ', '_')}_{year}.csv.gz"
            if not stance_path.exists():
                errors.append(f"Missing {stance_path}")
                continue
            stance = pd.read_csv(stance_path)
            expected = {"date", "candidate", "topic", "stance", "paragraph_count"}
            if set(stance.columns) != expected:
                errors.append(f"{stance_path.name}: columns are {stance.columns.tolist()}, expected {sorted(expected)}")
            if "paragraph_count" in stance.columns and (pd.to_numeric(stance["paragraph_count"], errors="coerce").fillna(-1) < 0).any():
                errors.append(f"{stance_path.name}: negative paragraph counts")

    for year in (2020, 2024):
        exposure_path = repo_path(config, "geography") / f"page_state_exposure_{year}.csv.gz"
        if exposure_path.exists():
            exposure = pd.read_csv(exposure_path)
            expected = {"page_id", "state", "state_abbr", "impression_value", "impression_share"}
            if set(exposure.columns) != expected:
                errors.append(
                    f"{exposure_path.name}: columns are {exposure.columns.tolist()}, expected {sorted(expected)}"
                )
                continue
            sums = exposure.groupby("page_id")["impression_share"].sum()
            if not sums.between(0.999999, 1.000001).all():
                errors.append(f"{exposure_path.name}: exposure shares do not sum to one for every page")
            if (pd.to_numeric(exposure["impression_value"], errors="coerce").fillna(-1) < 0).any():
                errors.append(f"{exposure_path.name}: negative or invalid impression values")
    return errors


def assert_public_data(config: dict[str, Any]) -> None:
    errors = validate_public_data(config)
    if errors:
        raise ValueError("Public-data validation failed:\n- " + "\n- ".join(errors))
