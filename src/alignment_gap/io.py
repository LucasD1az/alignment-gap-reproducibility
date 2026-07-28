"""Small, explicit I/O helpers for public and temporary data."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {source}:{line_number}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {source}:{line_number}")
            yield obj


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> Path:
    target = ensure_parent(path)
    opener = gzip.open if target.suffix == ".gz" else open
    with opener(target, "wt", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    return target


def read_public_year(posts_path: str | Path, labels_path: str | Path) -> pd.DataFrame:
    posts = pd.read_csv(posts_path, dtype={"post_id": "string", "page_id": "string"})
    labels = pd.DataFrame(read_jsonl(labels_path))
    if labels.empty:
        labels = pd.DataFrame(columns=["post_id", "topic", "stance", "candidate_stance"])
    labels["post_id"] = labels["post_id"].astype("string")
    merged = posts.merge(labels, on="post_id", how="left", validate="one_to_one")
    merged["creation_time"] = pd.to_datetime(merged["creation_time"], utc=True, errors="coerce")
    for col in ("like_count", "reaction_count"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    return merged


def find_existing(base_dir: str | Path, patterns: Iterable[str], **format_values: object) -> Path | None:
    base = Path(base_dir)
    for pattern in patterns:
        candidate = base / pattern.format(**format_values)
        if candidate.exists():
            return candidate
    return None


def require_existing(base_dir: str | Path, patterns: Iterable[str], **format_values: object) -> Path:
    result = find_existing(base_dir, patterns, **format_values)
    if result is None:
        rendered = [str(Path(base_dir) / p.format(**format_values)) for p in patterns]
        raise FileNotFoundError("None of the expected files exists:\n  - " + "\n  - ".join(rendered))
    return result


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(obj: Any, path: str | Path, *, indent: int = 2) -> Path:
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")
    return target
