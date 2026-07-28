"""Transform private temporary inputs into the minimal public datasets."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .anonymize import Anonymizer, load_or_create_salt, normalize_raw_id
from .config import repo_path
from .constants import (
    CANDIDATE_STANCE_ALIASES,
    DEMOCRACY_MACRO_LABELS,
    DEMOCRACY_SUBTOPIC_MAP,
    STANCE_PRO_ANTI,
    STATE_ABBR_TO_NAME,
    canonical_candidate_stance,
    canonical_stance,
    canonical_topic,
    state_abbr,
)
from .io import find_existing, iter_jsonl, require_existing, sha256_file, write_json, write_jsonl

POST_COLUMN_ALIASES = {
    "post_id": ("id", "post_id"),
    # Keep both page identifiers when available.  The original geographic
    # notebooks joined ads to posts through post_owner.username, while other
    # datasets use post_owner.id.  Selecting only the first one silently breaks
    # the page mapping whenever both columns are present.
    "page_id_key": ("post_owner.id", "page_id", "owner_id"),
    "page_username_key": ("post_owner.username", "username"),
    "creation_time": ("creation_time", "created_time", "date", "timestamp"),
    "like_count": ("statistics.like_count", "like_count", "likes"),
    "reaction_count": ("statistics.reaction_count", "reaction_count", "reactions"),
}


def _pick_column(columns: Iterable[str], aliases: Iterable[str], *, required: bool = True) -> str | None:
    available = {str(c).strip().casefold(): str(c) for c in columns}
    for alias in aliases:
        if alias.casefold() in available:
            return available[alias.casefold()]
    if required:
        raise KeyError(f"None of the expected columns {list(aliases)} exists. Available: {list(columns)}")
    return None


def _read_csv_selected(path: Path, alias_groups: dict[str, tuple[str, ...]]) -> tuple[pd.DataFrame, dict[str, str | None]]:
    header = pd.read_csv(path, nrows=0)
    mapping: dict[str, str | None] = {}
    selected: list[str] = []
    optional_keys = {"reaction_count", "page_id_key", "page_username_key"}
    for key, aliases in alias_groups.items():
        required = key not in optional_keys
        col = _pick_column(header.columns, aliases, required=required)
        mapping[key] = col
        if col is not None:
            selected.append(col)
    df = pd.read_csv(path, usecols=selected, low_memory=False)
    return df, mapping


def _records_by_id(path: Path | None, *, value_keys: tuple[str, ...]) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    out: dict[str, Any] = {}
    for obj in iter_jsonl(path):
        raw_id = obj.get("id", obj.get("post_id", obj.get("p_id")))
        if raw_id is None:
            continue
        value = None
        for key in value_keys:
            if key in obj and obj[key] is not None:
                value = obj[key]
                break
        if value is not None:
            normalized_id = normalize_raw_id(raw_id)
        if normalized_id:
            out[normalized_id] = value
    return out


def _load_page_mapping(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path, dtype="string")
    known_headers = {
        "ad_page_id", "page_id", "mcl_page_id", "post_page_id",
        "post_owner.id", "post_owner.username", "username", "owner_id", "page_key",
    }
    # Legacy `{year}_ids.csv` files are headerless: username,page_id.
    if not any(str(column).casefold() in known_headers for column in df.columns):
        df = pd.read_csv(path, header=None, names=["username", "page_id"], dtype="string")
    if df.shape[1] < 2:
        raise ValueError(f"Page mapping must contain at least two columns: {path}")

    right = _pick_column(df.columns, ("ad_page_id", "page_id", "mcl_page_id"), required=False)
    if right is None:
        right = str(df.columns[-1])
    left_candidates = [c for c in df.columns if c != right]
    left = _pick_column(
        left_candidates,
        ("post_page_id", "post_owner.id", "post_owner.username", "username", "owner_id", "page_key"),
        required=False,
    )
    if left is None:
        left = str(left_candidates[0])

    work = df[[left, right]].dropna().copy()
    work[left] = work[left].map(normalize_raw_id)
    work[right] = work[right].map(normalize_raw_id)
    work = work[(work[left] != "") & (work[right] != "")]
    return dict(zip(work[left], work[right]))


def _candidate_stance_by_id(path: Path | None, config: dict[str, Any]) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path, low_memory=False)
    id_col = _pick_column(df.columns, ("p_id", "id", "post_id"))
    class_col = _pick_column(df.columns, ("class", "label", "candidate_stance"))
    conf_col = _pick_column(
        df.columns,
        ("conf", "confidence", "score", "prob", "probability", "max_prob", "pred_conf"),
        required=False,
    )
    threshold = config["preparation"].get("candidate_stance_confidence_threshold")
    strict = bool(config["preparation"].get("strict_candidate_confidence", False))

    work = df[[id_col, class_col] + ([conf_col] if conf_col else [])].copy()
    work[id_col] = work[id_col].map(normalize_raw_id)
    work[class_col] = work[class_col].map(canonical_candidate_stance)
    work = work[work[id_col] != ""].copy()
    if conf_col:
        work[conf_col] = pd.to_numeric(work[conf_col], errors="coerce")
        # Some classifier exports contain repeated IDs. Keep the most confident prediction.
        work = work.sort_values(conf_col, ascending=False, na_position="last").drop_duplicates(id_col, keep="first")
        if threshold is not None:
            low = work[conf_col].isna() | (work[conf_col] < float(threshold))
            work.loc[low, class_col] = "Neither"
    elif strict and threshold is not None:
        raise ValueError(f"Confidence threshold requested but no confidence column found in {path}")
    else:
        work = work.drop_duplicates(id_col, keep="last")

    valid = set(CANDIDATE_STANCE_ALIASES.values())
    work.loc[~work[class_col].isin(valid), class_col] = "Neither"
    return dict(zip(work[id_col], work[class_col]))


def _resolve_input(config: dict[str, Any], key: str, *, year: int) -> Path | None:
    temp = repo_path(config, "temp")
    patterns = config["input_patterns"][key]
    return find_existing(temp, patterns, year=year)


def _public_topic(raw_topic: object, democracy_subtopic: object | None = None) -> str:
    topic = canonical_topic(raw_topic)
    if topic in DEMOCRACY_MACRO_LABELS:
        if democracy_subtopic is not None:
            sub = str(democracy_subtopic).strip()
            mapped = DEMOCRACY_SUBTOPIC_MAP.get(sub, canonical_topic(sub))
            if mapped != "Not specified":
                return mapped
        # A direct final Democratic-concerns label is already public-ready.
        if topic == "Democratic concerns":
            return topic
        # The broad legacy macro-label is not one of the manuscript's final topics.
        return "Not specified"
    return topic


def prepare_posts_and_labels(config: dict[str, Any], year: int, anonymizer: Anonymizer) -> dict[str, Any]:
    temp = repo_path(config, "temp")
    raw_path = require_existing(temp, config["input_patterns"]["raw_posts"], year=year)
    first_path = require_existing(temp, config["input_patterns"]["first_pass"], year=year)
    stance_path = _resolve_input(config, "stance", year=year)
    democracy_path = _resolve_input(config, "democracy_subtopics", year=year)
    parties_stance_path = _resolve_input(config, "democratic_concerns_stance", year=year)
    candidate_path = _resolve_input(config, "candidate_stance", year=year)
    page_mapping_path = _resolve_input(config, "page_mapping", year=year)

    raw, cols = _read_csv_selected(raw_path, POST_COLUMN_ALIASES)
    rename = {source: target for target, source in cols.items() if source is not None}
    raw = raw.rename(columns=rename)
    if "reaction_count" not in raw.columns:
        raw["reaction_count"] = 0
    if "page_id_key" not in raw.columns:
        raw["page_id_key"] = None
    if "page_username_key" not in raw.columns:
        raw["page_username_key"] = None
    if raw["page_id_key"].isna().all() and raw["page_username_key"].isna().all():
        raise KeyError(
            "No page identifier found in the posts file. Expected one of "
            "post_owner.id/page_id/owner_id or post_owner.username/username."
        )

    raw["raw_post_id"] = raw["post_id"].map(normalize_raw_id)
    raw["raw_page_id_key"] = raw["page_id_key"].map(normalize_raw_id)
    raw["raw_page_username_key"] = raw["page_username_key"].map(normalize_raw_id)
    raw["creation_time"] = pd.to_datetime(raw["creation_time"], utc=True, errors="coerce")
    raw["like_count"] = pd.to_numeric(raw["like_count"], errors="coerce").fillna(0).clip(lower=0).round().astype("int64")
    raw["reaction_count"] = pd.to_numeric(raw["reaction_count"], errors="coerce").fillna(0).clip(lower=0).round().astype("int64")
    raw = raw.dropna(subset=["creation_time"]).copy()
    has_page = raw["raw_page_id_key"].ne("") | raw["raw_page_username_key"].ne("")
    raw = raw[(raw["raw_post_id"] != "") & has_page].copy()
    raw = raw.drop_duplicates("raw_post_id", keep="last")

    page_mapping = _load_page_mapping(page_mapping_path)

    def resolve_page(row: pd.Series) -> tuple[str, str, bool]:
        # Explicit mappings take precedence.  Try both identifiers because the
        # legacy YYYY_ids.csv files map username -> ads page_id, whereas newer
        # mapping tables often map post_owner.id -> ads page_id.
        candidates = [row["raw_page_id_key"], row["raw_page_username_key"]]
        matched = [(candidate, page_mapping[candidate]) for candidate in candidates if candidate and candidate in page_mapping]
        mapped_values = {value for _, value in matched}
        if len(mapped_values) > 1:
            raise ValueError(
                "Ambiguous page mapping: post_owner.id and post_owner.username "
                f"map to different ads pages for post {row['raw_post_id']}."
            )
        if matched:
            source_key, canonical = matched[0]
            return source_key, canonical, True

        fallback = row["raw_page_id_key"] or row["raw_page_username_key"]
        return fallback, fallback, False

    resolved_pages = raw.apply(resolve_page, axis=1, result_type="expand")
    resolved_pages.columns = ["raw_page_key", "canonical_raw_page_id", "page_mapping_matched"]
    raw = pd.concat([raw, resolved_pages], axis=1)
    raw["post_id"] = raw["raw_post_id"].map(lambda value: anonymizer.identifier(f"post{year}", value))
    raw["page_id"] = raw["canonical_raw_page_id"].map(lambda value: anonymizer.identifier("page", value))

    first_topics = _records_by_id(first_path, value_keys=("label", "topic"))
    democracy_subtopics = _records_by_id(democracy_path, value_keys=("label", "subtopic", "democracy_subtopic"))
    general_stances = _records_by_id(stance_path, value_keys=("stance", "label"))
    parties_stances = _records_by_id(parties_stance_path, value_keys=("stance", "label"))
    candidate_stances = _candidate_stance_by_id(candidate_path, config)

    labels_rows: list[dict[str, Any]] = []
    for row in raw[["raw_post_id", "post_id"]].itertuples(index=False):
        raw_topic = first_topics.get(row.raw_post_id, "Not specified")
        topic = _public_topic(raw_topic, democracy_subtopics.get(row.raw_post_id))
        if topic == "Democratic concerns":
            stance = canonical_stance(parties_stances.get(row.raw_post_id))
        elif topic in STANCE_PRO_ANTI:
            stance = canonical_stance(general_stances.get(row.raw_post_id))
        else:
            stance = None
        labels_rows.append(
            {
                "post_id": row.post_id,
                "topic": topic,
                "stance": stance,
                "candidate_stance": candidate_stances.get(row.raw_post_id),
            }
        )

    posts_out = repo_path(config, "posts") / f"posts_{year}.csv.gz"
    labels_out = repo_path(config, "labels") / f"labels_{year}.jsonl.gz"
    posts_out.parent.mkdir(parents=True, exist_ok=True)
    labels_out.parent.mkdir(parents=True, exist_ok=True)

    public_posts = raw[["post_id", "page_id", "creation_time", "like_count", "reaction_count"]].copy()
    public_posts["creation_time"] = public_posts["creation_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    public_posts = public_posts.sort_values("post_id")
    public_posts.to_csv(posts_out, index=False, compression="gzip")
    write_jsonl(sorted(labels_rows, key=lambda item: item["post_id"]), labels_out)

    if config["privacy"].get("save_private_id_maps", True):
        private_dir = temp / "private_maps"
        private_dir.mkdir(parents=True, exist_ok=True)
        raw[[
            "raw_post_id",
            "raw_page_id_key",
            "raw_page_username_key",
            "raw_page_key",
            "canonical_raw_page_id",
            "page_mapping_matched",
            "post_id",
            "page_id",
        ]].to_csv(
            private_dir / f"id_map_{year}.csv.gz", index=False, compression="gzip"
        )

    return {
        "year": year,
        "posts_path": str(posts_out.relative_to(Path(config["_repo_root"]))),
        "labels_path": str(labels_out.relative_to(Path(config["_repo_root"]))),
        "n_posts": int(len(public_posts)),
        "n_pages": int(public_posts["page_id"].nunique()),
        "n_posts_with_page_mapping": int(raw["page_mapping_matched"].sum()),
        "n_raw_pages_matched": int(raw.loc[raw["page_mapping_matched"], "raw_page_key"].nunique()),
        "n_topic_labels": int(sum(item["topic"] != "Not specified" for item in labels_rows)),
        "n_stance_labels": int(sum(item["stance"] is not None for item in labels_rows)),
        "n_candidate_stance_labels": int(sum(item["candidate_stance"] is not None for item in labels_rows)),
    }


def _speech_column(df: pd.DataFrame, aliases: tuple[str, ...], required: bool = True) -> str | None:
    return _pick_column(df.columns, aliases, required=required)


def prepare_speech_file(
    config: dict[str, Any],
    key: str,
    year: int,
    candidate: str,
    anonymizer: Anonymizer,
) -> dict[str, Any]:
    temp = repo_path(config, "temp")
    patterns = config["input_patterns"]["speeches"][key]
    path = require_existing(temp, patterns)
    df = pd.read_pickle(path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Speech pickle must contain a DataFrame: {path}")

    speech_col = _speech_column(df, ("speech_id", "Speech_id", "speech", "id_speech"))
    paragraph_col = _speech_column(df, ("paragraph_id", "Paragraph_id", "id", "paragraph_index"), required=False)
    date_col = _speech_column(df, ("date", "Date", "speech_date", "creation_time", "datetime"))
    # In the original speech-stance exports, ``label`` is the stance column
    # while ``topic`` stores the topic.  Older topic-only pickles sometimes use
    # ``label`` for the topic instead, so only use it as a topic fallback.
    topic_col = _speech_column(df, ("topic", "Topic", "ollama_topic"), required=False)
    if topic_col is None:
        topic_col = _speech_column(df, ("label",))
    subtopic_col = _speech_column(
        df, ("democracy_subtopic", "subtopic", "Subtopic", "ollama_subtopic"), required=False
    )
    stance_col = _speech_column(
        df, ("stance", "Stance", "ollama_stance", "stance_label", "ollama_label"), required=False
    )
    if stance_col is None and "label" in df.columns and topic_col != "label":
        stance_col = "label"
    subtopics = df[subtopic_col] if subtopic_col else pd.Series([None] * len(df), index=df.index)

    work = pd.DataFrame(
        {
            "raw_speech_id": df[speech_col].map(normalize_raw_id),
            "date": pd.to_datetime(df[date_col], utc=True, errors="coerce"),
            "topic": [
                _public_topic(topic, subtopic)
                for topic, subtopic in zip(df[topic_col], subtopics)
            ],
            "stance": df[stance_col].map(canonical_stance) if stance_col else None,
        }
    )
    if paragraph_col:
        work["raw_paragraph_id"] = df[paragraph_col].map(normalize_raw_id)
    else:
        work["raw_paragraph_id"] = work.groupby("raw_speech_id").cumcount().astype(str)
    work = work.dropna(subset=["date"]).copy()
    work = work[(work["raw_speech_id"] != "") & (work["raw_paragraph_id"] != "")].copy()
    work.loc[~work["topic"].isin(STANCE_PRO_ANTI), "stance"] = None
    work["speech_id"] = work["raw_speech_id"].map(
        lambda value: anonymizer.identifier(f"speech_{year}_{candidate.casefold().replace(' ', '_')}", value)
    )
    work["paragraph_id"] = [
        anonymizer.identifier("paragraph", f"{candidate}:{year}:{sid}:{pid}")
        for sid, pid in zip(work["raw_speech_id"], work["raw_paragraph_id"])
    ]
    work["candidate"] = candidate
    work["date"] = work["date"].dt.strftime("%Y-%m-%d")

    out_path = repo_path(config, "speeches") / f"speeches_{candidate.casefold().replace(' ', '_')}_{year}.csv.gz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    public = work[["speech_id", "paragraph_id", "candidate", "date", "topic", "stance"]].sort_values(
        ["date", "speech_id", "paragraph_id"]
    )
    public.to_csv(out_path, index=False, compression="gzip")
    return {
        "year": year,
        "candidate": candidate,
        "path": str(out_path.relative_to(Path(config["_repo_root"]))),
        "n_speeches": int(public["speech_id"].nunique()),
        "n_paragraphs": int(len(public)),
    }



def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.casefold() in {".pkl", ".pickle"}:
        obj = pd.read_pickle(path)
        if not isinstance(obj, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame in {path}")
        return obj
    return pd.read_csv(path, low_memory=False)


def _speech_stance_long_from_table(df: pd.DataFrame, candidate: str) -> pd.DataFrame:
    """Normalize either notebook-wide or paragraph-long speech stance data.

    The notebook source used files such as
    ``2024_speech_stance_trump_time_series.pkl`` with columns named
    ``Topic - Stance``.  It also created paragraph-level CSVs with columns
    ``Date``, ``topic`` and ``label``.  Both formats are accepted here and
    converted to the public long schema used only for aggregate figure colors.
    """
    date_col = _speech_column(df, ("date", "Date", "speech_date", "creation_time", "datetime"), required=False)
    topic_col = _speech_column(df, ("topic", "Topic", "ollama_topic"), required=False)
    stance_col = _speech_column(
        df,
        ("stance", "Stance", "ollama_stance", "stance_label", "ollama_label", "label"),
        required=False,
    )
    count_col = _speech_column(
        df,
        ("paragraph_count", "n_paragraphs", "count", "n", "value", "weight"),
        required=False,
    )

    rows: list[dict[str, Any]] = []
    if topic_col is not None and stance_col is not None:
        dates = pd.to_datetime(df[date_col], utc=True, errors="coerce") if date_col else pd.Series(pd.NaT, index=df.index)
        counts = pd.to_numeric(df[count_col], errors="coerce").fillna(0.0) if count_col else pd.Series(1.0, index=df.index)
        for date, topic, stance, count in zip(dates, df[topic_col], df[stance_col], counts):
            rows.append(
                {
                    "date": date,
                    "candidate": candidate,
                    "topic": canonical_topic(topic),
                    "stance": canonical_stance(stance),
                    "paragraph_count": float(count),
                }
            )
    else:
        if date_col is None:
            raise ValueError(
                "Wide speech stance files must contain a date column and columns named 'Topic - Stance'."
            )
        dates = pd.to_datetime(df[date_col], utc=True, errors="coerce")
        for column in df.columns:
            if column == date_col or " - " not in str(column):
                continue
            topic_raw, stance_raw = str(column).split(" - ", 1)
            values = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
            for date, count in zip(dates, values):
                rows.append(
                    {
                        "date": date,
                        "candidate": candidate,
                        "topic": canonical_topic(topic_raw),
                        "stance": canonical_stance(stance_raw),
                        "paragraph_count": float(count),
                    }
                )

    out = pd.DataFrame(rows, columns=["date", "candidate", "topic", "stance", "paragraph_count"])
    if out.empty:
        return out
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
    out["paragraph_count"] = pd.to_numeric(out["paragraph_count"], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["date", "stance"])
    out = out[out["topic"].isin(STANCE_PRO_ANTI)].copy()
    valid = {
        topic: {mapping["pro"], mapping["anti"], "Neutral"}
        for topic, mapping in STANCE_PRO_ANTI.items()
    }
    out = out[
        [stance in valid.get(topic, set()) for topic, stance in zip(out["topic"], out["stance"])]
    ].copy()
    out = out[out["paragraph_count"] > 0].copy()
    if out.empty:
        return out
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return (
        out.groupby(["date", "candidate", "topic", "stance"], as_index=False)["paragraph_count"]
        .sum()
        .sort_values(["date", "topic", "stance"])
    )


def prepare_speech_stance_counts(
    config: dict[str, Any],
    key: str,
    year: int,
    candidate: str,
) -> dict[str, Any]:
    """Prepare the separate candidate-stance source used in 07_correlations.

    Prefer the original stance time-series/paragraph CSV when supplied.  If it
    is absent, fall back to any stance labels already present in the prepared
    paragraph-level public speech file.
    """
    temp = repo_path(config, "temp")
    patterns = config.get("input_patterns", {}).get("speech_stance", {}).get(key, [])
    source = find_existing(temp, patterns) if patterns else None
    if source is not None:
        long = _speech_stance_long_from_table(_read_table(source), candidate)
        source_kind = "separate_speech_stance_input"
    else:
        speech_path = repo_path(config, "speeches") / f"speeches_{candidate.casefold().replace(' ', '_')}_{year}.csv.gz"
        paragraph_df = pd.read_csv(speech_path)
        long = _speech_stance_long_from_table(paragraph_df, candidate)
        source_kind = "prepared_speech_paragraph_fallback"

    out_path = repo_path(config, "speeches") / f"stance_counts_{candidate.casefold().replace(' ', '_')}_{year}.csv.gz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if long.empty:
        long = pd.DataFrame(columns=["date", "candidate", "topic", "stance", "paragraph_count"])
    long.to_csv(out_path, index=False, compression="gzip")
    return {
        "year": year,
        "candidate": candidate,
        "path": str(out_path.relative_to(Path(config["_repo_root"]))),
        "source_kind": source_kind,
        "n_stance_rows": int(len(long)),
        "n_stance_paragraphs": float(pd.to_numeric(long["paragraph_count"], errors="coerce").fillna(0.0).sum()),
    }

def _parse_share(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    text = str(value).strip().replace(",", "")
    if not text:
        return 0.0
    if text.endswith("%"):
        try:
            return float(text[:-1]) / 100.0
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def prepare_region_exposure(config: dict[str, Any], year: int, anonymizer: Anonymizer) -> dict[str, Any]:
    path = _resolve_input(config, "region_exposure", year=year)
    if path is None:
        return {"year": year, "skipped": True, "reason": "region exposure not found"}
    df = pd.read_csv(path, dtype="string")
    page_col = _pick_column(df.columns, ("page_id", "ad_page_id", "mcl_page_id"))
    state_cols = [c for c in df.columns if c != page_col and state_abbr(c) is not None]
    if not state_cols:
        raise ValueError(f"No U.S. state columns detected in {path}")

    records: list[dict[str, Any]] = []
    for row in df[[page_col] + state_cols].itertuples(index=False, name=None):
        raw_page = normalize_raw_id(row[0])
        if not raw_page:
            continue
        values = np.array([_parse_share(value) for value in row[1:]], dtype=float)
        total = float(values.sum())
        if total <= 0:
            continue
        values /= total
        anon_page = anonymizer.identifier("page", raw_page)
        for column, share in zip(state_cols, values):
            if share <= 0:
                continue
            abbr = state_abbr(column)
            if abbr is None:
                continue
            records.append(
                {
                    "page_id": anon_page,
                    "state": STATE_ABBR_TO_NAME[abbr],
                    "state_abbr": abbr,
                    # geography_v3.ipynb retained both the raw state value and
                    # the row-normalized P(state|page) share.
                    "impression_value": float(share * total),
                    "impression_share": float(share),
                }
            )

    out_path = repo_path(config, "geography") / f"page_state_exposure_{year}.csv.gz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(records).sort_values(["page_id", "state_abbr"])
    out.to_csv(out_path, index=False, compression="gzip")
    return {
        "year": year,
        "path": str(out_path.relative_to(Path(config["_repo_root"]))),
        "n_pages": int(out["page_id"].nunique()) if not out.empty else 0,
        "n_states": int(out["state_abbr"].nunique()) if not out.empty else 0,
    }


def _clean_percent(value: object) -> float:
    if value is None:
        return np.nan
    text = str(value).strip().replace("%", "").replace(",", "")
    return pd.to_numeric(text, errors="coerce")


def _normalize_candidate(value: object) -> str:
    text = str(value).casefold()
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(text.split())


def _parse_election_results_headered(df: pd.DataFrame, year: int) -> pd.DataFrame | None:
    try:
        state_col = _pick_column(df.columns, ("state", "State", "jurisdiction"))
    except KeyError:
        return None
    dem_col = _pick_column(
        df.columns,
        ("democrat_pct", "dem_pct", "democratic_pct", "democratic_percent", "democrat", "democratic"),
        required=False,
    )
    rep_col = _pick_column(
        df.columns,
        ("republican_pct", "rep_pct", "gop_pct", "republican_percent", "republican", "gop"),
        required=False,
    )
    if dem_col is None or rep_col is None:
        return None
    out = pd.DataFrame(
        {
            "state": df[state_col].astype(str),
            "democrat_pct": df[dem_col].map(_clean_percent),
            "republican_pct": df[rep_col].map(_clean_percent),
        }
    )
    return out


def _parse_election_results_wikipedia(path: Path, year: int) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    state_series = df.iloc[:, 0].astype(str)
    first_matches = state_series.str.contains(r"\bAlabama\b", case=False, regex=True, na=False)
    last_matches = state_series.str.contains(r"\bWyoming\b", case=False, regex=True, na=False)
    if first_matches.any() and last_matches.any():
        first = first_matches[first_matches].index.min()
        last = last_matches[last_matches].index.max()
        df = df.loc[first:last].copy()
    if df.shape[1] < 6:
        raise ValueError(f"Election result file is neither a clean table nor the expected Wikipedia export: {path}")

    state = df.iloc[:, 0].astype(str).str.replace(r"\[[^\]]+\]", "", regex=True).str.strip()
    cand_a = df.iloc[:, 1].map(_normalize_candidate)
    pct_a = df.iloc[:, 2].map(_clean_percent)
    cand_b = df.iloc[:, 4].map(_normalize_candidate)
    pct_b = df.iloc[:, 5].map(_clean_percent)

    democrat_names = {2020: ("joe biden", "biden"), 2024: ("kamala harris", "harris", "joe biden", "biden")}[year]
    republican_names = ("donald trump", "trump")

    def matches_candidate(candidate: str, aliases: tuple[str, ...]) -> bool:
        # Wikipedia/CSV exports sometimes append party names, footnotes, or ticket information.
        # Substring matching is therefore safer than exact equality after normalization.
        return any(alias == candidate or alias in candidate for alias in aliases)

    rows = []
    for st, ca, pa, cb, pb in zip(state, cand_a, pct_a, cand_b, pct_b):
        if not np.isfinite(pa) or not np.isfinite(pb):
            continue
        a_dem = matches_candidate(ca, democrat_names)
        b_dem = matches_candidate(cb, democrat_names)
        a_rep = matches_candidate(ca, republican_names)
        b_rep = matches_candidate(cb, republican_names)
        if a_dem and b_rep:
            dem, rep = pa, pb
        elif b_dem and a_rep:
            dem, rep = pb, pa
        else:
            # The source Results tables place the national winner's candidate block first.
            # Therefore the first block is Democratic in 2020 (Biden) and Republican in
            # 2024 (Trump). This fallback is only used when candidate labels cannot be read.
            if year == 2020:
                dem, rep = pa, pb
            elif year == 2024:
                rep, dem = pa, pb
            else:
                raise ValueError(f"Unsupported election year for winner-first fallback: {year}")
        rows.append({"state": st, "democrat_pct": dem, "republican_pct": rep})
    return pd.DataFrame(rows)


def prepare_election_results(config: dict[str, Any], year: int) -> dict[str, Any]:
    path = _resolve_input(config, "election_results", year=year)
    if path is None:
        return {"year": year, "skipped": True, "reason": "election results not found"}
    headered = pd.read_csv(path, low_memory=False)
    out = _parse_election_results_headered(headered, year)
    if out is None:
        out = _parse_election_results_wikipedia(path, year)

    out["state_abbr"] = out["state"].map(state_abbr)
    out = out.dropna(subset=["state_abbr", "democrat_pct", "republican_pct"]).copy()
    out["state"] = out["state_abbr"].map(STATE_ABBR_TO_NAME)
    out["dem_minus_rep"] = out["democrat_pct"] - out["republican_pct"]

    # Guard against a global party-column inversion in winner-first Results exports.
    # These signs are stable in both elections analyzed and make the failure explicit
    # before Figure 5 is reproduced with a reversed electoral axis.
    expected_margin_sign = {"CA": 1, "TX": -1}
    for abbreviation, expected_sign in expected_margin_sign.items():
        check = out.loc[out["state_abbr"] == abbreviation, "dem_minus_rep"]
        if not check.empty and np.sign(float(check.iloc[0])) != expected_sign:
            raise ValueError(
                f"Election result columns appear inverted for {year}: "
                f"expected {'Democratic' if expected_sign > 0 else 'Republican'} advantage in {abbreviation}."
            )

    out = out[["state", "state_abbr", "democrat_pct", "republican_pct", "dem_minus_rep"]]
    out = out.drop_duplicates("state_abbr", keep="last").sort_values("state_abbr")

    out_path = repo_path(config, "geography") / f"election_results_{year}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return {
        "year": year,
        "path": str(out_path.relative_to(Path(config["_repo_root"]))),
        "n_states": int(len(out)),
    }


def prepare_all(config: dict[str, Any]) -> dict[str, Any]:
    temp = repo_path(config, "temp")
    salt = load_or_create_salt(temp)
    anonymizer = Anonymizer(salt=salt, digest_chars=int(config["privacy"].get("id_digest_chars", 16)))

    manifest: dict[str, Any] = {"posts_and_labels": [], "speeches": [], "speech_stance": [], "geography": []}
    for year in (2016, 2020, 2024):
        manifest["posts_and_labels"].append(prepare_posts_and_labels(config, year, anonymizer))

    speech_specs = (
        ("biden_2020", 2020, "Joe Biden"),
        ("trump_2020", 2020, "Donald Trump"),
        ("harris_2024", 2024, "Kamala Harris"),
        ("trump_2024", 2024, "Donald Trump"),
    )
    for key, year, candidate in speech_specs:
        manifest["speeches"].append(prepare_speech_file(config, key, year, candidate, anonymizer))
        manifest["speech_stance"].append(prepare_speech_stance_counts(config, key, year, candidate))

    for year in (2020, 2024):
        manifest["geography"].append(prepare_region_exposure(config, year, anonymizer))
        manifest["geography"].append(prepare_election_results(config, year))

    public_files = []
    for directory_key in ("posts", "labels", "speeches", "geography"):
        for path in repo_path(config, directory_key).glob("*"):
            if path.is_file():
                public_files.append(
                    {
                        "path": str(path.relative_to(Path(config["_repo_root"]))),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    manifest["files"] = sorted(public_files, key=lambda item: item["path"])
    manifest_path = Path(config["_repo_root"]) / "data" / "manifest.json"
    write_json(manifest, manifest_path)
    return manifest
