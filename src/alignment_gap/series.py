"""Build the daily and aggregate signals used by Figures 1–5."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import repo_path, year_period
from .constants import CANDIDATES_BY_YEAR, STANCE_PRO_ANTI, canonical_stance, canonical_topic
from .io import read_public_year


def load_posts_labels(config: dict[str, Any], year: int) -> pd.DataFrame:
    posts = repo_path(config, "posts") / f"posts_{year}.csv.gz"
    labels = repo_path(config, "labels") / f"labels_{year}.jsonl.gz"
    if not posts.exists() or not labels.exists():
        raise FileNotFoundError(f"Prepared public data missing for {year}. Run prepare_public_data.py first.")
    df = read_public_year(posts, labels)
    df["topic"] = df["topic"].fillna("Not specified").map(canonical_topic)
    return df


def load_speeches(config: dict[str, Any], year: int, candidate: str) -> pd.DataFrame:
    path = repo_path(config, "speeches") / f"speeches_{candidate.casefold().replace(' ', '_')}_{year}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Prepared speech data missing: {path}")
    df = pd.read_csv(path, dtype={"speech_id": "string", "paragraph_id": "string"})
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["topic"] = df["topic"].fillna("Not specified").map(canonical_topic)
    if "stance" not in df.columns:
        df["stance"] = None
    else:
        df["stance"] = df["stance"].map(canonical_stance)
    return df.dropna(subset=["date"])




def load_speech_stance_counts(config: dict[str, Any], year: int, candidate: str) -> pd.DataFrame:
    path = repo_path(config, "speeches") / f"stance_counts_{candidate.casefold().replace(' ', '_')}_{year}.csv.gz"
    if not path.exists():
        return pd.DataFrame(columns=["date", "candidate", "topic", "stance", "paragraph_count"])
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["topic"] = df["topic"].fillna("Not specified").map(canonical_topic)
    df["stance"] = df["stance"].map(canonical_stance)
    df["paragraph_count"] = pd.to_numeric(df["paragraph_count"], errors="coerce").fillna(0.0)
    return df.dropna(subset=["date"])

def filter_period(
    df: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    date_col: str,
) -> pd.DataFrame:
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)
    dates = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    return df.loc[(dates >= start_ts) & (dates <= end_ts)].copy()


def _full_daily_index(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True), freq="D")


def centered_rolling_sum(df: pd.DataFrame | pd.Series, days: int) -> pd.DataFrame | pd.Series:
    return df.rolling(int(days), center=True, min_periods=1).sum()


def daily_topic_volume_posts(
    config: dict[str, Any],
    year: int,
    *,
    metric: str = "like_count",
    start: str | None = None,
    end: str | None = None,
    rolling_days: int | None = None,
) -> pd.DataFrame:
    if metric not in {"like_count", "reaction_count", "post_count"}:
        raise ValueError("metric must be like_count, reaction_count, or post_count")
    if start is None or end is None:
        start, end = year_period(config, year)
    df = load_posts_labels(config, year)
    df = filter_period(df, start, end, date_col="creation_time")
    df["date"] = df["creation_time"].dt.floor("D")
    df = df[df["topic"] != "Not specified"].copy()
    if metric == "post_count":
        grouped = df.groupby(["date", "topic"]).size().rename("value")
    else:
        grouped = df.groupby(["date", "topic"])[metric].sum().rename("value")
    wide = grouped.unstack(fill_value=0.0).reindex(_full_daily_index(start, end), fill_value=0.0)
    wide.index.name = "date"
    if rolling_days:
        wide = centered_rolling_sum(wide, rolling_days)
    return wide.astype(float)


def daily_topic_like_ratio_posts(
    config: dict[str, Any],
    year: int,
    *,
    start: str | None = None,
    end: str | None = None,
    rolling_days: int | None = None,
) -> pd.DataFrame:
    """Daily mean likes per post for each topic.

    This reconstructs the public-reaction signal used by the final speech
    correlation notebooks: first compute ``daily likes / daily posts`` for
    each topic, replace days without posts by zero, and only then apply the
    centered rolling mean.  It is intentionally *not* a rolling sum of likes.
    """
    if start is None or end is None:
        start, end = year_period(config, year)

    df = load_posts_labels(config, year)
    df = filter_period(df, start, end, date_col="creation_time")
    df["date"] = df["creation_time"].dt.floor("D")
    df = df[df["topic"] != "Not specified"].copy()

    full_index = _full_daily_index(start, end)
    likes = (
        df.groupby(["date", "topic"])["like_count"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(full_index, fill_value=0.0)
        .astype(float)
    )
    posts = (
        df.groupby(["date", "topic"])
        .size()
        .unstack(fill_value=0.0)
        .reindex(index=full_index, columns=likes.columns, fill_value=0.0)
        .astype(float)
    )

    ratio = likes.divide(posts.replace(0.0, np.nan)).fillna(0.0)
    ratio.index.name = "date"
    if rolling_days:
        days = int(rolling_days)
        # The original loader used pandas' default min_periods=window.
        ratio = ratio.rolling(window=days, center=True, min_periods=days).mean()
    return ratio.astype(float)


def daily_topic_volume_speeches(
    config: dict[str, Any],
    year: int,
    candidate: str,
    *,
    start: str | None = None,
    end: str | None = None,
    rolling_days: int | None = None,
) -> pd.DataFrame:
    if start is None or end is None:
        start, end = year_period(config, year)
    df = load_speeches(config, year, candidate)
    df = filter_period(df, start, end, date_col="date")
    df["date"] = df["date"].dt.floor("D")
    df = df[df["topic"] != "Not specified"].copy()
    wide = (
        df.groupby(["date", "topic"])
        .size()
        .rename("value")
        .unstack(fill_value=0.0)
        .reindex(_full_daily_index(start, end), fill_value=0.0)
    )
    wide.index.name = "date"
    if rolling_days:
        wide = centered_rolling_sum(wide, rolling_days)
    return wide.astype(float)



def notebook_daily_topic_like_ratio_posts(
    config: dict[str, Any],
    year: int,
    *,
    start: str | None = None,
    end: str | None = None,
    rolling_days: int = 7,
) -> pd.DataFrame:
    """Recreate the reaction series loaded by ``07_correlations.ipynb``.

    The original pipeline computed daily likes divided by daily posts, left
    topic-days with no posts as ``NaN``, retained only dates present in the
    source table, and then applied a centered rolling mean with
    ``min_periods=1``. These details materially affect the exported
    correlation matrices.
    """
    if start is None or end is None:
        start, end = year_period(config, year)
    df = load_posts_labels(config, year)
    df = filter_period(df, start, end, date_col="creation_time")
    df["date"] = df["creation_time"].dt.floor("D")
    df = df[df["topic"] != "Not specified"].copy()

    likes = df.groupby(["date", "topic"])["like_count"].sum().unstack(fill_value=0.0).sort_index()
    posts = df.groupby(["date", "topic"]).size().unstack(fill_value=0.0).sort_index()
    columns = sorted(set(likes.columns) | set(posts.columns))
    index = likes.index.union(posts.index).sort_values()
    likes = likes.reindex(index=index, columns=columns, fill_value=0.0).astype(float)
    posts = posts.reindex(index=index, columns=columns, fill_value=0.0).astype(float)
    ratio = likes.divide(posts.replace(0.0, np.nan))
    ratio.index.name = "date"
    return ratio.rolling(int(rolling_days), center=True, min_periods=1).mean().astype(float)


def notebook_daily_topic_volume_speeches(
    config: dict[str, Any],
    year: int,
    candidate: str,
    *,
    start: str | None = None,
    end: str | None = None,
    rolling_days: int = 7,
) -> pd.DataFrame:
    """Recreate the sparse speech series used by ``07_correlations.ipynb``.

    Only dates represented in the speech input are retained; non-speech days
    are not inserted as zero rows. Counts are smoothed with the notebook's
    centered rolling mean and ``min_periods=1``.
    """
    if start is None or end is None:
        start, end = year_period(config, year)
    df = load_speeches(config, year, candidate)
    df = filter_period(df, start, end, date_col="date")
    df["date"] = df["date"].dt.floor("D")
    df = df[df["topic"] != "Not specified"].copy()
    wide = df.groupby(["date", "topic"]).size().rename("value").unstack(fill_value=0.0).sort_index()
    wide.index.name = "date"
    return wide.rolling(int(rolling_days), center=True, min_periods=1).mean().astype(float)

def _stance_counts(
    df: pd.DataFrame,
    *,
    weight_col: str | None,
    date_col: str | None,
    neutral_label: str = "Neutral",
) -> pd.DataFrame:
    work = df.copy()
    work = work[work["topic"].isin(STANCE_PRO_ANTI)].copy()
    if weight_col is None:
        work["_weight"] = 1.0
        weight_col = "_weight"
    else:
        work[weight_col] = pd.to_numeric(work[weight_col], errors="coerce").fillna(0.0)
    group_cols = ([date_col] if date_col else []) + ["topic", "stance"]
    grouped = work.groupby(group_cols, dropna=False)[weight_col].sum().rename("weight").reset_index()

    rows: list[dict[str, Any]] = []
    leading_groups = [date_col, "topic"] if date_col else ["topic"]
    for keys, sub in grouped.groupby(leading_groups, dropna=False):
        if date_col:
            date_value, topic = keys
        else:
            date_value = None
            topic = keys[0] if isinstance(keys, tuple) else keys
        mapping = STANCE_PRO_ANTI[topic]
        pro = float(sub.loc[sub["stance"] == mapping["pro"], "weight"].sum())
        anti = float(sub.loc[sub["stance"] == mapping["anti"], "weight"].sum())
        neutral = float(sub.loc[sub["stance"] == neutral_label, "weight"].sum())
        total = pro + anti + neutral
        row = {
            "topic": topic,
            "likes_pro": pro,
            "likes_anti": anti,
            "likes_neutral": neutral,
            "likes_total": total,
            "bias": (pro - anti) / total if total > 0 else np.nan,
        }
        if date_col:
            row["date"] = date_value
        rows.append(row)
    return pd.DataFrame(rows)


def daily_stance_bias_posts(
    config: dict[str, Any],
    year: int,
    *,
    start: str | None = None,
    end: str | None = None,
    rolling_days: int = 7,
) -> pd.DataFrame:
    if start is None or end is None:
        start, end = year_period(config, year)
    df = load_posts_labels(config, year)
    df = filter_period(df, start, end, date_col="creation_time")
    df["date"] = df["creation_time"].dt.floor("D")

    # Aggregate categories first, then smooth category volumes, then form the ratio.
    raw = _stance_counts(df, weight_col="like_count", date_col="date")
    if raw.empty:
        return raw
    full = _full_daily_index(start, end)
    out = []
    for topic in sorted(raw["topic"].unique()):
        sub = raw[raw["topic"] == topic].set_index("date").reindex(full)
        counts = sub[["likes_pro", "likes_anti", "likes_neutral"]].fillna(0.0)
        counts = centered_rolling_sum(counts, rolling_days)
        counts["likes_total"] = counts.sum(axis=1)
        counts["bias"] = (counts["likes_pro"] - counts["likes_anti"]) / counts["likes_total"].replace(0, np.nan)
        counts["topic"] = topic
        counts["date"] = counts.index
        out.append(counts.reset_index(drop=True))
    return pd.concat(out, ignore_index=True)


def aggregate_stance_bias_posts(config: dict[str, Any], year: int, start: str, end: str) -> pd.Series:
    df = filter_period(load_posts_labels(config, year), start, end, date_col="creation_time")
    result = _stance_counts(df, weight_col="like_count", date_col=None)
    return result.set_index("topic")["bias"] if not result.empty else pd.Series(dtype=float)


def aggregate_stance_bias_speeches(
    config: dict[str, Any], year: int, candidate: str, start: str, end: str
) -> pd.Series:
    # 07_correlations.ipynb loaded candidate stance from separate
    # {year}_speech_stance_{candidate}_time_series.pkl files. Prefer the
    # public aggregate generated from those inputs, then fall back to stance
    # labels embedded in the paragraph-level speech table.
    counts = load_speech_stance_counts(config, year, candidate)
    if not counts.empty:
        counts = filter_period(counts, start, end, date_col="date")
        result = _stance_counts(counts, weight_col="paragraph_count", date_col=None)
    else:
        df = filter_period(load_speeches(config, year, candidate), start, end, date_col="date")
        result = _stance_counts(df, weight_col=None, date_col=None)
    return result.set_index("topic")["bias"] if not result.empty else pd.Series(dtype=float)


def candidate_support_series(
    config: dict[str, Any],
    year: int,
    *,
    start: str | None = None,
    end: str | None = None,
    rolling_days: int = 7,
) -> pd.DataFrame:
    if year not in CANDIDATES_BY_YEAR:
        raise ValueError(f"Candidate support is only defined for {sorted(CANDIDATES_BY_YEAR)}")
    if start is None or end is None:
        start, end = year_period(config, year)
    df = load_posts_labels(config, year)
    df = filter_period(df, start, end, date_col="creation_time")
    df = df.dropna(subset=["candidate_stance"]).copy()
    df["date"] = df["creation_time"].dt.floor("D")

    democrat = CANDIDATES_BY_YEAR[year]["democrat"].split()[-1]
    expected = [f"Pro-{democrat}", f"Anti-{democrat}", "Pro-Trump", "Anti-Trump", "Neither"]
    grouped = df.groupby(["date", "candidate_stance"])["like_count"].sum().unstack(fill_value=0.0)
    grouped = grouped.reindex(_full_daily_index(start, end), fill_value=0.0)
    for label in expected:
        if label not in grouped.columns:
            grouped[label] = 0.0
    grouped = centered_rolling_sum(grouped[expected], rolling_days)

    dem_favorable = grouped[f"Pro-{democrat}"] + grouped["Anti-Trump"]
    rep_favorable = grouped["Pro-Trump"] + grouped[f"Anti-{democrat}"]
    total = grouped.sum(axis=1)
    support = (dem_favorable - rep_favorable) / total.replace(0, np.nan)
    out = grouped.copy()
    out["dem_favorable_likes"] = dem_favorable
    out["rep_favorable_likes"] = rep_favorable
    out["total_likes"] = total
    out["support_difference"] = support
    out["date"] = out.index
    return out.reset_index(drop=True)


def layer_volume_shares(wide: pd.DataFrame) -> pd.Series:
    totals = wide.sum(axis=0)
    denom = totals.sum()
    return totals / denom if denom > 0 else totals * np.nan
