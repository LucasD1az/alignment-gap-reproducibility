"""Figure 1 data export.

The final bump/alluvial drawing was produced outside Python in the paper, so this
module exports the exact compact tables needed for that external step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import repo_path, year_period
from .constants import STANCE_PRO_ANTI
from .series import filter_period, load_posts_labels


def build_figure_1_table(config: dict[str, Any], year: int) -> pd.DataFrame:
    start, end = year_period(config, year, end_key="figure_1_end")
    df = filter_period(load_posts_labels(config, year), start, end, date_col="creation_time")
    df = df[df["topic"] != "Not specified"].copy()
    threshold = float(config["figure_1"]["minor_topic_threshold_pct"]) / 100.0
    split_topics = set(config["figure_1"]["split_topics"])

    totals = (
        df.groupby("topic", as_index=False)
        .agg(post_count=("post_id", "size"), like_count=("like_count", "sum"), reaction_count=("reaction_count", "sum"))
    )
    grand_total = float(totals["like_count"].sum())
    totals["like_share"] = totals["like_count"] / grand_total if grand_total > 0 else np.nan
    major_topics = set(totals.loc[totals["like_share"] >= threshold, "topic"])

    rows: list[dict[str, Any]] = []
    for topic, sub in df.groupby("topic"):
        if topic not in major_topics:
            continue
        mapping = STANCE_PRO_ANTI.get(topic)
        if topic in split_topics and mapping:
            for stance in (mapping["pro"], mapping["anti"]):
                part = sub[sub["stance"] == stance]
                rows.append(
                    {
                        "year": year,
                        "topic": topic,
                        "stance": stance,
                        "post_count": int(len(part)),
                        "like_count": int(part["like_count"].sum()),
                        "reaction_count": int(part["reaction_count"].sum()),
                    }
                )
        else:
            rows.append(
                {
                    "year": year,
                    "topic": topic,
                    "stance": None,
                    "post_count": int(len(sub)),
                    "like_count": int(sub["like_count"].sum()),
                    "reaction_count": int(sub["reaction_count"].sum()),
                }
            )

    minor = totals[~totals["topic"].isin(major_topics)]
    if not minor.empty:
        rows.append(
            {
                "year": year,
                "topic": "Others",
                "stance": None,
                "post_count": int(minor["post_count"].sum()),
                "like_count": int(minor["like_count"].sum()),
                "reaction_count": int(minor["reaction_count"].sum()),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["like_share"] = out["like_count"] / out["like_count"].sum()
    topic_order = (
        out.groupby("topic")["like_count"].sum().sort_values(ascending=False).index.tolist()
    )
    out["topic_order"] = out["topic"].map({topic: i + 1 for i, topic in enumerate(topic_order)})
    return out.sort_values(["topic_order", "stance"], na_position="last").reset_index(drop=True)


def reproduce_figure_1(config: dict[str, Any]) -> list[Path]:
    output_dir = repo_path(config, "results") / "figure_1"
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    all_years = []
    for year in (2016, 2020, 2024):
        table = build_figure_1_table(config, year)
        path = output_dir / f"figure_1_{year}.csv"
        table.to_csv(path, index=False)
        written.append(path)
        all_years.append(table)
    combined = pd.concat(all_years, ignore_index=True)
    combined_path = output_dir / "figure_1_all_years.csv"
    combined.to_csv(combined_path, index=False)
    written.append(combined_path)
    return written
