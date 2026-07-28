"""Prepare the three source-specific topic layers used in Figures 2 and 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .config import year_period
from .constants import CANDIDATES_BY_YEAR, TOPIC_ORDER
from .series import (
    aggregate_stance_bias_posts,
    aggregate_stance_bias_speeches,
    daily_topic_volume_posts,
    daily_topic_volume_speeches,
    layer_volume_shares,
    notebook_daily_topic_like_ratio_posts,
    notebook_daily_topic_volume_speeches,
)


@dataclass
class ThreeLayerData:
    year: int
    left_label: str
    right_label: str
    left: pd.DataFrame
    public: pd.DataFrame
    right: pd.DataFrame
    left_volume: pd.Series
    public_volume: pd.Series
    right_volume: pd.Series
    left_bias: pd.Series
    public_bias: pd.Series
    right_bias: pd.Series

    @property
    def topic_order(self) -> list[str]:
        present = set(self.left.columns) | set(self.public.columns) | set(self.right.columns)
        ordered = [topic for topic in TOPIC_ORDER if topic in present]
        return ordered + sorted(present - set(ordered))


def prepare_three_layers(
    config: dict[str, Any],
    year: int,
    *,
    topics: list[str] | None = None,
) -> ThreeLayerData:
    """Prepare smoothed series and unsmoothed node volumes.

    When ``topics`` is provided (Figure 2), the same explicit topic ring is
    retained in all three layers, matching the final plotting notebook. A missing
    requested topic raises an explicit error. When omitted (Figure 3),
    each layer keeps topics above the configured minimum share.
    """
    if year not in CANDIDATES_BY_YEAR:
        raise ValueError(f"Three-layer analysis is defined for {sorted(CANDIDATES_BY_YEAR)}")
    default_start, default_end = year_period(config, year)
    period_override = config["figure_2"].get("correlation_periods", {}).get(year, {})
    start = str(period_override.get("start", default_start))
    end = str(period_override.get("end", default_end))
    window = int(config["figure_2"]["centered_window_days"])
    minimum_share = float(config["figure_2"]["minimum_layer_share"])
    candidates = CANDIDATES_BY_YEAR[year]
    left_label = candidates["democrat"]
    right_label = candidates["republican"]

    # Raw totals determine node volume. Correlations reproduce the notebook
    # signals: centered seven-day mean paragraph counts for speeches and a
    # centered seven-day mean of daily likes-per-post for public reaction.
    public_raw = daily_topic_volume_posts(
        config, year, metric="like_count", start=start, end=end, rolling_days=None
    )
    left_raw = daily_topic_volume_speeches(
        config, year, left_label, start=start, end=end, rolling_days=None
    )
    right_raw = daily_topic_volume_speeches(
        config, year, right_label, start=start, end=end, rolling_days=None
    )
    public_all = notebook_daily_topic_like_ratio_posts(
        config, year, start=start, end=end, rolling_days=window
    )
    left_all = notebook_daily_topic_volume_speeches(
        config, year, left_label, start=start, end=end, rolling_days=window
    )
    right_all = notebook_daily_topic_volume_speeches(
        config, year, right_label, start=start, end=end, rolling_days=window
    )

    if topics is not None:
        public_topics = list(topics)
        left_topics = list(topics)
        right_topics = list(topics)
        missing = {
            "public": [topic for topic in public_topics if topic not in public_all.columns],
            left_label: [topic for topic in left_topics if topic not in left_all.columns],
            right_label: [topic for topic in right_topics if topic not in right_all.columns],
        }
        missing = {layer: values for layer, values in missing.items() if values}
        if missing:
            details = "; ".join(f"{layer}: {', '.join(values)}" for layer, values in missing.items())
            raise ValueError(f"Figure 2 requested topics absent from prepared layer data ({details})")
        public = public_all[public_topics].copy()
        left = left_all[left_topics].copy()
        right = right_all[right_topics].copy()
    else:
        public_shares = layer_volume_shares(public_raw)
        left_shares = layer_volume_shares(left_raw)
        right_shares = layer_volume_shares(right_raw)
        public_topics = public_shares[public_shares >= minimum_share].index.tolist()
        left_topics = left_shares[left_shares >= minimum_share].index.tolist()
        right_topics = right_shares[right_shares >= minimum_share].index.tolist()
        public = public_all[public_topics]
        left = left_all[left_topics]
        right = right_all[right_topics]

    return ThreeLayerData(
        year=year,
        left_label=left_label,
        right_label=right_label,
        left=left,
        public=public,
        right=right,
        left_volume=left_raw.sum(axis=0).reindex(left_topics, fill_value=0.0),
        public_volume=public_raw.sum(axis=0).reindex(public_topics, fill_value=0.0),
        right_volume=right_raw.sum(axis=0).reindex(right_topics, fill_value=0.0),
        left_bias=aggregate_stance_bias_speeches(config, year, left_label, start, end).reindex(left_topics),
        public_bias=aggregate_stance_bias_posts(config, year, start, end).reindex(public_topics),
        right_bias=aggregate_stance_bias_speeches(config, year, right_label, start, end).reindex(right_topics),
    )
