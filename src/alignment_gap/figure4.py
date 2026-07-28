"""Reproduce candidate support and issue-level stance profiles (Figure 4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from .config import repo_path, year_period
from .constants import CANDIDATES_BY_YEAR, TOPIC_COLORS
from .series import candidate_support_series, daily_stance_bias_posts

TOPIC_INITIALS = {
    "Democratic concerns": "DC",
    "Wokeness": "W",
    "Economy": "E",
    "Immigration": "I",
    "Abortion": "A",
}


def _figure_4_stance_bias(
    config: dict[str, Any],
    year: int,
    *,
    start: str,
    end: str,
    rolling_days: int,
) -> pd.DataFrame:
    """Recreate the Figure 4 stance convention from ``06_time_series.ipynb``.

    The notebook explicitly overrode the Democratic-concerns polarity and
    treated ``Democrats threaten democracy`` as the positive class, while the
    manuscript-wide convention used elsewhere in the repository treats
    ``Republicans threaten democracy`` as positive.  To reproduce the plotted
    Figure 4 without changing Figures 2, 3, or 5, swap the category totals only
    in this figure.
    """
    bias = daily_stance_bias_posts(
        config,
        year,
        start=start,
        end=end,
        rolling_days=rolling_days,
    ).copy()
    if not bool(config["figure_4"].get("democratic_concerns_notebook_orientation", True)):
        return bias

    mask = bias["topic"].eq("Democratic concerns")
    if not mask.any():
        return bias

    pro = bias.loc[mask, "likes_pro"].copy()
    bias.loc[mask, "likes_pro"] = bias.loc[mask, "likes_anti"].to_numpy()
    bias.loc[mask, "likes_anti"] = pro.to_numpy()
    bias.loc[mask, "bias"] = -pd.to_numeric(bias.loc[mask, "bias"], errors="coerce")
    return bias


def _radar_summary(bias_df: pd.DataFrame, topics: list[str], start: str, end: str) -> pd.DataFrame:
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)
    sub = bias_df[(bias_df["date"] >= start_ts) & (bias_df["date"] <= end_ts) & bias_df["topic"].isin(topics)].copy()
    rows = []
    for topic in topics:
        topic_df = sub[sub["topic"] == topic]
        pro = float(topic_df["likes_pro"].sum())
        anti = float(topic_df["likes_anti"].sum())
        neutral = float(topic_df["likes_neutral"].sum())
        total = pro + anti + neutral
        rows.append(
            {
                "topic": topic,
                "likes_pro": pro,
                "likes_anti": anti,
                "likes_neutral": neutral,
                "likes_total": total,
                "stance_bias": (pro - anti) / total if total > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _plot_radar(ax, summary: pd.DataFrame, label: str) -> None:
    topics = summary["topic"].tolist()
    values = summary["stance_bias"].fillna(0.0).to_numpy(dtype=float)
    angles = np.linspace(0, 2 * np.pi, len(topics), endpoint=False)
    width = 2 * np.pi / len(topics) * 0.72
    colors = ["#4575b4" if value >= 0 else "#d73027" for value in values]
    ax.bar(angles, np.abs(values), width=width, color=colors, alpha=0.58, edgecolor="black", linewidth=0.35)
    ax.set_xticks(angles)
    ax.set_xticklabels([TOPIC_INITIALS.get(topic, topic[:2]) for topic in topics], fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(["0.5", "1.0"], fontsize=5)
    ax.set_title(label, fontsize=9, weight="bold", pad=4)
    ax.grid(alpha=0.35)


def _plot_year_column(
    fig,
    grid,
    config: dict[str, Any],
    year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start, end = year_period(config, year)
    window = int(config["figure_4"]["centered_window_days"])
    topics = list(config["figure_4"]["topics"])
    support = candidate_support_series(config, year, start=start, end=end, rolling_days=window)
    bias = _figure_4_stance_bias(
        config,
        year,
        start=start,
        end=end,
        rolling_days=window,
    )
    bias = bias[bias["topic"].isin(topics)].copy()

    ax_support = fig.add_subplot(grid[0])
    x = support["date"]
    y = support["support_difference"].to_numpy(dtype=float)
    ax_support.fill_between(x, 0, y, where=y >= 0, color="#4575b4", alpha=0.30)
    ax_support.fill_between(x, 0, y, where=y <= 0, color="#d73027", alpha=0.30)
    ax_support.plot(x, y, color="black", linewidth=1.25)
    ax_support.axhline(0, color="black", linewidth=0.6, linestyle="--")
    democrat = CANDIDATES_BY_YEAR[year]["democrat"].split()[-1]
    ax_support.set_title(f"{democrat} - Trump support difference", fontsize=10)
    ax_support.set_ylim(-1, 1)
    ax_support.grid(axis="y", alpha=0.25, linestyle="--")
    ax_support.tick_params(labelsize=7)

    ax_bias = fig.add_subplot(grid[1], sharex=ax_support)
    wide = bias.pivot(index="date", columns="topic", values="bias").sort_index()
    for topic in topics:
        if topic not in wide.columns:
            continue
        ax_bias.plot(wide.index, wide[topic], label=topic, color=TOPIC_COLORS.get(topic), linewidth=1.2)
    ax_bias.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax_bias.set_ylim(-1, 1)
    ax_bias.set_title("Stance bias", fontsize=10)
    ax_bias.grid(alpha=0.22, linestyle="--")
    ax_bias.tick_params(labelsize=7)

    windows = config["figure_4"]["radar_windows"][year]
    radar_grid = grid[2].subgridspec(1, len(windows), wspace=0.62)
    radar_rows = []
    for index, (window_start, window_end, label) in enumerate(windows):
        ax_radar = fig.add_subplot(radar_grid[0, index], projection="polar")
        summary = _radar_summary(bias, topics, window_start, window_end)
        _plot_radar(ax_radar, summary, label)
        summary["window_start"] = window_start
        summary["window_end"] = window_end
        summary["window_label"] = label
        summary["year"] = year
        radar_rows.append(summary)
        center = pd.to_datetime(window_start, utc=True) + (pd.to_datetime(window_end, utc=True) - pd.to_datetime(window_start, utc=True)) / 2
        ax_support.axvline(center, color="0.25", linewidth=0.65, linestyle="--")
        ax_bias.axvline(center, color="0.25", linewidth=0.65, linestyle="--")

    ax_bias.xaxis.set_major_locator(mdates.MonthLocator())
    ax_bias.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.setp(ax_support.get_xticklabels(), visible=False)
    plt.setp(ax_bias.get_xticklabels(), rotation=0, ha="center")
    ax_support.text(-0.07, 1.18, "(a)" if year == 2020 else "(b)", transform=ax_support.transAxes, fontsize=11, weight="bold")
    ax_support.text(0.5, 1.18, f"Support and stance profiles ({year})", transform=ax_support.transAxes, fontsize=12, weight="bold", ha="center")

    return support, bias, pd.concat(radar_rows, ignore_index=True)


def reproduce_figure_4(config: dict[str, Any]) -> list[Path]:
    output_dir = repo_path(config, "results") / "figure_4"
    data_dir = output_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(17, 8.2))
    outer = GridSpec(3, 2, figure=fig, height_ratios=[0.85, 1.18, 1.75], hspace=0.34, wspace=0.20)
    written: list[Path] = []
    for column, year in enumerate(config["figure_4"]["years"]):
        cells = (outer[0, column], outer[1, column], outer[2, column])
        support, bias, radars = _plot_year_column(fig, cells, config, int(year))
        support.to_csv(data_dir / f"candidate_support_{year}.csv", index=False)
        bias.to_csv(data_dir / f"stance_bias_{year}.csv", index=False)
        radars.to_csv(data_dir / f"radar_profiles_{year}.csv", index=False)

    legend_handles = [
        Patch(facecolor=TOPIC_COLORS.get(topic, "0.5"), label=topic)
        for topic in config["figure_4"]["topics"]
    ]
    fig.legend(legend_handles, [handle.get_label() for handle in legend_handles], loc="lower center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.01))
    fig.text(0.5, 0.055, "DC = Democratic concerns   |   W = Wokeness   |   E = Economy   |   I = Immigration   |   A = Abortion", ha="center", fontsize=8)
    fig.text(0.5, 0.035, "Blue radar bars: positive stance bias; red radar bars: negative stance bias", ha="center", fontsize=8)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.12)

    for suffix in ("pdf", "svg", "png"):
        path = output_dir / f"figure_4.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    written.extend(sorted(data_dir.glob("*")))
    return written
