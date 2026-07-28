"""Reproduce the decomposed state-level panels used in Figure 5."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import pearsonr

from .config import repo_path, year_period
from .constants import STANCE_PRO_ANTI
from .series import filter_period, load_posts_labels


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    sigma = float(values.std(ddof=0))
    if not np.isfinite(sigma) or sigma == 0:
        out = pd.Series(np.nan, index=series.index, dtype=float)
        out.loc[values.notna()] = 0.0
        return out
    return (values - float(values.mean())) / sigma


STATE_TOPIC_COLUMNS = [
    "year", "topic", "state_abbr", "likes_pro", "likes_anti", "likes_neutral",
    "likes", "likes_total", "n_posts", "impressions", "stance_bias",
    "share_in_state", "weighted_stance_bias", "stance_bias_z", "state",
    "democrat_pct", "republican_pct", "dem_minus_rep",
]


def load_election_results(config: dict[str, Any], year: int) -> pd.DataFrame:
    """Load the prepared election table independently of topic metrics.

    The electoral map is a standalone Figure 5 component and must not depend on
    whether posts and ads pages successfully overlap.
    """
    path = repo_path(config, "geography") / f"election_results_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Prepared election results missing for {year}: {path}")
    election = pd.read_csv(path, dtype={"state_abbr": "string"})
    required = {"state", "state_abbr", "democrat_pct", "republican_pct", "dem_minus_rep"}
    missing = sorted(required - set(election.columns))
    if missing:
        raise ValueError(
            f"Prepared election results for {year} are missing columns {missing}. "
            "Run scripts/prepare_public_data.py again with the current code."
        )
    election["state_abbr"] = election["state_abbr"].astype("string").str.strip().str.upper()
    for column in ("democrat_pct", "republican_pct", "dem_minus_rep"):
        election[column] = pd.to_numeric(election[column], errors="coerce")
    return election.dropna(subset=["state_abbr", "dem_minus_rep"]).drop_duplicates("state_abbr")


def geography_diagnostics(config: dict[str, Any], year: int) -> dict[str, Any]:
    """Summarize the page join used by the geographic model."""
    exposure_path = repo_path(config, "geography") / f"page_state_exposure_{year}.csv.gz"
    posts_path = repo_path(config, "posts") / f"posts_{year}.csv.gz"
    labels_path = repo_path(config, "labels") / f"labels_{year}.jsonl.gz"

    exposure = pd.read_csv(exposure_path, usecols=["page_id"], dtype={"page_id": "string"})
    posts = pd.read_csv(posts_path, usecols=["post_id", "page_id"], dtype={"post_id": "string", "page_id": "string"})
    exposure_pages = set(exposure["page_id"].dropna().astype(str))
    post_pages = set(posts["page_id"].dropna().astype(str))
    overlap = exposure_pages & post_pages

    labels = pd.read_json(labels_path, lines=True, compression="gzip")
    labeled = posts.merge(labels[["post_id", "topic", "stance"]], on="post_id", how="left")
    overlap_posts = labeled[labeled["page_id"].isin(overlap)].copy()
    valid_stance = overlap_posts[overlap_posts["stance"].notna()]

    return {
        "year": year,
        "n_post_pages": len(post_pages),
        "n_exposure_pages": len(exposure_pages),
        "n_overlapping_pages": len(overlap),
        "n_posts": int(len(posts)),
        "n_posts_on_overlapping_pages": int(len(overlap_posts)),
        "n_stance_labeled_posts_on_overlapping_pages": int(len(valid_stance)),
        "topics_on_overlapping_pages": sorted(overlap_posts["topic"].dropna().astype(str).unique().tolist()),
        "hint": (
            "If n_overlapping_pages is 0, rerun prepare_public_data.py with v0.1.7 or later. "
            "The original geography notebook joins posts to ads through post_owner.username; "
            "older releases selected post_owner.id whenever both columns existed."
        ),
    }


def build_state_topic_metrics(config: dict[str, Any], year: int) -> pd.DataFrame:
    """Recreate ``build_topic_state_df`` from ``geography_v3.ipynb``.

    Each post inherits its page-level state distribution.  The notebook first
    normalized each page row, zeroed state shares below ``min_state_weight``
    without renormalizing, and then distributed post likes across states.  Only
    the topic-specific pro, anti, and neutral stance classes enter the bias.
    """
    exposure_path = repo_path(config, "geography") / f"page_state_exposure_{year}.csv.gz"
    results_path = repo_path(config, "geography") / f"election_results_{year}.csv"
    if not exposure_path.exists() or not results_path.exists():
        raise FileNotFoundError(f"Prepared geography files missing for {year}. Run prepare_public_data.py first.")

    exposure = pd.read_csv(exposure_path, dtype={"page_id": "string", "state_abbr": "string"})
    exposure["impression_share"] = pd.to_numeric(exposure["impression_share"], errors="coerce").fillna(0.0)
    if "impression_value" not in exposure.columns:
        # Backward compatibility with releases prepared before v0.1.6.
        exposure["impression_value"] = exposure["impression_share"]
    exposure["impression_value"] = pd.to_numeric(exposure["impression_value"], errors="coerce").fillna(0.0)

    exposure_matrix = exposure.pivot_table(
        index="page_id",
        columns="state_abbr",
        values="impression_share",
        aggfunc="sum",
        fill_value=0.0,
    ).astype(float)
    row_sums = exposure_matrix.sum(axis=1)
    exposure_matrix = exposure_matrix.loc[row_sums > 0].div(row_sums[row_sums > 0], axis=0)

    minimum = float(config["figure_5"].get("min_state_weight", 0.001))
    exposure_thresholded = exposure_matrix.where(exposure_matrix >= minimum, 0.0)
    exposure_present = (exposure_thresholded > 0).astype(float)
    state_impressions = exposure.groupby("state_abbr")["impression_value"].sum()

    start, end = year_period(config, year)
    posts = filter_period(load_posts_labels(config, year), start, end, date_col="creation_time")
    posts = posts[posts["page_id"].isin(exposure_matrix.index)].copy()
    posts["like_count"] = pd.to_numeric(posts["like_count"], errors="coerce").fillna(0.0)

    rows: list[dict[str, Any]] = []
    matrix_values = exposure_thresholded.to_numpy(dtype=float)
    present_values = exposure_present.to_numpy(dtype=float)
    page_index = exposure_thresholded.index

    for topic, mapping in STANCE_PRO_ANTI.items():
        topic_posts = posts[
            posts["topic"].eq(topic)
            & posts["stance"].isin([mapping["pro"], mapping["anti"], "Neutral"])
        ].copy()
        if topic_posts.empty:
            continue

        category_vectors: dict[str, np.ndarray] = {}
        for category, label in (
            ("likes_pro", mapping["pro"]),
            ("likes_anti", mapping["anti"]),
            ("likes_neutral", "Neutral"),
        ):
            page_weights = (
                topic_posts.loc[topic_posts["stance"].eq(label)]
                .groupby("page_id")["like_count"]
                .sum()
                .reindex(page_index, fill_value=0.0)
            )
            category_vectors[category] = page_weights.to_numpy(dtype=float) @ matrix_values

        page_post_counts = topic_posts.groupby("page_id").size().reindex(page_index, fill_value=0.0)
        n_posts_by_state = page_post_counts.to_numpy(dtype=float) @ present_values

        for state_index, state in enumerate(exposure_thresholded.columns):
            pro = float(category_vectors["likes_pro"][state_index])
            anti = float(category_vectors["likes_anti"][state_index])
            neutral = float(category_vectors["likes_neutral"][state_index])
            total = pro + anti + neutral
            rows.append(
                {
                    "year": year,
                    "topic": topic,
                    "state_abbr": state,
                    "likes_pro": pro,
                    "likes_anti": anti,
                    "likes_neutral": neutral,
                    "likes": total,
                    "likes_total": total,
                    "n_posts": int(round(float(n_posts_by_state[state_index]))),
                    "impressions": float(state_impressions.get(state, np.nan)),
                    "stance_bias": (pro - anti) / total if total > 0 else np.nan,
                }
            )

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return pd.DataFrame(columns=STATE_TOPIC_COLUMNS)

    state_totals = metrics.groupby("state_abbr")["likes"].transform("sum")
    metrics["share_in_state"] = metrics["likes"] / state_totals.replace(0, np.nan)
    metrics["weighted_stance_bias"] = metrics["stance_bias"] * metrics["share_in_state"]
    metrics["stance_bias_z"] = metrics.groupby("topic")["stance_bias"].transform(_zscore)

    election = load_election_results(config, year)
    merged = metrics.merge(election, on="state_abbr", how="left", validate="many_to_one")
    return merged.reindex(columns=STATE_TOPIC_COLUMNS)


def _add_state_labels(fig, data: pd.DataFrame) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - guarded by caller
        raise RuntimeError("Figure 5 requires plotly. Install the project dependencies.") from exc
    fig.add_trace(
        go.Scattergeo(
            locations=data["state_abbr"],
            locationmode="USA-states",
            text=data["state_abbr"],
            mode="text",
            textfont={"size": 8, "color": "black"},
            hoverinfo="skip",
            showlegend=False,
        )
    )


def _electoral_map(election: pd.DataFrame, config: dict[str, Any], year: int):
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("Figure 5 requires plotly. Install the project dependencies.") from exc

    data = election.drop_duplicates("state_abbr").dropna(subset=["state_abbr", "dem_minus_rep"]).copy()
    zmin, zmax = [float(v) for v in config["figure_5"]["map"].get("election_range", [-75, 75])]
    fig = go.Figure(
        go.Choropleth(
            locations=data["state_abbr"],
            z=data["dem_minus_rep"],
            locationmode="USA-states",
            colorscale="RdBu",
            zmin=zmin,
            zmax=zmax,
            zmid=0,
            marker_line_color="white",
            marker_line_width=0.7,
            colorbar={"title": "DEM − GOP", "orientation": "h", "y": -0.08, "len": 0.72},
            text=data["state"],
            hovertemplate="%{text}<br>DEM − GOP: %{z:.2f}<extra></extra>",
        )
    )
    if bool(config["figure_5"]["map"].get("show_state_labels", True)):
        _add_state_labels(fig, data)
    fig.update_geos(scope="usa", showlakes=False, bgcolor="rgba(0,0,0,0)")
    fig.update_layout(
        title={"text": f"Electoral result ({year})", "x": 0.5, "xanchor": "center"},
        width=760,
        height=500,
        margin={"l": 10, "r": 10, "t": 60, "b": 65},
        template="plotly_white",
    )
    return fig


def _immigration_map(metrics: pd.DataFrame, config: dict[str, Any], year: int):
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("Figure 5 requires plotly. Install the project dependencies.") from exc

    topic = str(config["figure_5"].get("map_topic", "Immigration"))
    if metrics.empty:
        raise ValueError(f"No state-topic metrics were produced for {year}; see the geography diagnostics JSON.")
    data = metrics[metrics["topic"].eq(topic)].dropna(subset=["state_abbr", "stance_bias_z"]).copy()
    if data.empty:
        raise ValueError(f"No usable {topic} state metrics were produced for {year}.")
    zmin, zmax = [float(v) for v in config["figure_5"]["map"].get("immigration_z_range", [-2, 2])]
    fig = go.Figure(
        go.Choropleth(
            locations=data["state_abbr"],
            z=data["stance_bias_z"],
            locationmode="USA-states",
            colorscale="RdBu",
            zmin=zmin,
            zmax=zmax,
            zmid=0,
            marker_line_color="white",
            marker_line_width=0.7,
            colorbar={"title": "z stance bias", "orientation": "h", "y": -0.08, "len": 0.72},
            text=data["state"],
            hovertemplate="%{text}<br>z stance bias: %{z:.2f}<extra></extra>",
        )
    )
    if bool(config["figure_5"]["map"].get("show_state_labels", True)):
        _add_state_labels(fig, data)
    fig.update_geos(scope="usa", showlakes=False, bgcolor="rgba(0,0,0,0)")
    fig.update_layout(
        title={"text": f"{topic} stance bias z-score ({year})", "x": 0.5, "xanchor": "center"},
        width=760,
        height=500,
        margin={"l": 10, "r": 10, "t": 60, "b": 65},
        template="plotly_white",
    )
    return fig


def _short_pole(label: str) -> str:
    replacements = {
        "Republicans threaten democracy": "Republicans\nthreaten democracy",
        "Democrats threaten democracy": "Democrats\nthreaten democracy",
        "Positive Economic Outlook": "Positive",
        "Negative Economic Outlook": "Negative",
        "Woke supporter": "Pro woke",
        "Woke opposer": "Anti woke",
        "Pro-immigration": "Pro immigration",
        "Anti-immigration": "Anti immigration",
    }
    return replacements.get(label, label)


def _scatter_grid(metrics: pd.DataFrame, config: dict[str, Any], year: int):
    if metrics.empty:
        raise ValueError(f"No state-topic metrics were produced for {year}; see the geography diagnostics JSON.")
    topics = list(config["figure_5"]["scatter_topics"])
    style = config["figure_5"].get("scatter", {})
    xlim = tuple(float(v) for v in style.get("xlim", {}).get(year, [-60, 50]))
    ylim = tuple(float(v) for v in style.get("ylim", {}).get(year, [-1, 1]))
    point_size = float(style.get("point_size", 85))
    cmap_name = str(style.get("cmap", "RdBu"))
    show_fit = bool(style.get("show_fit_line", True))
    show_pearson = bool(style.get("show_pearson", True))
    swing_states = {str(value) for value in style.get("swing_states", [])}

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.4), sharex=True, sharey=True)
    axes = np.asarray(axes)
    norm = Normalize(vmin=-1.0, vmax=1.0)
    cmap = plt.get_cmap(cmap_name)

    for index, topic in enumerate(topics):
        ax = axes.flat[index]
        sub = metrics[metrics["topic"].eq(topic)].dropna(subset=["dem_minus_rep", "stance_bias"]).copy()
        x = sub["dem_minus_rep"].to_numpy(dtype=float)
        y = sub["stance_bias"].to_numpy(dtype=float)
        colors = cmap(norm(y))

        is_swing = sub["state"].isin(swing_states) if topic == "Wokeness" else pd.Series(False, index=sub.index)
        regular = ~is_swing.to_numpy()
        ax.scatter(
            x[regular],
            y[regular],
            s=point_size,
            c=colors[regular],
            edgecolors="black",
            linewidths=0.35,
            alpha=0.95,
            zorder=2,
        )
        if is_swing.any():
            swing_mask = is_swing.to_numpy()
            ax.scatter(
                x[swing_mask],
                y[swing_mask],
                s=point_size * 1.25,
                c=colors[swing_mask],
                edgecolors="purple",
                linewidths=1.2,
                alpha=1.0,
                zorder=3,
            )
            for row in sub.loc[is_swing].itertuples(index=False):
                ax.annotate(
                    row.state_abbr,
                    (row.dem_minus_rep, row.stance_bias),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                )

        if show_fit and len(sub) >= 2 and np.nanstd(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            xfit = np.linspace(xlim[0], xlim[1], 200)
            ax.plot(xfit, slope * xfit + intercept, color="black", linewidth=1.2, zorder=4)

        if show_pearson and len(sub) >= 2 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
            rho, _ = pearsonr(x, y)
            ax.text(
                0.97,
                0.96,
                rf"$\rho_P = {rho:.2f}$",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=12,
                bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "black", "alpha": 0.95},
            )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(topic, fontsize=15, pad=6)
        ax.tick_params(labelsize=11)

        mapping = STANCE_PRO_ANTI[topic]
        ax.text(0.02, 0.98, _short_pole(mapping["pro"]), transform=ax.transAxes, ha="left", va="top", fontsize=8)
        ax.text(0.02, 0.02, _short_pole(mapping["anti"]), transform=ax.transAxes, ha="left", va="bottom", fontsize=8)

        inset = ax.inset_axes([1.02, 0.18, 0.025, 0.64])
        colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=inset, orientation="vertical")
        colorbar.set_ticks([])
        colorbar.outline.set_visible(False)

    for ax in axes[:, 0]:
        ax.set_ylabel("Stance bias", fontsize=12)
    for ax in axes[1, :]:
        ax.set_xlabel("DEM − GOP electoral result", fontsize=12)

    fig.suptitle(f"Regional stance bias and electoral outcome ({year})", fontsize=17, y=0.995)
    fig.subplots_adjust(left=0.08, right=0.94, top=0.91, bottom=0.10, wspace=0.22, hspace=0.20)
    return fig


def _write_plotly_static(fig, path: Path) -> bool:
    try:
        fig.write_image(path, scale=2)
        return True
    except Exception as exc:  # kaleido/Chrome availability varies by machine
        warnings.warn(f"Could not write {path.name}: {exc}. The HTML map was still created.")
        return False


def _write_plotly_panel(fig, stem: Path, static_formats: list[str]) -> list[Path]:
    written: list[Path] = []
    html_path = stem.with_suffix(".html")
    fig.write_html(html_path, include_plotlyjs="cdn")
    written.append(html_path)
    for suffix in static_formats:
        path = stem.with_suffix(f".{suffix}")
        if _write_plotly_static(fig, path):
            written.append(path)
    return written


def reproduce_figure_5(config: dict[str, Any]) -> list[Path]:
    output_dir = repo_path(config, "results") / "figure_5"
    data_dir = output_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    static_formats = [str(value) for value in config["figure_5"].get("static_formats", [])]

    for year_value in config["figure_5"]["years"]:
        year = int(year_value)
        diagnostics = geography_diagnostics(config, year)
        diagnostics_path = data_dir / f"geography_diagnostics_{year}.json"
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append(diagnostics_path)

        election_data = load_election_results(config, year)
        election = _electoral_map(election_data, config, year)
        written.extend(_write_plotly_panel(election, output_dir / f"figure_5_electoral_map_{year}", static_formats))

        metrics = build_state_topic_metrics(config, year)
        data_path = data_dir / f"state_topic_metrics_{year}.csv"
        metrics.to_csv(data_path, index=False)
        written.append(data_path)
        if metrics.empty:
            raise ValueError(
                f"Figure 5 could not match posts to ads exposure pages for {year}. "
                f"Overlapping pages: {diagnostics['n_overlapping_pages']}; "
                f"posts on overlapping pages: {diagnostics['n_posts_on_overlapping_pages']}. "
                f"See {diagnostics_path}. Rerun prepare_public_data.py after applying v0.1.7."
            )

        immigration = _immigration_map(metrics, config, year)
        written.extend(_write_plotly_panel(immigration, output_dir / f"figure_5_immigration_zscore_{year}", static_formats))

        scatter = _scatter_grid(metrics, config, year)
        for suffix in ("pdf", "svg", "png"):
            path = output_dir / f"figure_5_scatter_grid_{year}.{suffix}"
            scatter.savefig(path, dpi=300, bbox_inches="tight")
            written.append(path)
        plt.close(scatter)

    return written
