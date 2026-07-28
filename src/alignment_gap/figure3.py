"""Reproduce the two independently composed components of Figure 3.

The paper panel was assembled outside Python.  This module therefore exports,
for each election year, (1) the pseudo-triangular three-layer block heatmap
created by ``plot_three_layer_block_heatmap_from_export`` and (2) the reduced
four-topic network created by ``plot_three_subnet_topic_network_matplotlib``.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from .config import repo_path
from .correlations import build_three_layer_correlations, hierarchical_order
from .layers import ThreeLayerData, prepare_three_layers


def _short(topic: str) -> str:
    special = {
        "Democratic concerns": "DC",
        "National security and foreign policy": "NS",
        "Guns control": "GC",
        "Healthcare": "H",
        "Wokeness": "W",
        "Economy": "E",
        "Immigration": "I",
        "Abortion": "A",
    }
    if topic in special:
        return special[topic]
    words = re.findall(r"[A-Za-z0-9]+", topic)
    return "".join(word[0].upper() for word in words if word.casefold() not in {"and", "of", "the", "to"})[:3]


def _correlations_for_year(config: dict[str, Any], year: int) -> tuple[ThreeLayerData, dict[str, Any]]:
    data = prepare_three_layers(config, year)
    corr = build_three_layer_correlations(
        data.left,
        data.public,
        data.right,
        left_label=data.left_label,
        right_label=data.right_label,
        alpha=float(config["figure_3"]["alpha"]),
        lag_min=int(config["figure_2"]["lag_min_days"]),
        lag_max=int(config["figure_2"]["lag_max_days"]),
        alignment=str(config["figure_2"].get("lag_alignment", "legacy_index")),
        fallback_to_all_lags=bool(config["figure_2"].get("fallback_to_all_lags", True)),
    )
    return data, corr


def _subset_square(df: pd.DataFrame, topics: list[str]) -> pd.DataFrame:
    keep = [topic for topic in topics if topic in df.index and topic in df.columns]
    return df.loc[keep, keep]


def _mask_block(
    values: pd.DataFrame,
    pvals: pd.DataFrame,
    *,
    alpha: float,
    minimum_abs: float,
    keep_diagonal: bool,
) -> pd.DataFrame:
    out = values.astype(float).copy()
    p = pvals.reindex_like(out).astype(float)
    keep = out.notna() & p.notna() & (p <= alpha) & (out.abs() >= minimum_abs)
    if keep_diagonal:
        for i in range(min(len(out.index), len(out.columns))):
            keep.iat[i, i] = True
    return out.where(keep)


def _scale_radius(values: pd.Series, minimum: float, maximum: float) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    finite = values.dropna()
    if finite.empty:
        return pd.Series((minimum + maximum) / 2.0, index=values.index)
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return pd.Series((minimum + maximum) / 2.0, index=values.index)
    return minimum + (values - lo) / (hi - lo) * (maximum - minimum)


def _stance_color(value: float | None, absmax: float) -> tuple[float, float, float, float]:
    if value is None or not np.isfinite(value):
        return mcolors.to_rgba("#BDBDBD")
    norm = mcolors.TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)
    return plt.get_cmap("RdBu")(norm(float(value)))


def _signed_corr_color(value: float) -> tuple[float, float, float, float]:
    transformed = np.sign(float(value)) * abs(float(value)) ** 2
    return plt.get_cmap("BrBG")((transformed + 1.0) / 2.0)


def _write_heatmap(
    config: dict[str, Any],
    year: int,
    data: ThreeLayerData,
    corr: dict[str, Any],
    output_dir: Path,
    data_dir: Path,
) -> list[Path]:
    cfg = config["figure_3"]
    style = cfg.get("heatmap", {})
    alpha = float(cfg["alpha"])
    thresholds = cfg["thresholds"][year]
    requested = list(cfg["topics"])

    left_corr = _subset_square(corr["left_corr"], requested)
    mid_corr = _subset_square(corr["public_corr"], requested)
    right_corr = _subset_square(corr["right_corr"], requested)
    left_p = corr["left_p"].loc[left_corr.index, left_corr.columns]
    mid_p = corr["public_p"].loc[mid_corr.index, mid_corr.columns]
    right_p = corr["right_p"].loc[right_corr.index, right_corr.columns]

    left_order = hierarchical_order(left_corr)
    mid_order = hierarchical_order(mid_corr)
    right_order = hierarchical_order(right_corr)
    orders = {"left": left_order, "public": mid_order, "right": right_order}

    left_block = _mask_block(
        left_corr.loc[left_order, left_order], left_p.loc[left_order, left_order],
        alpha=alpha, minimum_abs=float(thresholds["within_candidate"]), keep_diagonal=True,
    )
    mid_block = _mask_block(
        mid_corr.loc[mid_order, mid_order], mid_p.loc[mid_order, mid_order],
        alpha=alpha, minimum_abs=float(thresholds["within_public"]), keep_diagonal=True,
    )
    right_block = _mask_block(
        right_corr.loc[right_order, right_order], right_p.loc[right_order, right_order],
        alpha=alpha, minimum_abs=float(thresholds["within_candidate"]), keep_diagonal=True,
    )

    lm_rho = corr["left_public"]["rho"].loc[left_order, mid_order]
    lm_p = corr["left_public"]["p"].loc[left_order, mid_order]
    mr_rho = corr["public_right"]["rho"].loc[mid_order, right_order]
    mr_p = corr["public_right"]["p"].loc[mid_order, right_order]
    lm_block = _mask_block(
        lm_rho, lm_p, alpha=alpha,
        minimum_abs=float(thresholds["candidate_public"]), keep_diagonal=False,
    )
    mr_block = _mask_block(
        mr_rho, mr_p, alpha=alpha,
        minimum_abs=float(thresholds["candidate_public"]), keep_diagonal=False,
    )

    gap = int(style.get("gap", 2))
    sizes = [len(left_order), len(mid_order), len(right_order)]
    starts = [0, sizes[0] + gap, sizes[0] + gap + sizes[1] + gap]
    total = sum(sizes) + 2 * gap
    canvas = np.full((total, total), np.nan, dtype=float)
    s_left, s_mid, s_right = starts
    canvas[s_left:s_left + sizes[0], s_left:s_left + sizes[0]] = left_block.to_numpy()
    canvas[s_mid:s_mid + sizes[1], s_mid:s_mid + sizes[1]] = mid_block.to_numpy()
    canvas[s_right:s_right + sizes[2], s_right:s_right + sizes[2]] = right_block.to_numpy()
    # Notebook layout: inter-layer blocks only below the diagonal.
    canvas[s_mid:s_mid + sizes[1], s_left:s_left + sizes[0]] = lm_block.T.to_numpy()
    canvas[s_right:s_right + sizes[2], s_mid:s_mid + sizes[1]] = mr_block.T.to_numpy()

    labels = (
        [_short(topic) for topic in left_order] + [""] * gap
        + [_short(topic) for topic in mid_order] + [""] * gap
        + [_short(topic) for topic in right_order]
    )
    transformed = np.sign(canvas) * np.abs(canvas) ** 2
    masked = np.ma.masked_invalid(transformed)
    figsize = tuple(style.get("figsize", [12, 12]))
    fig, ax = plt.subplots(figsize=figsize)
    edges = np.arange(total + 1) - 0.5
    image = ax.pcolormesh(edges, edges, masked, cmap="BrBG", vmin=-1, vmax=1, shading="flat")
    ax.set_xlim(-0.5, total - 0.5)
    ax.set_ylim(total - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(total)); ax.set_yticks(range(total))
    fontsize = float(style.get("initials_fontsize", 13))
    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.set_yticklabels(labels, fontsize=fontsize)
    ax.tick_params(length=0)

    show_values = bool(style.get("show_values", True))
    if show_values:
        value_fontsize = float(style.get("value_fontsize", 7))
        white_threshold = float(style.get("value_text_threshold_abs", 0.70))
        for i in range(total):
            for j in range(total):
                value = canvas[i, j]
                if np.isfinite(value):
                    ax.text(
                        j, i, f"{value:.2f}", ha="center", va="center",
                        fontsize=value_fontsize,
                        color="white" if abs(value) >= white_threshold else "black",
                    )

    def rect(row: int, col: int, height: int, width: int, color: str = "lightgray", lw: float = 1.0) -> None:
        ax.add_patch(Rectangle((col - 0.5, row - 0.5), width, height, fill=False, edgecolor=color, linewidth=lw))

    rect(s_left, s_left, sizes[0], sizes[0])
    rect(s_mid, s_mid, sizes[1], sizes[1])
    rect(s_right, s_right, sizes[2], sizes[2])
    rect(s_mid, s_left, sizes[1], sizes[0])
    rect(s_right, s_mid, sizes[2], sizes[1])

    # Blue boxes correspond to the four-topic reduced network used in the paper composition.
    highlight = set(cfg["subnet_topics"][year])
    for i, mid_topic in enumerate(mid_order):
        for j, left_topic in enumerate(left_order):
            if mid_topic in highlight and left_topic in highlight and pd.notna(lm_block.loc[left_topic, mid_topic]):
                rect(s_mid + i, s_left + j, 1, 1, color="#2563eb", lw=2.2)
    for i, right_topic in enumerate(right_order):
        for j, mid_topic in enumerate(mid_order):
            if right_topic in highlight and mid_topic in highlight and pd.notna(mr_block.loc[mid_topic, right_topic]):
                rect(s_right + i, s_mid + j, 1, 1, color="#2563eb", lw=2.2)

    centers = [start + (size - 1) / 2 for start, size in zip(starts, sizes)]
    layer_font = float(style.get("layer_label_fontsize", 15))
    for center, label in zip(centers, [data.left_label, "Public Reaction", data.right_label]):
        ax.text(center, -1.7, label, ha="center", va="bottom", fontsize=layer_font, weight="bold", clip_on=False)

    # Topic legend occupies the intentionally empty left-right block.
    entries = [f"{_short(topic)} = {topic}" for topic in dict.fromkeys(left_order + mid_order + right_order)]
    legend_x = s_left
    legend_y = s_right
    ax.text(legend_x, legend_y, "Topic initials\n" + "\n".join(entries), ha="left", va="top", fontsize=9)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Spearman correlation")
    fig.tight_layout()

    written: list[Path] = []
    for suffix in ("pdf", "svg", "png"):
        path = output_dir / f"figure_3_heatmap_{year}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        written.append(path)
    plt.close(fig)

    matrix_labels = (
        [f"L|{topic}" for topic in left_order] + [f"gapL{i}" for i in range(gap)]
        + [f"P|{topic}" for topic in mid_order] + [f"gapP{i}" for i in range(gap)]
        + [f"R|{topic}" for topic in right_order]
    )
    canvas_df = pd.DataFrame(canvas, index=matrix_labels, columns=matrix_labels)
    matrix_path = data_dir / f"figure_3_heatmap_matrix_{year}.csv"
    orders_path = data_dir / f"figure_3_heatmap_orders_{year}.json"
    canvas_df.to_csv(matrix_path)
    orders_path.write_text(json.dumps(orders, indent=2) + "\n", encoding="utf-8")
    return written + [matrix_path, orders_path]


def _layer_positions(center: tuple[float, float], radius: float) -> list[tuple[float, float]]:
    cx, cy = center
    positions = [(cx, cy)]
    for angle in (90, 210, 330):
        theta = np.deg2rad(angle)
        positions.append((cx + radius * np.cos(theta), cy + radius * np.sin(theta)))
    return positions


def _shrink_segment(start: tuple[float, float], end: tuple[float, float], pad_start: float, pad_end: float):
    x0, y0 = start; x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length <= pad_start + pad_end or length == 0:
        return x0, y0, x1, y1
    ux, uy = dx / length, dy / length
    return x0 + ux * pad_start, y0 + uy * pad_start, x1 - ux * pad_end, y1 - uy * pad_end


def _write_subnetwork(
    config: dict[str, Any],
    year: int,
    data: ThreeLayerData,
    corr: dict[str, Any],
    output_dir: Path,
    data_dir: Path,
) -> list[Path]:
    cfg = config["figure_3"]
    style = cfg.get("subnet", {})
    alpha = float(cfg["alpha"])
    thresholds = cfg["thresholds"][year]
    topics = [topic for topic in cfg["subnet_topics"][year] if topic in data.public.columns]
    if len(topics) != 4:
        raise ValueError(f"Figure 3 subnetwork for {year} requires exactly four available topics; got {topics}")

    centers = {
        "left": tuple(style.get("left_center", [-3.8, -1.0])),
        "public": tuple(style.get("public_center", [0.0, 1.7])),
        "right": tuple(style.get("right_center", [3.8, -1.0])),
    }
    ring = float(style.get("ring_radius", 1.15))
    positions: dict[tuple[str, str], tuple[float, float]] = {}
    for layer in ("left", "public", "right"):
        for topic, xy in zip(topics, _layer_positions(centers[layer], ring)):
            positions[(layer, topic)] = xy

    volumes = {"left": data.left_volume, "public": data.public_volume, "right": data.right_volume}
    biases = {"left": data.left_bias, "public": data.public_bias, "right": data.right_bias}
    rmin = float(style.get("node_radius_min", 0.22))
    rmax = float(style.get("node_radius_max", 0.40))
    radii = {layer: _scale_radius(volume.reindex(topics), rmin, rmax) for layer, volume in volumes.items()}
    all_bias = pd.concat([series.reindex(topics) for series in biases.values()]).dropna()
    bias_absmax = float(all_bias.abs().max()) if not all_bias.empty else 1.0
    if bias_absmax <= 0:
        bias_absmax = 1.0

    fig, ax = plt.subplots(figsize=tuple(style.get("figsize", [10, 8])))
    edge_rows: list[dict[str, Any]] = []

    def draw_intra(matrix_key: str, p_key: str, layer: str, threshold: float) -> None:
        values, pvals = corr[matrix_key], corr[p_key]
        for i, topic_i in enumerate(topics):
            for topic_j in topics[i + 1:]:
                rho, p = values.loc[topic_i, topic_j], pvals.loc[topic_i, topic_j]
                if not (np.isfinite(rho) and np.isfinite(p) and p <= alpha and abs(rho) >= threshold):
                    continue
                x0, y0 = positions[(layer, topic_i)]; x1, y1 = positions[(layer, topic_j)]
                ax.plot([x0, x1], [y0, y1], color="#444444", lw=1.2 + 3.0 * abs(rho), alpha=0.8, zorder=1)
                edge_rows.append({"edge_type": "intra", "layer_pair": layer, "from_topic": topic_i, "to_topic": topic_j, "rho": rho, "p": p, "lag": 0})

    draw_intra("left_corr", "left_p", "left", float(thresholds["within_candidate"]))
    draw_intra("public_corr", "public_p", "public", float(thresholds["within_public"]))
    draw_intra("right_corr", "right_p", "right", float(thresholds["within_candidate"]))

    def draw_inter(outputs: dict[str, pd.DataFrame], row_layer: str, col_layer: str, pair: str) -> None:
        values, pvals, lags = outputs["rho"], outputs["p"], outputs["lag"]
        for row_topic in topics:
            for col_topic in topics:
                rho, p, lag = values.loc[row_topic, col_topic], pvals.loc[row_topic, col_topic], lags.loc[row_topic, col_topic]
                if not (np.isfinite(rho) and np.isfinite(p) and p <= alpha and abs(rho) >= float(thresholds["candidate_public"])):
                    continue
                row_xy = positions[(row_layer, row_topic)]
                col_xy = positions[(col_layer, col_topic)]
                if pair == "left_public":
                    start_layer, start_topic, end_layer, end_topic = (
                        (row_layer, row_topic, col_layer, col_topic) if lag < 0
                        else (col_layer, col_topic, row_layer, row_topic)
                    )
                else:  # public_right: row=public, col=right candidate
                    start_layer, start_topic, end_layer, end_topic = (
                        (col_layer, col_topic, row_layer, row_topic) if lag < 0
                        else (row_layer, row_topic, col_layer, col_topic)
                    )
                start = positions[(start_layer, start_topic)]
                end = positions[(end_layer, end_topic)]
                start_r = float(radii[start_layer].get(start_topic, (rmin + rmax) / 2))
                end_r = float(radii[end_layer].get(end_topic, (rmin + rmax) / 2))
                x0, y0, x1, y1 = _shrink_segment(start, end, 1.05 * start_r, 1.25 * end_r)
                arrow = FancyArrowPatch(
                    (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13 + 4 * abs(rho),
                    linewidth=2.0 + 4.0 * abs(rho), color=_signed_corr_color(rho), alpha=0.85,
                    connectionstyle="arc3,rad=0.0", zorder=2,
                )
                ax.add_patch(arrow)
                edge_rows.append({"edge_type": "inter", "layer_pair": pair, "from_layer": start_layer, "to_layer": end_layer, "from_topic": start_topic, "to_topic": end_topic, "rho": rho, "p": p, "lag": lag})

    draw_inter(corr["left_public"], "left", "public", "left_public")
    draw_inter(corr["public_right"], "public", "right", "public_right")

    node_rows: list[dict[str, Any]] = []
    labels = {"left": data.left_label, "public": "Public Reaction", "right": data.right_label}
    for layer in ("left", "public", "right"):
        for topic in topics:
            x, y = positions[(layer, topic)]
            radius = float(radii[layer].get(topic, (rmin + rmax) / 2))
            bias = float(biases[layer].get(topic, np.nan))
            ax.add_patch(Circle((x, y), radius=radius, facecolor=_stance_color(bias, bias_absmax), edgecolor="black", linewidth=1.2, zorder=3))
            ax.text(x, y, _short(topic), ha="center", va="center", fontsize=float(style.get("node_text_fontsize", 12)), zorder=4)
            node_rows.append({"layer": labels[layer], "topic": topic, "volume": float(volumes[layer].get(topic, np.nan)), "stance_bias": bias, "x": x, "y": y, "radius": radius})
        cx, cy = centers[layer]
        ax.text(cx, cy + ring + 0.75, labels[layer], ha="center", va="bottom", fontsize=float(style.get("layer_title_fontsize", 15)))

    ax.set_aspect("equal"); ax.axis("off")
    xy = np.asarray(list(positions.values()))
    ax.set_xlim(xy[:, 0].min() - 1.5, xy[:, 0].max() + 1.5)
    ax.set_ylim(xy[:, 1].min() - 1.5, xy[:, 1].max() + 1.6)
    fig.tight_layout()

    written: list[Path] = []
    for suffix in ("pdf", "svg", "png"):
        path = output_dir / f"figure_3_subnetwork_{year}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        written.append(path)
    plt.close(fig)

    nodes_path = data_dir / f"figure_3_subnetwork_nodes_{year}.csv"
    edges_path = data_dir / f"figure_3_subnetwork_edges_{year}.csv"
    pd.DataFrame(node_rows).to_csv(nodes_path, index=False)
    pd.DataFrame(edge_rows).to_csv(edges_path, index=False)
    return written + [nodes_path, edges_path]


def reproduce_figure_3(config: dict[str, Any]) -> list[Path]:
    """Write heatmap and reduced network as separate files for each year."""
    output_dir = repo_path(config, "results") / "figure_3"
    data_dir = output_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for year_value in config["figure_3"]["years"]:
        year = int(year_value)
        data, corr = _correlations_for_year(config, year)
        written.extend(_write_heatmap(config, year, data, corr, output_dir, data_dir))
        written.extend(_write_subnetwork(config, year, data, corr, output_dir, data_dir))
    return written
