"""Reproduce the multilayer discourse–engagement networks in Figure 2.

The visual layout is a static Matplotlib port of the final
``plot_multilayer_topic_network`` function used in ``07_speeches_bis.ipynb``:
three elliptical topic rings, perspective planes, black within-layer links and
purple directed lagged links between matching topics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Polygon

from .config import repo_path
from .correlations import (
    inter_correlation_outputs,
    intra_correlation_matrices,
    intra_links_from_matrices,
    matching_topic_links,
)
from .layers import ThreeLayerData, prepare_three_layers


_LAYER_KEYS = ("left", "public", "right")


def _scale_diameters(values: pd.Series, low: float, high: float) -> dict[str, float]:
    """Min–max scale values to marker diameters, matching the notebook logic."""
    values = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    if values.empty:
        return {}
    vmin, vmax = float(values.min()), float(values.max())
    if vmax <= vmin:
        return {str(index): (low + high) / 2.0 for index in values.index}
    return {
        str(index): low + (high - low) * (float(value) - vmin) / (vmax - vmin)
        for index, value in values.items()
    }




def _scale(values: pd.Series, low: float = 70.0, high: float = 700.0) -> dict[str, float]:
    """Backward-compatible area scaling used by the Figure 3 inset."""
    diameters = _scale_diameters(values, np.sqrt(low), np.sqrt(high))
    return {topic: float(diameter) ** 2 for topic, diameter in diameters.items()}


def _node_color(bias: float | None, cmap: str = "RdBu") -> tuple[float, float, float, float]:
    """Backward-compatible fixed [-1, 1] stance color for Figure 3."""
    if bias is None or not np.isfinite(bias):
        return (0.92, 0.92, 0.92, 1.0)
    return tuple(plt.get_cmap(cmap)((float(np.clip(bias, -1.0, 1.0)) + 1.0) / 2.0))




def _display_topic(topic: str) -> str:
    """Compact labels matching the terminology used in the manuscript."""
    return {
        "National security and foreign policy": "National security\nand foreign policy",
        "Democratic concerns": "Democratic\nconcerns",
        "Guns control": "Guns control",
    }.get(str(topic), str(topic))


def _hex_to_rgb(hex_color: str) -> np.ndarray:
    value = str(hex_color).strip().lstrip("#")
    if len(value) == 3:
        value = "".join(char * 2 for char in value)
    return np.array([int(value[i : i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def _stance_color(
    value: float | None,
    *,
    absmax: float,
    negative: str = "#C62828",
    midpoint: str = "#F2F2F2",
    positive: str = "#1565C0",
    missing: str = "#BDBDBD",
) -> tuple[float, float, float]:
    if value is None or not np.isfinite(value):
        return tuple(_hex_to_rgb(missing))
    absmax = max(float(absmax), 1e-12)
    value = float(np.clip(value, -absmax, absmax))
    if value < 0:
        t = (value + absmax) / absmax
        rgb = (1.0 - t) * _hex_to_rgb(negative) + t * _hex_to_rgb(midpoint)
    elif value > 0:
        t = value / absmax
        rgb = (1.0 - t) * _hex_to_rgb(midpoint) + t * _hex_to_rgb(positive)
    else:
        rgb = _hex_to_rgb(midpoint)
    return tuple(rgb)


def _bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    samples: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, int(samples))
    one = 1.0 - t
    x = one**3 * p0[0] + 3 * one**2 * t * p1[0] + 3 * one * t**2 * p2[0] + t**3 * p3[0]
    y = one**3 * p0[1] + 3 * one**2 * t * p1[1] + 3 * one * t**2 * p2[1] + t**3 * p3[1]
    return x, y


def _trim_curve_near_target(
    xs: np.ndarray,
    ys: np.ndarray,
    target: tuple[float, float],
    pad: float,
) -> tuple[np.ndarray, np.ndarray]:
    if pad <= 0:
        return xs, ys
    distance = np.sqrt((xs - target[0]) ** 2 + (ys - target[1]) ** 2)
    candidates = np.where(distance >= pad)[0]
    if len(candidates) == 0:
        return xs[:2], ys[:2]
    index = max(1, min(int(candidates[-1]), len(xs) - 1))
    return xs[: index + 1], ys[: index + 1]


def _positions(
    topics: list[str],
    *,
    layer_x: dict[str, float],
    ellipse_rx: float,
    ellipse_ry: float,
    rotation_deg: float,
) -> dict[tuple[str, str], tuple[float, float]]:
    rotation = np.deg2rad(float(rotation_deg))
    angles = np.linspace(0.0, 2.0 * np.pi, len(topics), endpoint=False) + rotation
    positions: dict[tuple[str, str], tuple[float, float]] = {}
    for topic, angle in zip(topics, angles):
        for layer in _LAYER_KEYS:
            positions[(layer, topic)] = (
                float(layer_x[layer]) + float(ellipse_rx) * np.cos(float(angle)),
                float(ellipse_ry) * np.sin(float(angle)),
            )
    return positions


def _plane_vertices(
    center_x: float,
    *,
    ellipse_rx: float,
    ellipse_ry: float,
    pad_x: float,
    pad_y: float,
    perspective: float,
    shear: float,
) -> list[tuple[float, float]]:
    x_right = float(ellipse_rx + pad_x)
    x_left = float(-ellipse_rx - pad_x)
    height_right = float(2 * ellipse_ry + 2 * pad_y)
    height_left = float(height_right * perspective)
    return [
        (center_x + x_left + shear, height_right / 2),
        (center_x + x_right + shear, height_left / 2),
        (center_x + x_right, -height_right / 2),
        (center_x + x_left, -height_left / 2),
    ]


def _draw_plane(ax: Axes, vertices: list[tuple[float, float]]) -> None:
    ax.add_patch(
        Polygon(
            vertices,
            closed=True,
            facecolor=(210 / 255, 210 / 255, 210 / 255, 0.60),
            edgecolor=(0, 0, 0, 0.10),
            linewidth=1.0,
            zorder=2,
        )
    )


def _intra_edges(
    corr: pd.DataFrame,
    pvals: pd.DataFrame,
    *,
    alpha: float,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    topics = list(corr.columns)
    for i, topic_i in enumerate(topics):
        for topic_j in topics[i + 1 :]:
            rho = corr.loc[topic_i, topic_j]
            p = pvals.loc[topic_i, topic_j]
            if not np.isfinite(rho) or not np.isfinite(p):
                continue
            if p > alpha or abs(float(rho)) < threshold:
                continue
            rows.append({"topic_i": topic_i, "topic_j": topic_j, "rho": float(rho), "p": float(p)})
    return pd.DataFrame(rows, columns=["topic_i", "topic_j", "rho", "p"])


def _figure_2_outputs(
    config: dict[str, Any], year: int
) -> tuple[ThreeLayerData, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Rebuild the matrices exported by ``export_three_layer_correlations``.

    The original workflow first exported full all-topic-pair inter-layer
    matrices and only afterwards selected matching-topic links for the
    multilayer plot. Keeping those two stages separate is necessary for an
    auditable reproduction of the notebook output.
    """
    topics = list(config["figure_2"]["topics"][year])
    data = prepare_three_layers(config, year, topics=topics)
    alpha = float(config["figure_2"]["inter_alpha"])
    lag_min = int(config["figure_2"]["lag_min_days"])
    lag_max = int(config["figure_2"]["lag_max_days"])
    inter_threshold = float(config["figure_2"]["inter_min_abs_correlation"][year])
    alignment = str(config["figure_2"].get("lag_alignment", "legacy_index"))
    fallback = bool(config["figure_2"].get("fallback_to_all_lags", True))

    left_corr, left_p = intra_correlation_matrices(data.left, alignment=alignment)
    public_corr, public_p = intra_correlation_matrices(data.public, alignment=alignment)
    right_corr, right_p = intra_correlation_matrices(data.right, alignment=alignment)

    left_public_all = inter_correlation_outputs(
        data.left,
        data.public,
        alpha=alpha,
        lag_min=lag_min,
        lag_max=lag_max,
        left_label=data.left_label,
        right_label="Public Reaction",
        pair_key="left_mid",
        pair_label="Left–Mid",
        alignment=alignment,
        fallback_to_all_lags=fallback,
    )
    public_right_all = inter_correlation_outputs(
        data.public,
        data.right,
        alpha=alpha,
        lag_min=lag_min,
        lag_max=lag_max,
        left_label="Public Reaction",
        right_label=data.right_label,
        pair_key="mid_right",
        pair_label="Mid–Right",
        alignment=alignment,
        fallback_to_all_lags=fallback,
    )

    same_left_public = matching_topic_links(
        left_public_all, minimum_abs_rho=inter_threshold
    )
    same_public_right = matching_topic_links(
        public_right_all, minimum_abs_rho=inter_threshold
    )
    left_intra = intra_links_from_matrices(
        left_corr, left_p, layer_key="left", layer_label=data.left_label
    )
    public_intra = intra_links_from_matrices(
        public_corr, public_p, layer_key="mid", layer_label="Public Reaction"
    )
    right_intra = intra_links_from_matrices(
        right_corr, right_p, layer_key="right", layer_label=data.right_label
    )

    matrices = {
        "left_corr": left_corr,
        "left_p": left_p,
        "public_corr": public_corr,
        "public_p": public_p,
        "right_corr": right_corr,
        "right_p": right_p,
        "left_public_rho": left_public_all["rho"],
        "left_public_p": left_public_all["p"],
        "left_public_lag": left_public_all["lag"],
        "left_public_n": left_public_all["n"],
        "left_public_selected_from": left_public_all["selected_from"],
        "public_right_rho": public_right_all["rho"],
        "public_right_p": public_right_all["p"],
        "public_right_lag": public_right_all["lag"],
        "public_right_n": public_right_all["n"],
        "public_right_selected_from": public_right_all["selected_from"],
    }
    links = {
        "intra_left_all": left_intra,
        "intra_public_all": public_intra,
        "intra_right_all": right_intra,
        "intra_all": pd.concat([left_intra, public_intra, right_intra], ignore_index=True),
        "left_public_all_pairs": left_public_all["links"],
        "public_right_all_pairs": public_right_all["links"],
        "inter_all_pairs": pd.concat(
            [left_public_all["links"], public_right_all["links"]], ignore_index=True
        ),
        "left_public": same_left_public,
        "public_right": same_public_right,
    }
    return data, matrices, links


def _draw_inter_group(
    ax: Axes,
    frame: pd.DataFrame,
    positions: dict[tuple[str, str], tuple[float, float]],
    layer_to_key: dict[str, str],
    *,
    width_min: float,
    width_max: float,
    global_min: float,
    global_max: float,
    curve_strength: float,
    arrow_end_pad: float,
) -> None:
    if frame.empty:
        return
    denominator = max(global_max - global_min, 1e-12)
    for row in frame.itertuples(index=False):
        source_key = layer_to_key[row.from_layer]
        target_key = layer_to_key[row.to_layer]
        source = positions[(source_key, row.topic)]
        target = positions[(target_key, row.topic)]
        dx = (target[0] - source[0]) * float(curve_strength)
        xs, ys = _bezier(
            source,
            (source[0] + dx, source[1]),
            (target[0] - dx, target[1]),
            target,
            samples=50,
        )
        xs, ys = _trim_curve_near_target(xs, ys, target, float(arrow_end_pad))
        if len(xs) < 2:
            continue
        scaled = (abs(float(row.rho)) - global_min) / denominator
        width = float(width_min + (width_max - width_min) * np.clip(scaled, 0.0, 1.0))
        # The Plotly notebook used width**2. This produces the same ribbon-like emphasis.
        linewidth = max(1.2, width**2 * 0.55)
        ax.plot(xs, ys, color="#7D3CFF", linewidth=linewidth, alpha=1.0, zorder=1)
        marker = ">" if target[0] >= source[0] else "<"
        ax.scatter(
            [xs[-1]],
            [ys[-1]],
            marker=marker,
            s=max(45.0, width**2 * 9.0),
            color="#7D3CFF",
            edgecolors="none",
            zorder=1.2,
        )


def plot_multilayer_topic_network(
    ax: Axes,
    config: dict[str, Any],
    year: int,
    *,
    panel_label: str | None = None,
    prepared: tuple[ThreeLayerData, dict[str, pd.DataFrame], dict[str, pd.DataFrame]] | None = None,
) -> tuple[ThreeLayerData, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Draw one year using the final notebook's multilayer visual grammar."""
    data, matrices, links = prepared if prepared is not None else _figure_2_outputs(config, year)
    topics = list(config["figure_2"]["topics"][year])
    style = config["figure_2"].get("style", {})

    layer_x = {"left": -3.6, "public": 0.0, "right": 3.6}
    ellipse_rx = float(style.get("ellipse_rx", 0.8))
    ellipse_ry = float(style.get("ellipse_ry", 2.5))
    rotation_deg = float(style.get("rotation_deg", 9.0))
    positions = _positions(
        topics,
        layer_x=layer_x,
        ellipse_rx=ellipse_rx,
        ellipse_ry=ellipse_ry,
        rotation_deg=rotation_deg,
    )
    planes = {
        layer: _plane_vertices(
            layer_x[layer],
            ellipse_rx=ellipse_rx,
            ellipse_ry=ellipse_ry,
            pad_x=float(style.get("plane_pad_x", 0.5)),
            pad_y=float(style.get("plane_pad_y", 0.1)),
            perspective=float(style.get("plane_perspective", 1.3)),
            shear=float(style.get("plane_y_shear", 0.10)),
        )
        for layer in _LAYER_KEYS
    }

    all_links = pd.concat([links["left_public"], links["public_right"]], ignore_index=True)
    abs_rho = all_links["rho"].abs() if not all_links.empty else pd.Series(dtype=float)
    global_min = float(abs_rho.min()) if not abs_rho.empty else 0.0
    global_max = float(abs_rho.max()) if not abs_rho.empty else 1.0
    layer_to_key = {data.left_label: "left", "Public Reaction": "public", data.right_label: "right"}

    # The order mirrors the Plotly trace order: far plane, links, middle plane,
    # links, near plane. It creates the same perspective occlusion.
    _draw_plane(ax, planes["left"])
    _draw_inter_group(
        ax,
        links["left_public"],
        positions,
        layer_to_key,
        width_min=float(style.get("inter_width_min", 2.5)),
        width_max=float(style.get("inter_width_max", 4.2)),
        global_min=global_min,
        global_max=global_max,
        curve_strength=float(style.get("curve_strength", 0.45)),
        arrow_end_pad=float(style.get("arrow_end_pad", 0.3)),
    )
    _draw_plane(ax, planes["public"])
    _draw_inter_group(
        ax,
        links["public_right"],
        positions,
        layer_to_key,
        width_min=float(style.get("inter_width_min", 2.5)),
        width_max=float(style.get("inter_width_max", 4.2)),
        global_min=global_min,
        global_max=global_max,
        curve_strength=float(style.get("curve_strength", 0.45)),
        arrow_end_pad=float(style.get("arrow_end_pad", 0.3)),
    )
    _draw_plane(ax, planes["right"])

    intra_thresholds = config["figure_2"]["intra_min_abs_correlation"][year]
    alpha = float(config["figure_2"]["intra_alpha"])
    intra_by_layer = {
        "left": _intra_edges(
            matrices["left_corr"], matrices["left_p"], alpha=alpha, threshold=float(intra_thresholds["candidate"])
        ),
        "public": _intra_edges(
            matrices["public_corr"], matrices["public_p"], alpha=alpha, threshold=float(intra_thresholds["public"])
        ),
        "right": _intra_edges(
            matrices["right_corr"], matrices["right_p"], alpha=alpha, threshold=float(intra_thresholds["candidate"])
        ),
    }
    for layer, frame in intra_by_layer.items():
        for row in frame.itertuples(index=False):
            start = positions[(layer, row.topic_i)]
            end = positions[(layer, row.topic_j)]
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="black",
                linewidth=float(style.get("intra_width", 1.4)),
                alpha=float(style.get("intra_alpha", 0.55)),
                zorder=3,
            )

    # Missing speech bias usually means that the public speech CSVs were built
    # before the ``label``-as-stance fallback was added.  Keep the plot usable,
    # but tell the user exactly how to regenerate the colors.
    for layer_label, layer_bias in (
        (data.left_label, data.left_bias),
        ("Public Reaction", data.public_bias),
        (data.right_label, data.right_bias),
    ):
        if pd.to_numeric(layer_bias, errors="coerce").notna().sum() == 0:
            warnings.warn(
                f"Figure 2 has no stance-bias values for {layer_label}. "
                "Re-run scripts/prepare_public_data.py so speech stance labels are rebuilt.",
                RuntimeWarning,
                stacklevel=2,
            )

    bias_values = pd.concat([data.left_bias, data.public_bias, data.right_bias], axis=0)
    finite_bias = pd.to_numeric(bias_values, errors="coerce").dropna().to_numpy(dtype=float)
    bias_absmax = float(np.max(np.abs(finite_bias))) if finite_bias.size else 1.0
    if bias_absmax <= 0:
        bias_absmax = 1.0

    candidate_sizes = {
        "left": _scale_diameters(
            data.left_volume,
            float(style.get("node_size_candidate_min", 10.0)),
            float(style.get("node_size_candidate_max", 24.0)),
        ),
        "right": _scale_diameters(
            data.right_volume,
            float(style.get("node_size_candidate_min", 10.0)),
            float(style.get("node_size_candidate_max", 24.0)),
        ),
    }
    public_sizes = _scale_diameters(
        data.public_volume,
        float(style.get("node_size_reaction_min", 10.0)),
        float(style.get("node_size_reaction_max", 24.0)),
    )
    bias_by_layer = {"left": data.left_bias, "public": data.public_bias, "right": data.right_bias}

    for layer in _LAYER_KEYS:
        for topic in topics:
            diameter = public_sizes.get(topic, 12.0) if layer == "public" else candidate_sizes[layer].get(topic, 12.0)
            color = _stance_color(bias_by_layer[layer].get(topic, np.nan), absmax=bias_absmax)
            x, y = positions[(layer, topic)]
            ax.scatter(
                [x],
                [y],
                s=diameter**2,
                color=[color],
                edgecolors="#141414",
                linewidths=float(style.get("node_edge_width", 0.9)),
                zorder=5,
            )

    # The original Plotly function labeled each topic once on the public
    # reaction layer. Repeating the same label on all three planes makes the
    # static two-panel figure unreadable, so the target layer is configurable.
    if bool(style.get("show_topic_labels", True)):
        label_layer = str(style.get("topic_label_layer", "public"))
        if label_layer not in _LAYER_KEYS:
            raise ValueError("figure_2.style.topic_label_layer must be left, public, or right")
        offset_x = float(style.get("topic_label_offset_x", 5.0))
        offset_y = float(style.get("topic_label_offset_y", 5.0))
        label_center_x = layer_x[label_layer]
        for topic in topics:
            x, y = positions[(label_layer, topic)]
            # Put labels outside the ellipse rather than over the nodes.
            direction = 1.0 if x >= label_center_x else -1.0
            ax.annotate(
                _display_topic(topic),
                xy=(x, y),
                xytext=(direction * offset_x, offset_y),
                textcoords="offset points",
                ha="left" if direction > 0 else "right",
                va="bottom",
                fontsize=float(style.get("topic_label_fontsize", 9.0)),
                color="black",
                linespacing=0.95,
                clip_on=False,
                zorder=6,
            )

    layer_titles = {
        "left": f"{data.left_label}\nSpeeches",
        "public": "Public Reaction\nLikes per post",
        "right": f"{data.right_label}\nSpeeches",
    }
    for layer in _LAYER_KEYS:
        title_y = max(y for _, y in planes[layer]) + 0.55
        ax.text(
            layer_x[layer],
            title_y,
            layer_titles[layer],
            ha="center",
            va="center",
            fontsize=float(style.get("layer_title_fontsize", 14)),
            bbox={"facecolor": "white", "edgecolor": (0, 0, 0, 0.18), "alpha": 0.88, "pad": 5},
            zorder=7,
        )

    if panel_label:
        ax.set_title(panel_label, fontsize=13, fontweight="bold", pad=14)
    all_vertices = [point for vertices in planes.values() for point in vertices]
    xs = [point[0] for point in all_vertices]
    ys = [point[1] for point in all_vertices]
    ax.set_xlim(min(xs) - 0.35, max(xs) + 0.35)
    ax.set_ylim(min(ys) - 0.35, max(ys) + 1.25)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return data, matrices, links


def _write_tables(
    output_dir: Path,
    year: int,
    data: ThreeLayerData,
    matrices: dict[str, pd.DataFrame],
    links: dict[str, pd.DataFrame],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, frame in matrices.items():
        path = output_dir / f"{year}_{name}.csv"
        frame.to_csv(path)
        written.append(path)
    for name, frame in links.items():
        if name in {"left_public", "public_right"}:
            filename = f"{year}_{name}_same_topic_links.csv"
        else:
            filename = f"{year}_{name}.csv"
        path = output_dir / filename
        frame.to_csv(path, index=False)
        written.append(path)

    rows: list[dict[str, Any]] = []
    for layer, volume, bias in (
        (data.left_label, data.left_volume, data.left_bias),
        ("Public Reaction", data.public_volume, data.public_bias),
        (data.right_label, data.right_volume, data.right_bias),
    ):
        for topic in volume.index:
            rows.append(
                {
                    "year": year,
                    "layer": layer,
                    "topic": topic,
                    "volume": float(volume.get(topic, 0.0)),
                    "stance_bias": float(bias.get(topic, np.nan)),
                }
            )
    nodes_path = output_dir / f"{year}_nodes.csv"
    pd.DataFrame(rows).to_csv(nodes_path, index=False)
    written.append(nodes_path)
    return written


def reproduce_figure_2(config: dict[str, Any]) -> list[Path]:
    output_dir = repo_path(config, "results") / "figure_2"
    data_dir = output_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    years = [int(year) for year in config["figure_2"]["years"]]
    written: list[Path] = []

    # Standalone panels preserve the dimensions of the notebook exports.
    cached: dict[int, tuple[ThreeLayerData, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]] = {}
    for year in years:
        result = _figure_2_outputs(config, year)
        cached[year] = result
        fig, ax = plt.subplots(figsize=(12.5, 7.0))
        plot_multilayer_topic_network(ax, config, year, prepared=result)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        for suffix in ("pdf", "svg", "png"):
            path = output_dir / f"figure_2_{year}.{suffix}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            written.append(path)
        plt.close(fig)
        written.extend(_write_tables(data_dir, year, *result))

    # Combined two-panel version used in the manuscript.
    fig, axes = plt.subplots(1, len(years), figsize=(22.0, 7.8))
    if len(years) == 1:
        axes = [axes]
    for index, (ax, year) in enumerate(zip(axes, years)):
        plot_multilayer_topic_network(
            ax,
            config,
            year,
            panel_label=f"({'abcdefghijklmnopqrstuvwxyz'[index]}) Support and stance profiles ({year})",
            prepared=cached[year],
        )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0.02)
    for suffix in ("pdf", "svg", "png"):
        path = output_dir / f"figure_2.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written
