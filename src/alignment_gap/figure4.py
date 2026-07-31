"""Reproduce Figure 4 as separately composable components.

The plotting functions below are ports of ``plot_support_bump_stance_v2`` and
``plot_multiple_stance_radars`` from ``06_time_series.ipynb``.  The only
substantive correction is that every topic, including Democratic concerns,
uses the manuscript-wide stance convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from .config import repo_path, year_period
from .constants import CANDIDATES_BY_YEAR
from .series import candidate_support_series, daily_stance_bias_posts

FIGURE_4_TOPIC_COLORS = {
    "Democratic concerns": "#FA9E5B",
    "Wokeness": "#486FB0",
    "Economy": "#9BD7A3",
    "Immigration": "#BC2249",
    "Abortion": "#5E4FA2",
}

def plot_support_bump_stance_v2(
    bias_df: pd.DataFrame,
    likes_df: pd.DataFrame | None,
    year: int,
    *,
    # panel 1
    poll_series: pd.Series | None = None,
    poll_df: pd.DataFrame | None = None,
    poll_label: str = "Poll margin (Trump–Harris)",
    support_title: str | None = None,
    sentiment_label_top: str = "Sentiment",

    # topics
    stance_topics: list[str] | None = None,
    topic_order: list[str] | None = None,

    # rango temporal
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,

    # selección de paneles
    panels: list[int] | None = None,   # 1=support, 2=recuadro vacío, 3=stance bias

    # columnas
    bias_date_col: str = "date",
    bias_topic_col: str = "topic",
    bias_value_col: str = "bias",

    # stance panel
    stance_lw: float = 2.0,
    stance_alpha: float = 0.95,
    stance_show_legend: bool = True,
    stance_legend_outside_left: bool = False,
    stance_legend_bbox: tuple[float, float] = (-0.20, 0.5),
    stance_legend_loc: str = "center left",
    stance_legend_ncol: int = 1,
    stance_legend_frameon: bool = True,

    # colores
    colormap_topics: dict[str, str] | None = None,
    default_cmap: str = "tab20",

    # formato / layout
    figsize: tuple[float, float] = (14.0, 9.0),
    height_ratios: tuple[float, float, float] = (1.0, 1.8, 1.8),
    top_lw: float = 2.4,
    zero_lw: float = 1.0,
    spine_lw: float = 1.2,
    tick_width: float = 1.2,
    tick_length: float = 5.0,
    panel_title_fontsize: int = 14,
    axis_label_fontsize: int = 12,
    tick_labelsize: int = 11,
    date_tick_labelsize: int = 10,
    legend_fontsize: int = 9,
    grid_alpha: float = 0.25,
    grid_lw: float = 0.8,
    fill_alpha: float = 0.22,
    xtick_rotation: float = 0,
    xtick_ha: str = "center",
    show_top_legend: bool = False,   # <- ahora default False
    rename_democratic_concerns: bool = True,
    date_fmt: str = "%Y-%m-%d",
    x_minor_ticks: bool = True,
    x_minor_locator: mdates.DateLocator | None = None,
    rotate_dates: bool = False,

    # panel 2 vacío
    panel2_title: str | None = None,
    panel2_show_frame: bool = True,
    panel2_facecolor: str = "white",
    panel2_hide_y: bool = True,

    # líneas verticales
    vline_dates: list[str | pd.Timestamp] | None = None,
    vline_color: str = "black",
    vline_ls: str = "--",
    vline_lw: float = 1.8,
    vline_alpha: float = 0.6,

    # panel 1 range
    panel1_ymin: float | None = None,
    panel1_ymax: float | None = None,

    # labels de las líneas verticales en panel 1
    vline_labels: list[str] | None = None,   # si None: a, b, c, ...
    vline_label_fontsize: int = 12,
    vline_label_xoffset_days: int = 2,
    vline_label_yfrac: float = 0.96,
):
    """
    Paneles:
      1 = support / polls
      2 = recuadro vacío (placeholder para pegar imagen después)
      3 = stance bias

    Todos los paneles comparten eje x.
    """

    # ---------- helpers ----------
    def _to_naive_ts(x):
        if x is None:
            return None
        ts = pd.to_datetime(x, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.tz_convert(None)

    def _style_axis(ax):
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=tick_labelsize,
            width=tick_width,
            length=tick_length,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            width=max(0.8, tick_width * 0.8),
            length=max(3.0, tick_length * 0.55),
        )
        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_linewidth(spine_lw)

    def _parse_vline_dates(vals):
        if vals is None:
            return []
        out = []
        for x in vals:
            ts = _to_naive_ts(x)
            if ts is not None:
                out.append(ts)
        return out
    
    def _default_vline_labels(n: int) -> list[str]:
        labels = []
        for i in range(n):
            s = ""
            x = i
            while True:
                s = chr(ord("a") + (x % 26)) + s
                x = x // 26 - 1
                if x < 0:
                    break
            labels.append(s)
        return labels
    
    topic_rename_map = (
        {"Parties, leadership and democratic responsibility": "Democratic concerns"}
        if rename_democratic_concerns
        else {}
    )
    inv_topic_rename_map = {v: k for k, v in topic_rename_map.items()} if topic_rename_map else {}

    def _map_topics_list(lst):
        if lst is None:
            return None
        return [topic_rename_map.get(t, t) for t in lst]

    def _resolve_color(topic: str, fallback_colors: dict[str, tuple]):
        if colormap_topics:
            if topic in colormap_topics:
                return colormap_topics[topic]
            orig = inv_topic_rename_map.get(topic)
            if orig and orig in colormap_topics:
                return colormap_topics[orig]
        return fallback_colors[topic]

    # ---------- panels ----------
    if panels is None:
        panels = [1, 2, 3]

    valid_panels = {1, 2, 3}
    if any(p not in valid_panels for p in panels):
        raise ValueError("`panels` solo puede contener valores entre 1 y 3.")

    seen = set()
    panels = [p for p in panels if not (p in seen or seen.add(p))]

    stance_topics = _map_topics_list(stance_topics)
    topic_order = _map_topics_list(topic_order)

    # ---------- prepare bias_df ----------
    dfb = bias_df.copy()
    required_bias = {bias_date_col, bias_topic_col, bias_value_col}
    if not required_bias.issubset(dfb.columns):
        raise ValueError(f"bias_df debe tener columnas: {sorted(required_bias)}")

    dfb[bias_date_col] = pd.to_datetime(dfb[bias_date_col], errors="coerce", utc=True).dt.tz_convert(None)
    dfb[bias_value_col] = pd.to_numeric(dfb[bias_value_col], errors="coerce")
    dfb = dfb.dropna(subset=[bias_topic_col, bias_date_col, bias_value_col])

    if topic_rename_map:
        dfb[bias_topic_col] = dfb[bias_topic_col].replace(topic_rename_map)

    start = _to_naive_ts(start_date)
    end = _to_naive_ts(end_date)
    if start is not None:
        dfb = dfb[dfb[bias_date_col] >= start]
    if end is not None:
        dfb = dfb[dfb[bias_date_col] <= end]

    dfb = (
        dfb.groupby([bias_topic_col, bias_date_col], as_index=False)[bias_value_col]
        .mean()
    )

    if dfb.empty:
        raise ValueError("No hay datos en bias_df luego de aplicar filtros.")

    # ---------- prepare support ----------
    ps = None
    pdf = None

    if poll_df is not None and len(poll_df):
        pdf = poll_df.copy()
        pdf.index = pd.to_datetime(pdf.index, errors="coerce", utc=True).tz_convert(None)
        pdf = pdf.sort_index()
        if start is not None:
            pdf = pdf[pdf.index >= start]
        if end is not None:
            pdf = pdf[pdf.index <= end]
    elif poll_series is not None and len(poll_series):
        ps = poll_series.copy()
        ps.index = pd.to_datetime(ps.index, errors="coerce", utc=True).tz_convert(None)
        ps = ps.sort_index()
        ps = pd.to_numeric(ps, errors="coerce").dropna()
        if start is not None:
            ps = ps[ps.index >= start]
        if end is not None:
            ps = ps[ps.index <= end]

    if 1 in panels and (pdf is None and ps is None):
        raise ValueError("Pediste panel 1 pero no hay poll_df ni poll_series disponibles tras los filtros.")

    # ---------- topic sets ----------
    available_bias_topics = sorted(dfb[bias_topic_col].dropna().unique().tolist())

    if topic_order is not None:
        global_order = topic_order.copy()
    else:
        tmp = dfb.groupby(bias_topic_col)[bias_value_col].mean().sort_values(ascending=False)
        global_order = tmp.index.tolist()

    fallback_cmap_obj = plt.get_cmap(default_cmap)
    fallback_colors = {t: fallback_cmap_obj(i % fallback_cmap_obj.N) for i, t in enumerate(global_order)}
    color_map = {t: _resolve_color(t, fallback_colors) for t in global_order}

    if stance_topics is None:
        stance_topics_use = [t for t in global_order if t in available_bias_topics]
    else:
        stance_topics_use = [t for t in stance_topics if t in available_bias_topics]

    if 3 in panels and len(stance_topics_use) == 0:
        raise ValueError("Pediste panel 3 pero no hay stance_topics disponibles tras los filtros.")

    # ---------- figure ----------
    selected_height_ratios = [height_ratios[p - 1] for p in panels]

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=figsize,
        sharex=True,   # <- ahora todos comparten x
        gridspec_kw={"height_ratios": selected_height_ratios},
    )

    if len(panels) == 1:
        axes = [axes]
    else:
        axes = list(np.ravel(axes))

    panel_to_ax = {p: ax for p, ax in zip(panels, axes)}

    # ===== Panel 1 =====
    if 1 in panel_to_ax:
        ax1 = panel_to_ax[1]

        if pdf is not None and len(pdf):
            req = {"fav_trump_share", "fav_kamala_share"}
            if not req.issubset(pdf.columns):
                raise KeyError("poll_df debe tener columnas {'fav_trump_share','fav_kamala_share'}")

            ax1.plot(
                pdf.index, pdf["fav_trump_share"].astype(float),
                color="black", lw=top_lw, label="Fav. Trump share"
            )
            ax1.plot(
                pdf.index, pdf["fav_kamala_share"].astype(float),
                color="gray", lw=top_lw, ls="--", label="Fav. Kamala share"
            )
            ax1.set_ylabel(sentiment_label_top, fontsize=axis_label_fontsize)
            ax1.set_title("Sentiment / Polls", fontsize=panel_title_fontsize)
            if show_top_legend:
                ax1.legend(loc="lower left", fontsize=legend_fontsize, frameon=False)
            ax1.grid(axis="y", linestyle="--", alpha=grid_alpha, linewidth=grid_lw)

        elif ps is not None and len(ps):
            x = ps.index
            y = ps.values.astype(float)
            ax1.set_title(support_title or "Harris - Trump support difference", fontsize=panel_title_fontsize)
            ax1.fill_between(x, 0, y, where=(y <= 0), color="red", alpha=fill_alpha)
            ax1.fill_between(x, 0, y, where=(y >= 0), color="blue", alpha=fill_alpha)
            ax1.plot(x, y, color="black", lw=top_lw, label=poll_label)
            ax1.axhline(0, color="black", lw=zero_lw, ls="--", alpha=0.6)
            ax1.set_ylabel("", fontsize=axis_label_fontsize)
            if show_top_legend:
                ax1.legend(loc="lower left", fontsize=legend_fontsize, frameon=False)
            ax1.grid(axis="y", linestyle="--", alpha=grid_alpha, linewidth=grid_lw)
        
        # rango vertical opcional del panel 1
        if panel1_ymin is not None or panel1_ymax is not None:
            y0, y1 = ax1.get_ylim()
            ax1.set_ylim(
                panel1_ymin if panel1_ymin is not None else y0,
                panel1_ymax if panel1_ymax is not None else y1,
            )
    # ===== Panel 2 =====
    if 2 in panel_to_ax:
        ax2 = panel_to_ax[2]

        if panel2_title is not None:
            ax2.set_title(panel2_title, fontsize=panel_title_fontsize)

        ax2.set_facecolor(panel2_facecolor)

        if panel2_hide_y:
            ax2.set_yticks([])
            ax2.tick_params(axis="y", left=False, labelleft=False)

        if not panel2_show_frame:
            for side in ["top", "right", "bottom", "left"]:
                ax2.spines[side].set_visible(False)

        ax2.grid(False)

    # ===== Panel 3 =====
    if 3 in panel_to_ax:
        ax3 = panel_to_ax[3]

        for t in stance_topics_use:
            dft = dfb[dfb[bias_topic_col] == t].sort_values(bias_date_col)
            if dft.empty:
                continue

            ax3.plot(
                dft[bias_date_col],
                dft[bias_value_col].astype(float),
                color=color_map[t],
                lw=stance_lw,
                alpha=stance_alpha,
                label=t,
            )
        
        ax3.set_ylim([-1,1])

        ax3.set_title("Stance bias", fontsize=panel_title_fontsize)
        ax3.axhline(0, ls="--", alpha=0.35, color="black", lw=zero_lw)
        ax3.set_ylabel("Stance bias", fontsize=axis_label_fontsize)
        ax3.grid(axis="both", linestyle="--", alpha=grid_alpha, linewidth=grid_lw)

        if stance_show_legend:
            if stance_legend_outside_left:
                ax3.legend(
                    ncol=stance_legend_ncol,
                    fontsize=legend_fontsize,
                    frameon=stance_legend_frameon,
                    loc=stance_legend_loc,
                    bbox_to_anchor=stance_legend_bbox,
                    borderaxespad=0.0,
                )
            else:
                ax3.legend(
                    ncol=stance_legend_ncol,
                    fontsize=legend_fontsize,
                    frameon=stance_legend_frameon,
                    loc="upper right",
                )

    # ----- style axes -----
    for ax in axes:
        _style_axis(ax)

    if 2 in panel_to_ax and panel2_hide_y:
        panel_to_ax[2].tick_params(axis="y", left=False, labelleft=False)

    # ----- x-range shared -----
    xmins, xmaxs = [], []
    xmins.append(dfb[bias_date_col].min())
    xmaxs.append(dfb[bias_date_col].max())

    if ps is not None and len(ps):
        xmins.append(ps.index.min())
        xmaxs.append(ps.index.max())
    if pdf is not None and len(pdf):
        xmins.append(pdf.index.min())
        xmaxs.append(pdf.index.max())

    xmin, xmax = min(xmins), max(xmaxs)
    for ax in axes:
        ax.set_xlim(xmin, xmax)

    # ----- vertical dashed lines -----
    vlines = _parse_vline_dates(vline_dates)

    for ax in axes:
        for dt in vlines:
            ax.axvline(
                dt,
                color=vline_color,
                linestyle=vline_ls,
                linewidth=vline_lw,
                alpha=vline_alpha,
                zorder=0,
            )

    # labels de las líneas SOLO en panel 1
    if 1 in panel_to_ax and len(vlines) > 0:
        ax1 = panel_to_ax[1]

        if vline_labels is None:
            labels_use = _default_vline_labels(len(vlines))
        else:
            if len(vline_labels) != len(vlines):
                raise ValueError("`vline_labels` debe tener la misma longitud que `vline_dates`.")
            labels_use = list(vline_labels)

        y0, y1 = ax1.get_ylim()
        y_text = y0 + vline_label_yfrac * (y1 - y0)

        for dt, lab in zip(vlines, labels_use):
            ax1.text(
                dt + pd.Timedelta(days=vline_label_xoffset_days),
                y_text,
                str(lab),
                ha="left",
                va="top",
                fontsize=vline_label_fontsize,
                color=vline_color,
                clip_on=True,
            )

    # ----- date formatting on bottom panel only -----
    bottom_ax = axes[-1]
    bottom_ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=[1, 15]))
    bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))

    if x_minor_ticks:
        if x_minor_locator is not None:
            bottom_ax.xaxis.set_minor_locator(x_minor_locator)
        else:
            bottom_ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=[8, 22]))

    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)

    bottom_ax.tick_params(axis="x", labelbottom=True, bottom=True)

    plt.setp(
        bottom_ax.get_xticklabels(),
        rotation=(xtick_rotation if rotate_dates else 0),
        ha=(xtick_ha if rotate_dates else "center"),
        fontsize=date_tick_labelsize,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = {
        "bias_df_plot": dfb,
        "stance_topics_use": stance_topics_use,
        "vline_dates_used": vlines,
    }
    return fig, out


def summarize_topic_stance_for_radar(
    bias_df: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    topics: list[str] | None = None,
    topic_order: list[str] | None = None,
    topic_col: str = "topic",
    date_col: str = "date",
    likes_pro_col: str = "likes_pro",
    likes_anti_col: str = "likes_anti",
    likes_total_col: str = "likes_total",
    rename_democratic_concerns: bool = True,
    exclude_topics: tuple[str, ...] = ("Not specified",),
) -> pd.DataFrame:
    """
    Resume por tópico, en un rango de fechas:
      - likes_pro_sum
      - likes_anti_sum
      - likes_total_sum
      - stance_bias = (pro - anti) / total
    """
    df = bias_df.copy()

    required = {topic_col, date_col, likes_pro_col, likes_anti_col, likes_total_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en bias_df: {sorted(missing)}")

    if rename_democratic_concerns:
        df[topic_col] = df[topic_col].replace(
            {"Parties, leadership and democratic responsibility": "Democratic concerns"}
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert(None)
    df[likes_pro_col] = pd.to_numeric(df[likes_pro_col], errors="coerce")
    df[likes_anti_col] = pd.to_numeric(df[likes_anti_col], errors="coerce")
    df[likes_total_col] = pd.to_numeric(df[likes_total_col], errors="coerce")
    df = df.dropna(subset=[topic_col, date_col, likes_pro_col, likes_anti_col, likes_total_col])

    if start_date is not None:
        start = pd.to_datetime(start_date, utc=True, errors="coerce")
        if pd.notna(start):
            start = start.tz_convert(None)
            df = df[df[date_col] >= start]

    if end_date is not None:
        end = pd.to_datetime(end_date, utc=True, errors="coerce")
        if pd.notna(end):
            end = end.tz_convert(None)
            df = df[df[date_col] <= end]

    exclude_lower = {str(x).strip().lower() for x in exclude_topics}
    df = df[~df[topic_col].astype(str).str.strip().str.lower().isin(exclude_lower)]

    if topics is not None:
        df = df[df[topic_col].isin(topics)]

    if df.empty:
        raise ValueError("No quedaron datos luego de filtrar por fechas/topics.")

    summary = (
        df.groupby(topic_col, as_index=False)
        .agg(
            likes_pro_sum=(likes_pro_col, "sum"),
            likes_anti_sum=(likes_anti_col, "sum"),
            likes_total_sum=(likes_total_col, "sum"),
        )
    )

    summary["stance_bias"] = np.where(
        summary["likes_total_sum"] > 0,
        (summary["likes_pro_sum"] - summary["likes_anti_sum"]) / summary["likes_total_sum"],
        np.nan,
    )

    if topic_order is not None:
        order_map = {t: i for i, t in enumerate(topic_order)}
        summary["__ord"] = summary[topic_col].map(order_map)
        summary = summary.sort_values("__ord", na_position="last").drop(columns="__ord")
    else:
        summary = summary.sort_values("likes_total_sum", ascending=False)

    return summary.reset_index(drop=True)


def _default_topic_initials(topics: list[str]) -> dict[str, str]:
    """
    Genera iniciales simples:
      - Democratic concerns -> DC
      - Wokeness -> W
      - Economy -> E
      - Immigration -> I
    """
    out = {}
    used = set()

    for topic in topics:
        words = [w for w in str(topic).replace("-", " ").split() if w]
        if len(words) >= 2:
            cand = "".join(w[0].upper() for w in words[:2])
        else:
            cand = words[0][0].upper()

        # desambiguación mínima si se repite
        if cand in used:
            extra = "".join(w[0].upper() for w in words)
            cand = extra[: max(2, len(cand) + 1)]
            k = 2
            while cand in used:
                cand = f"{extra[:2]}{k}"
                k += 1

        out[topic] = cand
        used.add(cand)

    return out


def plot_single_stance_radar(
    summary: pd.DataFrame,
    *,
    ax=None,
    topic_col: str = "topic",
    stance_col: str = "stance_bias",
    title: str | None = None,
    topic_initials_map: dict[str, str] | None = None,
    pro_color: str = "#4C78A8",
    anti_color: str = "#E45756",
    stance_bar_alpha: float = 0.85,
    stance_bar_width_frac: float = 0.72,
    stance_rmax: float = 1.0,
    stance_rticks: list[float] | None = None,
    title_fontsize: int = 15,
    topic_label_fontsize: int = 14,
    radial_tick_fontsize: int = 10,
    bias_value_fontsize: int = 10,
    show_bias_value_labels: bool = True,
    bias_value_color: str = "black",
):
    """
    Grafica un radar polar de stance bias.
    Cada tópico ocupa un ángulo, la magnitud radial es abs(SB) y el color indica el signo.
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    else:
        fig = ax.figure

    if summary.empty:
        raise ValueError("summary está vacío.")

    labels = summary[topic_col].tolist()
    bias_vals = summary[stance_col].to_numpy(dtype=float)

    if topic_initials_map is None:
        topic_initials_map = _default_topic_initials(labels)

    initials = [topic_initials_map.get(t, t) for t in labels]

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = (2 * np.pi / n) * stance_bar_width_frac

    # Real figure windows contain observed engagement for every selected topic.
    # Keep the plotting helper robust for sparse test/subsample windows by
    # rendering undefined values as zero-height bars.
    bias_vals_plot = np.nan_to_num(bias_vals, nan=0.0)
    bias_mag = np.abs(bias_vals_plot)
    bias_colors = [pro_color if v >= 0 else anti_color for v in bias_vals_plot]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.bar(
        angles,
        bias_mag,
        width=width,
        bottom=0.0,
        color=bias_colors,
        alpha=stance_bar_alpha,
        edgecolor="white",
        linewidth=1.0,
    )

    ax.set_xticks(angles)
    ax.set_xticklabels(initials, fontsize=topic_label_fontsize)

    if stance_rticks is None:
        #stance_rticks = [0.25, 0.50, 0.75, 1.00]
        stance_rticks = [0.25, 0.50, 0.75, 1.00]

    ax.set_ylim(0, stance_rmax)
    ax.set_yticks(stance_rticks, labels=["","","",""])
    #ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.2f}"))
    ax.tick_params(axis="y", labelsize=radial_tick_fontsize)

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize, pad=20)

    if show_bias_value_labels:
        for ang, mag, val in zip(angles, bias_mag, bias_vals):
            r_text = min(stance_rmax, mag + 0.2)
            ax.text(
                ang,
                r_text,
                f"{val:+.2f}",
                ha="center",
                va="center",
                fontsize=bias_value_fontsize,
                color=bias_value_color,
            )

    return fig, ax, topic_initials_map


def plot_multiple_stance_radars(
    bias_df: pd.DataFrame,
    *,
    date_windows: list,
    topics: list[str] | None = None,
    topic_order: list[str] | None = None,

    topic_col: str = "topic",
    date_col: str = "date",
    likes_pro_col: str = "likes_pro",
    likes_anti_col: str = "likes_anti",
    likes_total_col: str = "likes_total",

    rename_democratic_concerns: bool = True,
    exclude_topics: tuple[str, ...] = ("Not specified",),

    # initials
    topic_initials_map: dict[str, str] | None = None,

    # radar style
    figsize: tuple[float, float] = (15, 5.8),
    title_fontsize: int = 15,
    topic_label_fontsize: int = 15,
    radial_tick_fontsize: int = 10,
    bias_value_fontsize: int = 10,
    show_bias_value_labels: bool = True,
    bias_value_color: str = "black",
    pro_color: str = "#4C78A8",
    anti_color: str = "#E45756",
    stance_bar_alpha: float = 0.85,
    stance_bar_width_frac: float = 0.72,
    stance_rmax: float = 1.0,
    stance_rticks: list[float] | None = None,

    # legends below
    initials_legend_fontsize: int = 13,
    sign_legend_fontsize: int = 14,
    initials_legend_y: float = 0.08,
    sign_legend_y: float = 0.02,
):
    """
    Hace varios radares de stance bias lado a lado.

    date_windows puede ser una lista de:
      - (start_date, end_date)
      - (start_date, end_date, title)
      - {"start_date":..., "end_date":..., "title":...}
    """
    if not date_windows:
        raise ValueError("date_windows no puede estar vacío.")

    def _parse_window(w):
        if isinstance(w, dict):
            return (
                w.get("start_date"),
                w.get("end_date"),
                w.get("title"),
            )
        if isinstance(w, (list, tuple)):
            if len(w) == 2:
                return w[0], w[1], None
            if len(w) == 3:
                return w[0], w[1], w[2]
        raise ValueError(
            "Cada elemento de date_windows debe ser "
            "(start_date, end_date), (start_date, end_date, title) o un dict."
        )

    parsed_windows = [_parse_window(w) for w in date_windows]

    # construir initials globales una sola vez
    if topics is not None:
        topics_for_initials = topics.copy()
    else:
        tmp = bias_df.copy()
        if rename_democratic_concerns:
            tmp[topic_col] = tmp[topic_col].replace(
                {"Parties, leadership and democratic responsibility": "Democratic concerns"}
            )
        exclude_lower = {str(x).strip().lower() for x in exclude_topics}
        tmp = tmp[~tmp[topic_col].astype(str).str.strip().str.lower().isin(exclude_lower)]
        topics_for_initials = (
            topic_order if topic_order is not None
            else sorted(tmp[topic_col].dropna().unique().tolist())
        )

    if topic_initials_map is None:
        topic_initials_map = _default_topic_initials(topics_for_initials)

    n = len(parsed_windows)
    fig, axes = plt.subplots(
        1, n,
        figsize=figsize,
        subplot_kw={"projection": "polar"},
    )

    if n == 1:
        axes = [axes]

    summaries = []

    for ax, (start_date, end_date, title) in zip(axes, parsed_windows):
        summary = summarize_topic_stance_for_radar(
            bias_df,
            start_date=start_date,
            end_date=end_date,
            topics=topics,
            topic_order=topic_order,
            topic_col=topic_col,
            date_col=date_col,
            likes_pro_col=likes_pro_col,
            likes_anti_col=likes_anti_col,
            likes_total_col=likes_total_col,
            rename_democratic_concerns=rename_democratic_concerns,
            exclude_topics=exclude_topics,
        )

        if title is None:
            title = f"{pd.to_datetime(start_date).date()} to {pd.to_datetime(end_date).date()}"

        plot_single_stance_radar(
            summary,
            ax=ax,
            topic_col=topic_col,
            stance_col="stance_bias",
            title=title,
            topic_initials_map=topic_initials_map,
            pro_color=pro_color,
            anti_color=anti_color,
            stance_bar_alpha=stance_bar_alpha,
            stance_bar_width_frac=stance_bar_width_frac,
            stance_rmax=stance_rmax,
            stance_rticks=stance_rticks,
            title_fontsize=title_fontsize,
            topic_label_fontsize=topic_label_fontsize,
            radial_tick_fontsize=radial_tick_fontsize,
            bias_value_fontsize=bias_value_fontsize,
            show_bias_value_labels=show_bias_value_labels,
            bias_value_color=bias_value_color,
        )

        summary = summary.copy()
        summary["window_start"] = start_date
        summary["window_end"] = end_date
        summary["window_title"] = title
        summaries.append(summary)

    # leyenda de initials abajo
    initials_lines = [f"{topic_initials_map.get(t, t)} = {t}" for t in topics_for_initials if t in topic_initials_map]
    initials_text = "   |   ".join(initials_lines)

    fig.text(
        0.5,
        initials_legend_y,
        initials_text,
        ha="center",
        va="center",
        fontsize=initials_legend_fontsize,
    )

    # leyenda de signos abajo
    handles = [
        Patch(facecolor=pro_color, edgecolor="none", label="SB > 0"),
        Patch(facecolor=anti_color, edgecolor="none", label="SB < 0"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, sign_legend_y),
        ncol=2,
        frameon=False,
        fontsize=sign_legend_fontsize,
    )

    fig.tight_layout(rect=(0, 0.12, 1, 1))

    summary_all = pd.concat(summaries, ignore_index=True)
    return fig, summary_all, topic_initials_map


def _year_entry(mapping: dict, year: int) -> Any:
    """Read a year-keyed YAML mapping regardless of int/string key parsing."""
    if year in mapping:
        return mapping[year]
    return mapping[str(year)]


def _plot_period(config: dict[str, Any], year: int) -> tuple[str, str]:
    figure_config = config["figure_4"]
    periods = figure_config.get("support_periods", {})
    if year in periods or str(year) in periods:
        entry = _year_entry(periods, year)
        return str(entry["start"]), str(entry["end"])
    return year_period(config, year)


def _optional_ylim(config: dict[str, Any], year: int) -> tuple[float | None, float | None]:
    mapping = config["figure_4"].get("support_ylim", {})
    if year not in mapping and str(year) not in mapping:
        return None, None
    value = _year_entry(mapping, year)
    if value is None:
        return None, None
    return (
        None if value[0] is None else float(value[0]),
        None if value[1] is None else float(value[1]),
    )


def _save_figure(fig, output_dir: Path, stem: str, formats: list[str]) -> list[Path]:
    written: list[Path] = []
    for suffix in formats:
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def reproduce_figure_4(config: dict[str, Any]) -> list[Path]:
    """Write the time-series and radar components separately for each year.

    Outputs are intentionally not assembled into a single Figure 4 because the
    manuscript panel was composed externally.  Democratic concerns follows the
    manuscript convention throughout: positive means ``Republicans threaten
    democracy`` and negative means ``Democrats threaten democracy``.
    """
    output_dir = repo_path(config, "results") / "figure_4"
    data_dir = output_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    figure_config = config["figure_4"]
    support_topics = list(figure_config["support_topics"])
    radar_topics = list(figure_config["radar_topics"])
    all_topics = list(dict.fromkeys(support_topics + radar_topics))
    rolling_days = int(figure_config["centered_window_days"])
    formats = [str(value) for value in figure_config.get("static_formats", ["png", "pdf", "svg"])]
    written: list[Path] = []

    support_style = figure_config.get("support_style", {})
    radar_style = figure_config.get("radar_style", {})
    vline_mapping = figure_config.get("vline_dates", {})

    for raw_year in figure_config["years"]:
        year = int(raw_year)
        start, end = _plot_period(config, year)
        support = candidate_support_series(
            config,
            year,
            start=start,
            end=end,
            rolling_days=rolling_days,
        )
        # No Figure-4-specific polarity override: this is the same convention
        # used by the manuscript and by the other figures.
        bias = daily_stance_bias_posts(
            config,
            year,
            start=start,
            end=end,
            rolling_days=rolling_days,
        )
        bias = bias[bias["topic"].isin(all_topics)].copy()

        support_path = data_dir / f"candidate_support_{year}.csv"
        bias_path = data_dir / f"stance_bias_{year}.csv"
        support.to_csv(support_path, index=False)
        bias.to_csv(bias_path, index=False)
        written.extend([support_path, bias_path])

        democrat = CANDIDATES_BY_YEAR[year]["democrat"].split()[-1]
        ymin, ymax = _optional_ylim(config, year)
        vlines = list(_year_entry(vline_mapping, year)) if (year in vline_mapping or str(year) in vline_mapping) else []
        support_series = support.set_index("date")["support_difference"]

        support_fig, _ = plot_support_bump_stance_v2(
            bias_df=bias,
            likes_df=None,
            year=year,
            poll_series=support_series,
            poll_label=f"{democrat} - Trump support difference",
            support_title=f"{democrat} - Trump support difference",
            stance_topics=support_topics,
            topic_order=support_topics,
            start_date=start,
            end_date=end,
            panels=[1, 3],
            colormap_topics=FIGURE_4_TOPIC_COLORS,
            figsize=tuple(support_style.get("figsize", [16, 9])),
            panel_title_fontsize=int(support_style.get("panel_title_fontsize", 28)),
            axis_label_fontsize=int(support_style.get("axis_label_fontsize", 22)),
            tick_labelsize=int(support_style.get("tick_labelsize", 20)),
            date_tick_labelsize=int(support_style.get("date_tick_labelsize", 20)),
            legend_fontsize=int(support_style.get("legend_fontsize", 14)),
            show_top_legend=False,
            vline_dates=vlines,
            vline_label_fontsize=int(support_style.get("vline_label_fontsize", 16)),
            panel1_ymin=ymin,
            panel1_ymax=ymax,
            vline_lw=float(support_style.get("vline_lw", 2.5)),
            date_fmt=str(support_style.get("date_fmt", "%m-%d")),
        )
        written.extend(
            _save_figure(
                support_fig,
                output_dir,
                f"figure_4_support_stance_{year}",
                formats,
            )
        )

        windows = list(_year_entry(figure_config["radar_windows"], year))
        radar_fig, radar_summary, _ = plot_multiple_stance_radars(
            bias,
            topics=radar_topics,
            topic_order=radar_topics,
            date_windows=windows,
            figsize=tuple(radar_style.get("figsize", [15, 5.8])),
            topic_label_fontsize=int(radar_style.get("topic_label_fontsize", 20)),
            bias_value_fontsize=int(radar_style.get("bias_value_fontsize", 16)),
            title_fontsize=int(radar_style.get("title_fontsize", 24)),
            initials_legend_fontsize=int(radar_style.get("initials_legend_fontsize", 18)),
            sign_legend_fontsize=int(radar_style.get("sign_legend_fontsize", 20)),
            sign_legend_y=float(radar_style.get("sign_legend_y", 0.10)),
            initials_legend_y=float(radar_style.get("initials_legend_y", 0.02)),
            show_bias_value_labels=bool(radar_style.get("show_bias_value_labels", False)),
        )
        radar_summary["year"] = year
        radar_path = data_dir / f"radar_profiles_{year}.csv"
        radar_summary.to_csv(radar_path, index=False)
        written.append(radar_path)
        written.extend(
            _save_figure(
                radar_fig,
                output_dir,
                f"figure_4_radars_{year}",
                formats,
            )
        )

    return written
