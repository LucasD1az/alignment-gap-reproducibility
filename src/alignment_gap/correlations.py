"""Correlation and lag-selection routines shared by Figures 2 and 3.

Two lag-alignment modes are supported:

``positional``
    A conventional positional shift. This is the statistically natural
    implementation and remains available for new analyses.

``legacy_index``
    Reproduces the exact pandas slicing/alignment behavior used by
    ``export_three_layer_correlations`` in ``07_correlations.ipynb``. The
    sliced Series retain their DatetimeIndex and ``pd.concat`` aligns them by
    date before dropping missing values. This mode is used by the paper
    reproduction pipeline because it recreates the matrices and links that
    fed the original Figure 2 and Figure 3 notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

LagAlignment = Literal["positional", "legacy_index"]


@dataclass(frozen=True)
class LagResult:
    rho: float
    p: float
    lag: int
    n: int
    selected_from: str


def _validate_alignment(alignment: str) -> LagAlignment:
    if alignment not in {"positional", "legacy_index"}:
        raise ValueError("alignment must be 'positional' or 'legacy_index'")
    return alignment  # type: ignore[return-value]


def spearman_at_lag(
    x: pd.Series,
    y: pd.Series,
    lag: int,
    *,
    alignment: LagAlignment = "positional",
) -> tuple[float, float, int]:
    """Compute Spearman correlation at one non-zero or zero lag.

    The sign convention follows the notebooks: ``lag < 0`` means that ``x``
    leads ``y``; ``lag > 0`` means that ``y`` leads ``x``.

    ``legacy_index`` deliberately mirrors the original notebook code rather
    than silently replacing it with a positional shift. This matters when the
    speech layers contain only dates on which a candidate delivered a speech.
    """
    alignment = _validate_alignment(alignment)

    if alignment == "legacy_index":
        x_num = pd.to_numeric(x, errors="coerce")
        y_num = pd.to_numeric(y, errors="coerce")
        if lag > 0:
            x2, y2 = x_num.iloc[lag:], y_num.iloc[:-lag]
        elif lag < 0:
            x2, y2 = x_num.iloc[:lag], y_num.iloc[-lag:]
        else:
            x2, y2 = x_num, y_num

        # Exact effective behavior of the notebook: retain original indices
        # after positional slicing, align on their date intersection, and then
        # drop non-finite pairs. Avoiding a fresh two-column DataFrame for
        # every topic/lag makes the full all-pairs export substantially faster.
        common = x2.index.intersection(y2.index, sort=False)
        if len(common) == 0:
            return np.nan, np.nan, 0
        xv = x2.reindex(common).to_numpy(dtype=float)
        yv = y2.reindex(common).to_numpy(dtype=float)
        mask = np.isfinite(xv) & np.isfinite(yv)
        n = int(mask.sum())
        if n < 3 or np.nanstd(xv[mask]) == 0 or np.nanstd(yv[mask]) == 0:
            return np.nan, np.nan, n
        rho, p = spearmanr(xv[mask], yv[mask])
        if not np.isfinite(rho):
            return np.nan, np.nan, n
        return float(rho), float(p), n

    xv = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    yv = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    n_common = min(len(xv), len(yv))
    xv, yv = xv[:n_common], yv[:n_common]
    if lag > 0:
        xv, yv = xv[lag:], yv[:-lag]
    elif lag < 0:
        xv, yv = xv[:lag], yv[-lag:]
    mask = np.isfinite(xv) & np.isfinite(yv)
    n = int(mask.sum())
    if n < 3 or np.nanstd(xv[mask]) == 0 or np.nanstd(yv[mask]) == 0:
        return np.nan, np.nan, n
    rho, p = spearmanr(xv[mask], yv[mask])
    return float(rho), float(p), n


def best_lag(
    x: pd.Series,
    y: pd.Series,
    *,
    alpha: float,
    lag_min: int = 1,
    lag_max: int = 14,
    alignment: LagAlignment = "positional",
    fallback_to_all_lags: bool = True,
) -> LagResult | None:
    """Select the lag using the rule from ``export_three_layer_correlations``.

    The notebook first selected the largest absolute correlation among lags
    with ``p < alpha``. When no lag was significant, it fell back to the
    largest absolute correlation among every evaluated lag. Set
    ``fallback_to_all_lags=False`` to retain only significant links.
    """
    candidates: list[dict[str, Any]] = []
    for lag in list(range(-lag_max, -lag_min + 1)) + list(range(lag_min, lag_max + 1)):
        rho, p, n = spearman_at_lag(x, y, lag, alignment=alignment)
        if not np.isfinite(rho):
            continue
        candidates.append({"rho": rho, "p": p, "lag": lag, "n": n, "abs_rho": abs(rho)})
    if not candidates:
        return None

    frame = pd.DataFrame(candidates)
    significant = frame[frame["p"] < alpha]
    if not significant.empty:
        pool = significant
        selected_from = "significant"
    elif fallback_to_all_lags:
        pool = frame
        selected_from = "fallback_all_lags"
    else:
        return None

    best = pool.sort_values(["abs_rho", "p", "n"], ascending=[False, True, False]).iloc[0]
    return LagResult(
        rho=float(best["rho"]),
        p=float(best["p"]),
        lag=int(best["lag"]),
        n=int(best["n"]),
        selected_from=selected_from,
    )


def intra_correlation_matrices(
    wide: pd.DataFrame,
    *,
    alignment: LagAlignment = "positional",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    topics = list(wide.columns)
    corr = pd.DataFrame(np.eye(len(topics)), index=topics, columns=topics, dtype=float)
    pvals = pd.DataFrame(np.zeros((len(topics), len(topics))), index=topics, columns=topics, dtype=float)
    for i, topic_i in enumerate(topics):
        for j in range(i + 1, len(topics)):
            topic_j = topics[j]
            rho, p, _ = spearman_at_lag(wide[topic_i], wide[topic_j], 0, alignment=alignment)
            corr.loc[topic_i, topic_j] = corr.loc[topic_j, topic_i] = rho
            pvals.loc[topic_i, topic_j] = pvals.loc[topic_j, topic_i] = p
    return corr, pvals



def intra_links_from_matrices(
    corr: pd.DataFrame,
    pvals: pd.DataFrame,
    *,
    layer_key: str,
    layer_label: str,
) -> pd.DataFrame:
    """Return every upper-triangle intra-layer pair, without filtering."""
    records: list[dict[str, Any]] = []
    topics = list(corr.columns)
    for i, topic_i in enumerate(topics):
        for topic_j in topics[i + 1 :]:
            rho = corr.loc[topic_i, topic_j]
            p = pvals.loc[topic_i, topic_j]
            records.append(
                {
                    "layer_key": layer_key,
                    "layer_label": layer_label,
                    "topic_i": topic_i,
                    "topic_j": topic_j,
                    "rho": float(rho) if pd.notna(rho) else np.nan,
                    "p": float(p) if pd.notna(p) else np.nan,
                    "lag": 0,
                }
            )
    return pd.DataFrame(records)

def inter_correlation_outputs(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    alpha: float,
    lag_min: int,
    lag_max: int,
    left_label: str,
    right_label: str,
    pair_key: str | None = None,
    pair_label: str | None = None,
    alignment: LagAlignment = "positional",
    fallback_to_all_lags: bool = True,
) -> dict[str, pd.DataFrame]:
    """Build the full all-topic-pairs inter-layer export used by the notebook."""
    rho = pd.DataFrame(index=left.columns, columns=right.columns, dtype=float)
    pvals = pd.DataFrame(index=left.columns, columns=right.columns, dtype=float)
    lags = pd.DataFrame(index=left.columns, columns=right.columns, dtype=float)
    ns = pd.DataFrame(index=left.columns, columns=right.columns, dtype=float)
    selected = pd.DataFrame(index=left.columns, columns=right.columns, dtype="object")
    records: list[dict[str, Any]] = []

    for left_topic in left.columns:
        for right_topic in right.columns:
            result = best_lag(
                left[left_topic],
                right[right_topic],
                alpha=alpha,
                lag_min=lag_min,
                lag_max=lag_max,
                alignment=alignment,
                fallback_to_all_lags=fallback_to_all_lags,
            )
            if result is None:
                continue
            rho.loc[left_topic, right_topic] = result.rho
            pvals.loc[left_topic, right_topic] = result.p
            lags.loc[left_topic, right_topic] = result.lag
            ns.loc[left_topic, right_topic] = result.n
            selected.loc[left_topic, right_topic] = result.selected_from

            if result.lag < 0:
                from_label, to_label = left_label, right_label
                from_topic, to_topic = left_topic, right_topic
            else:
                from_label, to_label = right_label, left_label
                from_topic, to_topic = right_topic, left_topic

            records.append(
                {
                    "pair_key": pair_key,
                    "pair_label": pair_label,
                    "left_layer": left_label,
                    "right_layer": right_label,
                    "left_topic": left_topic,
                    "right_topic": right_topic,
                    "topic_a": left_topic,
                    "topic_b": right_topic,
                    "rho": result.rho,
                    "p": result.p,
                    "lag": result.lag,
                    "n": result.n,
                    "selected_from": result.selected_from,
                    "is_significant": bool(np.isfinite(result.p) and result.p < alpha),
                    "from_side": "A" if result.lag < 0 else "B",
                    "to_side": "B" if result.lag < 0 else "A",
                    "from_label": from_label,
                    "to_label": to_label,
                    # Actual labels retained for the plotting adapter.
                    "from_layer": from_label,
                    "to_layer": to_label,
                    "from_topic": from_topic,
                    "to_topic": to_topic,
                }
            )

    return {
        "rho": rho,
        "p": pvals,
        "lag": lags,
        "n": ns,
        "selected_from": selected,
        "links": pd.DataFrame(records),
    }


def matching_topic_links(
    inter_outputs: dict[str, pd.DataFrame],
    *,
    minimum_abs_rho: float = 0.0,
) -> pd.DataFrame:
    """Extract same-topic display links from the full notebook-style export."""
    links = inter_outputs["links"].copy()
    columns = [
        "topic",
        "rho",
        "p",
        "lag",
        "n",
        "selected_from",
        "is_significant",
        "from_layer",
        "to_layer",
    ]
    if links.empty:
        return pd.DataFrame(columns=columns)
    links = links[links["left_topic"] == links["right_topic"]].copy()
    links = links[links["rho"].abs() >= float(minimum_abs_rho)].copy()
    links["topic"] = links["left_topic"]
    return links[columns].reset_index(drop=True)


def build_three_layer_correlations(
    left: pd.DataFrame,
    public: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_label: str,
    right_label: str,
    alpha: float,
    lag_min: int,
    lag_max: int,
    alignment: LagAlignment = "positional",
    fallback_to_all_lags: bool = True,
) -> dict[str, Any]:
    left_corr, left_p = intra_correlation_matrices(left, alignment=alignment)
    public_corr, public_p = intra_correlation_matrices(public, alignment=alignment)
    right_corr, right_p = intra_correlation_matrices(right, alignment=alignment)
    left_public = inter_correlation_outputs(
        left,
        public,
        alpha=alpha,
        lag_min=lag_min,
        lag_max=lag_max,
        left_label=left_label,
        right_label="Public Reaction",
        pair_key="left_mid",
        pair_label="Left–Mid",
        alignment=alignment,
        fallback_to_all_lags=fallback_to_all_lags,
    )
    public_right = inter_correlation_outputs(
        public,
        right,
        alpha=alpha,
        lag_min=lag_min,
        lag_max=lag_max,
        left_label="Public Reaction",
        right_label=right_label,
        pair_key="mid_right",
        pair_label="Mid–Right",
        alignment=alignment,
        fallback_to_all_lags=fallback_to_all_lags,
    )
    return {
        "left_corr": left_corr,
        "left_p": left_p,
        "public_corr": public_corr,
        "public_p": public_p,
        "right_corr": right_corr,
        "right_p": right_p,
        "left_public": left_public,
        "public_right": public_right,
    }


def same_topic_inter_links(
    data_a: pd.DataFrame,
    data_b: pd.DataFrame,
    *,
    alpha: float,
    lag_min: int,
    lag_max: int,
    label_a: str,
    label_b: str,
    minimum_abs_rho: float = 0.0,
    alignment: LagAlignment = "positional",
    fallback_to_all_lags: bool = False,
) -> pd.DataFrame:
    """Convenience wrapper for matching-topic inter-layer links.

    Figure 2 now builds the full inter-layer matrices first and then calls
    :func:`matching_topic_links`, exactly as the notebook export workflow did.
    This wrapper remains available for tests and downstream users.
    """
    outputs = inter_correlation_outputs(
        data_a,
        data_b,
        alpha=alpha,
        lag_min=lag_min,
        lag_max=lag_max,
        left_label=label_a,
        right_label=label_b,
        alignment=alignment,
        fallback_to_all_lags=fallback_to_all_lags,
    )
    return matching_topic_links(outputs, minimum_abs_rho=minimum_abs_rho)


def hierarchical_order(corr: pd.DataFrame, *, use_abs_correlation: bool = True) -> list[str]:
    """Hierarchical order using a writable NumPy distance matrix.

    Pandas/NumPy combinations with copy-on-write can expose ``DataFrame.values``
    as read-only. Building the distance matrix with ``copy=True`` avoids that
    failure. The original Figure 3 notebook clustered on absolute correlation,
    so that remains the default here.
    """
    labels = list(corr.index)
    if len(labels) <= 2:
        return labels
    matrix = corr.reindex(index=labels, columns=labels).astype(float).fillna(0.0).clip(-1.0, 1.0)
    similarity = matrix.abs() if use_abs_correlation else (matrix + 1.0) / 2.0
    similarity_array = similarity.to_numpy(dtype=float, copy=True)
    distance = 1.0 - similarity_array
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    if np.allclose(distance, 0.0):
        return labels
    condensed = squareform(distance, checks=False)
    order = leaves_list(linkage(condensed, method="average", optimal_ordering=True))
    return [labels[i] for i in order]
