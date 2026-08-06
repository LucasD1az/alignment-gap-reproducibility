from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from alignment_gap.anonymize import Anonymizer, normalize_raw_id
from alignment_gap.correlations import best_lag
from alignment_gap.figure1 import build_figure_1_table
from alignment_gap.figure5 import build_state_topic_metrics, geography_diagnostics, load_election_results
from alignment_gap.prepare import _parse_election_results_wikipedia
from alignment_gap.series import candidate_support_series, load_posts_labels
from alignment_gap.validation import validate_public_data


def test_anonymization_is_deterministic_and_namespaced() -> None:
    anonymizer = Anonymizer("secret", digest_chars=10)
    assert anonymizer.identifier("page", "123") == anonymizer.identifier("page", 123.0)
    assert anonymizer.identifier("page", "123") != anonymizer.identifier("post2024", "123")
    assert normalize_raw_id(None) == ""
    assert normalize_raw_id(float("nan")) == ""


def test_preparation_schema_and_canonicalization(prepared_config: dict) -> None:
    assert validate_public_data(prepared_config) == []
    df = load_posts_labels(prepared_config, 2024)
    assert list(df.columns) == [
        "post_id",
        "page_id",
        "creation_time",
        "like_count",
        "reaction_count",
        "topic",
        "stance",
        "candidate_stance",
    ]
    assert "Economy" in set(df["topic"])
    assert "Wokeness" in set(df["topic"])
    assert "Democratic concerns" in set(df["topic"])
    assert "Healthcare" in set(df["topic"])
    assert "Taxes" not in set(df["topic"])
    assert "Danger to democracy" not in set(df["topic"])
    assert "Concern about Democratic Decline" not in set(df["stance"].dropna())
    assert df.loc[df["creation_time"].dt.day == 1, "candidate_stance"].iloc[0] == "Neither"

    public_path = Path(prepared_config["_repo_root"]) / "data" / "posts" / "posts_2024.csv.gz"
    public = pd.read_csv(public_path)
    assert "text" not in public.columns
    assert public["post_id"].str.startswith("post2024_").all()
    assert public["page_id"].str.startswith("page_").all()




def test_speech_label_column_is_preserved_as_stance(prepared_config: dict) -> None:
    from alignment_gap.series import aggregate_stance_bias_speeches, load_speeches

    speeches = load_speeches(prepared_config, 2024, "Kamala Harris")
    assert speeches["stance"].notna().any()
    bias = aggregate_stance_bias_speeches(
        prepared_config,
        2024,
        "Kamala Harris",
        "2024-06-01",
        "2024-06-24",
    )
    assert bias.notna().any()
    assert "Economy" in bias.index
    # The separate notebook speech-stance source is authoritative when present.
    assert bias["Economy"] < 0


def test_best_lag_recovers_lead_direction() -> None:
    rng = np.random.default_rng(7)
    x = pd.Series(rng.normal(size=120)).rolling(3, min_periods=1).mean()
    # x leads y by four positions under the documented convention.
    y = x.shift(4)
    result = best_lag(x, y, alpha=0.05, lag_min=1, lag_max=8)
    assert result is not None
    assert result.lag == -4
    assert result.rho > 0.99


def test_support_and_figure_one_tables(prepared_config: dict) -> None:
    support = candidate_support_series(prepared_config, 2024, rolling_days=1)
    assert {"support_difference", "dem_favorable_likes", "rep_favorable_likes", "total_likes"} <= set(support)
    assert support["support_difference"].dropna().between(-1, 1).all()

    figure_1 = build_figure_1_table(prepared_config, 2024)
    assert {"topic", "stance", "like_count", "reaction_count", "like_share", "topic_order"} <= set(figure_1)
    assert np.isclose(figure_1["like_share"].sum(), 1.0)



def test_winner_first_election_results_are_mapped_by_year(tmp_path: Path) -> None:
    # The raw Results exports put the national winner first. Candidate labels may be
    # absent/unreadable, so the year-specific fallback must preserve party identity.
    rows_2020 = [
        ["California", "winner", "63.5%", "x", "runner up", "34.3%"],
        ["Texas", "winner", "46.5%", "x", "runner up", "52.1%"],
    ]
    path_2020 = tmp_path / "2020_Results.csv"
    pd.DataFrame(rows_2020).to_csv(path_2020, index=False, header=False)
    parsed_2020 = _parse_election_results_wikipedia(path_2020, 2020)
    assert parsed_2020.loc[0, "democrat_pct"] == 63.5
    assert parsed_2020.loc[0, "republican_pct"] == 34.3
    assert parsed_2020.loc[1, "democrat_pct"] == 46.5
    assert parsed_2020.loc[1, "republican_pct"] == 52.1

    rows_2024 = [
        ["California", "winner", "34.3%", "x", "runner up", "63.5%"],
        ["Texas", "winner", "52.1%", "x", "runner up", "46.5%"],
    ]
    path_2024 = tmp_path / "2024_Results.csv"
    pd.DataFrame(rows_2024).to_csv(path_2024, index=False, header=False)
    parsed_2024 = _parse_election_results_wikipedia(path_2024, 2024)
    assert parsed_2024.loc[0, "republican_pct"] == 34.3
    assert parsed_2024.loc[0, "democrat_pct"] == 63.5
    assert parsed_2024.loc[1, "republican_pct"] == 52.1
    assert parsed_2024.loc[1, "democrat_pct"] == 46.5

def test_geography_uses_same_anonymized_pages(prepared_config: dict) -> None:
    metrics = build_state_topic_metrics(prepared_config, 2024)
    assert {"state_abbr", "topic", "stance_bias", "dem_minus_rep"} <= set(metrics)
    immigration = metrics[metrics["topic"] == "Immigration"]
    assert set(immigration["state_abbr"]) == {"CA", "TX"}
    assert immigration["likes_total"].sum() > 0
    margins = immigration.set_index("state_abbr")["dem_minus_rep"]
    assert margins["CA"] > 0
    assert margins["TX"] < 0




def test_legacy_username_page_mapping_is_used_when_owner_id_exists(prepared_config: dict) -> None:
    root = Path(prepared_config["_repo_root"])
    private_map = pd.read_csv(root / "data" / "temp" / "private_maps" / "id_map_2024.csv.gz")
    assert private_map["page_mapping_matched"].all()
    assert private_map["raw_page_key"].astype(str).str.startswith("page-user-").all()

    diagnostics = geography_diagnostics(prepared_config, 2024)
    assert diagnostics["n_overlapping_pages"] == 2
    assert diagnostics["n_posts_on_overlapping_pages"] > 0


def test_electoral_results_load_independently_of_topic_metrics(prepared_config: dict) -> None:
    election = load_election_results(prepared_config, 2020)
    assert {"state_abbr", "dem_minus_rep"} <= set(election.columns)
    margins = election.set_index("state_abbr")["dem_minus_rep"]
    assert margins["CA"] > 0
    assert margins["TX"] < 0


def test_manifest_contains_checksums(prepared_config: dict) -> None:
    manifest_path = Path(prepared_config["_repo_root"]) / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["files"]
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])


def test_reproduce_all_figures_smoke(prepared_config: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from alignment_gap.figure1 import reproduce_figure_1
    from alignment_gap.figure2 import reproduce_figure_2
    from alignment_gap.figure3 import reproduce_figure_3
    from alignment_gap.figure4 import reproduce_figure_4
    from alignment_gap.figure5 import reproduce_figure_5

    outputs = []
    for function in (
        reproduce_figure_1,
        reproduce_figure_2,
        reproduce_figure_3,
        reproduce_figure_4,
        reproduce_figure_5,
    ):
        outputs.extend(function(prepared_config))
    assert outputs
    assert all(Path(path).exists() for path in outputs)


def test_preflight_finds_complete_synthetic_inputs(prepared_config: dict) -> None:
    from alignment_gap.preflight import missing_required_inputs

    assert missing_required_inputs(prepared_config) == []


def test_hierarchical_order_accepts_read_only_backing_arrays() -> None:
    from alignment_gap.correlations import hierarchical_order

    corr = pd.DataFrame(
        [[1.0, 0.8, -0.1], [0.8, 1.0, -0.2], [-0.1, -0.2, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    # Reproduce the relevant condition: callers must not depend on mutating
    # pandas' backing array in place.
    backing = corr.to_numpy(copy=False)
    try:
        backing.flags.writeable = False
    except ValueError:
        pass
    order = hierarchical_order(corr)
    assert set(order) == {"A", "B", "C"}
    assert abs(order.index("A") - order.index("B")) == 1


def test_figure_2_inter_links_compare_matching_topics_only() -> None:
    from alignment_gap.correlations import same_topic_inter_links

    rng = np.random.default_rng(19)
    topic_a = pd.Series(rng.normal(size=100)).rolling(3, min_periods=1).mean()
    topic_b = pd.Series(rng.normal(size=100)).rolling(3, min_periods=1).mean()
    left = pd.DataFrame({"A": topic_a, "B": topic_b})
    public = pd.DataFrame({"A": topic_a.shift(3), "B": topic_b.shift(5), "C": topic_a})
    links = same_topic_inter_links(
        left,
        public,
        alpha=0.01,
        lag_min=1,
        lag_max=8,
        label_a="Candidate",
        label_b="Public Reaction",
        minimum_abs_rho=0.5,
    )
    assert set(links["topic"]) == {"A", "B"}
    assert "C" not in set(links["topic"])
    assert (links["p"] < 0.01).all()


def test_public_reaction_series_is_daily_likes_per_post(prepared_config: dict) -> None:
    from alignment_gap.series import daily_topic_like_ratio_posts

    ratio = daily_topic_like_ratio_posts(
        prepared_config,
        2024,
        start="2024-06-01",
        end="2024-06-24",
        rolling_days=None,
    )
    # June 1 is a Taxes post canonicalized to Economy with ten likes.
    assert ratio.loc[pd.Timestamp("2024-06-01", tz="UTC"), "Economy"] == 10.0
    # No Economy post on June 2: the daily likes-per-post signal is zero before smoothing.
    assert ratio.loc[pd.Timestamp("2024-06-02", tz="UTC"), "Economy"] == 0.0

    smoothed = daily_topic_like_ratio_posts(
        prepared_config,
        2024,
        start="2024-06-01",
        end="2024-06-24",
        rolling_days=7,
    )
    assert smoothed.iloc[:3].isna().all(axis=None)
    assert smoothed.iloc[-3:].isna().all(axis=None)


def test_legacy_lag_alignment_matches_notebook_index_semantics() -> None:
    from alignment_gap.correlations import spearman_at_lag
    from scipy.stats import spearmanr

    dates = pd.date_range("2020-06-01", periods=12, freq="D", tz="UTC")
    x = pd.Series(np.arange(12, dtype=float), index=dates, name="x")
    y = pd.Series(np.array([4, 1, 7, 2, 9, 3, 8, 5, 0, 11, 6, 10], dtype=float), index=dates, name="y")
    lag = 3
    manual = pd.concat([x.iloc[lag:], y.iloc[:-lag]], axis=1).dropna()
    expected_rho, expected_p = spearmanr(manual.iloc[:, 0], manual.iloc[:, 1])

    rho, p, n = spearman_at_lag(x, y, lag, alignment="legacy_index")
    assert n == len(manual)
    assert np.isclose(rho, expected_rho)
    assert np.isclose(p, expected_p)


def test_inter_export_uses_significant_first_then_fallback() -> None:
    from alignment_gap.correlations import inter_correlation_outputs

    rng = np.random.default_rng(314)
    index = pd.date_range("2020-06-01", periods=40, freq="D", tz="UTC")
    left = pd.DataFrame({"A": rng.normal(size=40), "B": rng.normal(size=40)}, index=index)
    right = pd.DataFrame({"A": rng.normal(size=40), "B": rng.normal(size=40)}, index=index)
    outputs = inter_correlation_outputs(
        left,
        right,
        alpha=1e-100,
        lag_min=1,
        lag_max=4,
        left_label="Candidate",
        right_label="Public Reaction",
        alignment="legacy_index",
        fallback_to_all_lags=True,
    )
    assert outputs["rho"].shape == (2, 2)
    assert outputs["rho"].notna().all(axis=None)
    assert set(outputs["links"]["selected_from"]) == {"fallback_all_lags"}
    assert len(outputs["links"]) == 4


def test_notebook_series_keep_sparse_speech_dates_and_missing_public_topic_days(prepared_config: dict) -> None:
    from alignment_gap.series import (
        notebook_daily_topic_like_ratio_posts,
        notebook_daily_topic_volume_speeches,
    )

    speech_path = (
        Path(prepared_config["_repo_root"])
        / "data"
        / "speeches"
        / "speeches_joe_biden_2020.csv.gz"
    )
    speeches = pd.read_csv(speech_path)
    keep_dates = sorted(speeches["date"].unique())[:4]
    speeches = speeches[speeches["date"].isin(keep_dates)]
    speeches.to_csv(speech_path, index=False, compression="gzip")

    speech_wide = notebook_daily_topic_volume_speeches(
        prepared_config,
        2020,
        "Joe Biden",
        start="2020-06-01",
        end="2020-06-24",
        rolling_days=7,
    )
    assert len(speech_wide.index) == 4
    assert speech_wide.index.equals(pd.DatetimeIndex(pd.to_datetime(keep_dates, utc=True)))
    assert speech_wide.iloc[0].notna().all()

    public = notebook_daily_topic_like_ratio_posts(
        prepared_config,
        2024,
        start="2024-06-01",
        end="2024-06-24",
        rolling_days=1,
    )
    assert public.loc[pd.Timestamp("2024-06-01", tz="UTC"), "Economy"] == 10.0
    assert pd.isna(public.loc[pd.Timestamp("2024-06-02", tz="UTC"), "Economy"])


def test_figure_3_writes_heatmap_and_subnetwork_separately(prepared_config: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from alignment_gap.figure3 import reproduce_figure_3

    outputs = [Path(path) for path in reproduce_figure_3(prepared_config)]
    names = {path.name for path in outputs}
    for year in (2020, 2024):
        assert f"figure_3_heatmap_{year}.pdf" in names
        assert f"figure_3_subnetwork_{year}.pdf" in names
    assert "figure_3.pdf" not in names


def test_figure_4_uses_manuscript_democratic_concerns_orientation(prepared_config: dict) -> None:
    from alignment_gap.constants import STANCE_PRO_ANTI
    from alignment_gap.series import daily_stance_bias_posts

    assert "democratic_concerns_notebook_orientation" not in prepared_config["figure_4"]
    assert STANCE_PRO_ANTI["Democratic concerns"] == {
        "pro": "Republicans threaten democracy",
        "anti": "Democrats threaten democracy",
    }

    bias = daily_stance_bias_posts(
        prepared_config,
        2024,
        start="2024-06-01",
        end="2024-06-24",
        rolling_days=1,
    )
    dc = bias[bias["topic"] == "Democratic concerns"].copy()
    assert not dc.empty
    expected = (dc["likes_pro"] - dc["likes_anti"]) / dc["likes_total"].replace(0, np.nan)
    assert np.allclose(dc["bias"].fillna(0.0), expected.fillna(0.0))


def test_figure_4_writes_support_and_radars_separately(prepared_config: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from alignment_gap.figure4 import reproduce_figure_4

    outputs = [Path(path) for path in reproduce_figure_4(prepared_config)]
    names = {path.name for path in outputs}
    for year in (2020, 2024):
        assert f"figure_4_support_stance_{year}.pdf" in names
        assert f"figure_4_radars_{year}.pdf" in names
    assert "figure_4.pdf" not in names


def test_figure_5_writes_decomposed_components(prepared_config: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from alignment_gap.figure5 import reproduce_figure_5

    outputs = [Path(path) for path in reproduce_figure_5(prepared_config)]
    names = {path.name for path in outputs}
    for year in (2020, 2024):
        assert f"figure_5_electoral_map_{year}.html" in names
        assert f"figure_5_immigration_zscore_{year}.html" in names
        assert f"figure_5_scatter_grid_{year}.pdf" in names
        assert f"figure_5_scatter_grid_{year}.png" in names
        assert f"figure_5_scatter_grid_{year}.svg" in names
        assert f"figure_5_{year}.html" not in names


def test_environment_report_contains_versions() -> None:
    from alignment_gap.environment import collect_environment

    report = collect_environment()
    assert report["python"]
    assert report["system"]
    assert "numpy" in report["packages"]
