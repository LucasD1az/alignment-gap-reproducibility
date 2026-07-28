from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from alignment_gap.prepare import prepare_all


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _speech_frame(year: int, candidate: str) -> pd.DataFrame:
    topics = [
        "Economy",
        "Wokeness",
        "Danger to democracy",
        "Immigration",
        "Healthcare/Science",
        "Abortion",
        "Guns control",
        "National security and foreign policy",
    ]
    stance_by_topic = {
        "Economy": "Positive Economic Outlook",
        "Wokeness": "Woke opposer",
        "Danger to democracy": "Democrats threaten democracy",
        "Immigration": "Anti-immigration",
        "Healthcare/Science": "Public Health & Science",
        "Abortion": "Pro-choice",
        "Guns control": "Gun Control Supporter",
        "National security and foreign policy": "Interventionist / Strong Defense",
    }
    start = pd.Timestamp(f"{year}-06-01")
    rows = []
    for day in range(24):
        for repeat in range(1 + day % 3):
            topic = topics[(day + repeat) % len(topics)]
            rows.append(
                {
                    "Speech_id": f"{candidate[:2]}-{day // 4}",
                    "paragraph_id": f"{day}-{repeat}",
                    "date": start + pd.Timedelta(days=day),
                    "topic": topic,
                    "democracy_subtopic": (
                        "Parties, leadership and democratic responsibility"
                        if topic == "Danger to democracy" else None
                    ),
                    # Original speech-stance exports use ``label`` for stance.
                    "label": stance_by_topic[topic],
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def prepared_config(tmp_path: Path) -> dict:
    source = Path(__file__).resolve().parents[1] / "config" / "analysis.yml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["_repo_root"] = str(tmp_path)
    config["privacy"]["save_private_id_maps"] = True
    config["privacy"]["id_digest_chars"] = 12
    config["figure_2"]["minimum_layer_share"] = 0.0
    config["figure_5"]["static_formats"] = []
    for year in (2016, 2020, 2024):
        config["periods"][year]["campaign_start"] = f"{year}-06-01"
        config["periods"][year]["campaign_end"] = f"{year}-06-24"
        config["periods"][year]["figure_1_end"] = f"{year}-06-24"

    temp = tmp_path / "data" / "temp"
    (temp / "first_pass").mkdir(parents=True)
    (temp / "second_pass").mkdir(parents=True)
    (temp / "stance" / "parties").mkdir(parents=True)
    (temp / "sentiment").mkdir(parents=True)
    (temp / "speeches").mkdir(parents=True)
    (temp / "geography").mkdir(parents=True)
    (temp / ".anonymization_salt").write_text("fixed-test-salt\n", encoding="utf-8")

    raw_topics = [
        "Taxes",
        "LGBT issues",
        "Danger to democracy",
        "Immigration",
        "Healthcare/Science",
        "Abortion",
        "Guns control",
        "National security and foreign policy",
        "Not specified",
    ]
    stances = [
        "Positive Economic Outlook",
        "Woke opposer",
        "Concern about Democratic Decline",
        "Anti-immigration",
        "Public Health & Science",
        "Pro-choice",
        "Gun Control Supporter",
        "Interventionist / Strong Defense",
        "Neutral",
    ]
    democracy_subtopics = [
        None,
        None,
        "Parties, leadership and democratic responsibility",
        None,
        None,
        None,
        None,
        None,
        None,
    ]

    for year in (2016, 2020, 2024):
        rows = []
        topic_rows = []
        stance_rows = []
        democracy_rows = []
        party_rows = []
        candidate_rows = []
        for i in range(24):
            raw_id = f"{year}-{i}"
            topic_index = i % len(raw_topics)
            rows.append(
                {
                    "id": raw_id,
                    "post_owner.id": f"post-page-{i % 2}",
                    "post_owner.username": f"page-user-{i % 2}",
                    "creation_time": f"{year}-06-{i + 1:02d}T12:00:00Z",
                    "statistics.like_count": 10 + i,
                    "statistics.reaction_count": 20 + i,
                    "text": "private text must never be released",
                }
            )
            topic_rows.append({"id": raw_id, "label": raw_topics[topic_index]})
            stance_rows.append({"id": raw_id, "stance": stances[topic_index]})
            if democracy_subtopics[topic_index]:
                democracy_rows.append({"id": raw_id, "label": democracy_subtopics[topic_index]})
                party_rows.append({"id": raw_id, "stance": "Democrats threaten democracy"})
            if year in (2020, 2024):
                democrat = "Biden" if year == 2020 else "Harris"
                labels = [f"Pro-{democrat}", "Anti-Trump", "Pro-Trump", f"Anti-{democrat}", "Neither"]
                candidate_rows.append(
                    {
                        "p_id": raw_id,
                        "class": labels[i % len(labels)],
                        "confidence": 0.50 if i == 0 else 0.99,
                    }
                )

        pd.DataFrame(rows).to_csv(temp / f"{year}USElections.csv", index=False)
        _write_jsonl(temp / "first_pass" / f"Topic_labels_ollama_{year}.jsonl", topic_rows)
        _write_jsonl(temp / "stance" / f"stance_labels_{year}.jsonl", stance_rows)
        _write_jsonl(temp / "second_pass" / f"Subtopic_democracy_labels_ollama_{year}.jsonl", democracy_rows)
        _write_jsonl(temp / "stance" / "parties" / f"{year}_Topic_labels.jsonl", party_rows)
        if year in (2020, 2024):
            pd.DataFrame(candidate_rows).to_csv(temp / "sentiment" / f"output_to_email_{year}.csv", index=False)

    for key, year, candidate, filename in (
        ("biden", 2020, "Joe Biden", "biden_paragraphs_2020_classified.pkl"),
        ("trump", 2020, "Donald Trump", "trump_paragraphs_2020_classified.pkl"),
        ("harris", 2024, "Kamala Harris", "harris_paragraphs_2024_classified.pkl"),
        ("trump", 2024, "Donald Trump", "trump_paragraphs_2024_classified.pkl"),
    ):
        _speech_frame(year, candidate).to_pickle(temp / "speeches" / filename)

    # Separate candidate-stance source used by 07_correlations.ipynb.  Make
    # Harris's Economy stance negative so tests can distinguish it from the
    # positive paragraph-level fallback in _speech_frame.
    pd.DataFrame(
        {
            "date": pd.date_range("2024-06-01", periods=4, freq="D"),
            "Economy - Positive Economic Outlook": [0, 0, 0, 0],
            "Economy - Negative Economic Outlook": [4, 3, 2, 1],
            "Economy - Neutral": [0, 0, 0, 0],
            "Wokeness - Woke supporter": [0, 0, 0, 0],
            "Wokeness - Woke opposer": [1, 1, 1, 1],
            "Wokeness - Neutral": [0, 0, 0, 0],
        }
    ).to_pickle(temp / "speeches" / "2024_speech_stance_harris_time_series.pkl")

    for year in (2020, 2024):
        # Legacy geography files map post_owner.username -> ads page_id and are
        # headerless.  Raw posts also contain post_owner.id, so this fixture
        # guards against accidentally selecting only the ID column.
        pd.DataFrame(
            [
                ["page-user-0", "ad-page-0"],
                ["page-user-1", "ad-page-1"],
            ]
        ).to_csv(temp / "geography" / f"{year}_ids.csv", index=False, header=False)
        pd.DataFrame(
            {
                "page_id": ["ad-page-0", "ad-page-1"],
                "California": [0.8, 0.2],
                "Texas": [0.2, 0.8],
            }
        ).to_csv(temp / "geography" / f"region_impressions_distribution_{year}.csv", index=False)
        pd.DataFrame(
            {
                "state": ["California", "Texas"],
                "democrat_pct": [63.5, 46.5],
                "republican_pct": [34.3, 52.1],
            }
        ).to_csv(temp / "geography" / f"{year}_Results.csv", index=False)

    prepare_all(config)
    return config
