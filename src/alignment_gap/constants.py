"""Stable domain constants used by the public reproduction pipeline."""

from __future__ import annotations

from typing import Final

YEARS: Final[tuple[int, ...]] = (2016, 2020, 2024)

PAPER_TOPICS: Final[tuple[str, ...]] = (
    "Abortion",
    "Immigration",
    "Economy",
    "Wokeness",
    "Crypto Currency",
    "Guns control",
    "National security and foreign policy",
    "Democratic concerns",
    "Climate change",
    "Healthcare",
    "Education",
    "Voter trust",
    "Courts",
    "Social media",
    "Policing and protest",
    "Not specified",
)

TOPIC_CANONICAL_MAP: Final[dict[str, str]] = {
    "Taxes": "Economy",
    "LGBT issues": "Wokeness",
    "Healthcare/Science": "Healthcare",
    "Parties threaten democracy": "Democratic concerns",
    "Parties, leadership and democratic responsibility": "Democratic concerns",
    "Electoral integrity and voter trust": "Voter trust",
    "Courts, Supreme Court and rule of law": "Courts",
    "Media, information and public discourse": "Social media",
    "State coercion, policing and protest": "Policing and protest",
}

DEMOCRACY_MACRO_LABELS: Final[set[str]] = {
    "Danger to democracy",
    "Democratic concerns",
}

DEMOCRACY_SUBTOPIC_MAP: Final[dict[str, str]] = {
    "Parties, leadership and democratic responsibility": "Democratic concerns",
    "Electoral integrity and voter trust": "Voter trust",
    "Courts, Supreme Court and rule of law": "Courts",
    "Media, information and public discourse": "Social media",
    "State coercion, policing and protest": "Policing and protest",
    "Not specified": "Not specified",
    "No close match": "Not specified",
}

# The eight stance definitions reported in Table 2 of the manuscript.
STANCE_PRO_ANTI: Final[dict[str, dict[str, str]]] = {
    "Abortion": {"pro": "Pro-choice", "anti": "Pro-life"},
    "Immigration": {"pro": "Pro-immigration", "anti": "Anti-immigration"},
    "Wokeness": {"pro": "Woke supporter", "anti": "Woke opposer"},
    "Healthcare": {
        "pro": "Public Health & Science",
        "anti": "Limited Government & Anti-Science Skepticism",
    },
    "Economy": {
        "pro": "Positive Economic Outlook",
        "anti": "Negative Economic Outlook",
    },
    "Democratic concerns": {
        "pro": "Republicans threaten democracy",
        "anti": "Democrats threaten democracy",
    },
    "National security and foreign policy": {
        "pro": "Interventionist / Strong Defense",
        "anti": "Non-Interventionist / Restraint Approach",
    },
    "Guns control": {
        "pro": "Gun Control Supporter",
        "anti": "Gun Rights Advocate",
    },
}

STANCE_LABEL_ALIASES: Final[dict[str, str]] = {
    "Limited Government and Anti-Science Skepticism": "Limited Government & Anti-Science Skepticism",
    "Republicans endanger democracy": "Republicans threaten democracy",
    "Democrats endanger democracy": "Democrats threaten democracy",
    "neutral": "Neutral",
}

CANDIDATES_BY_YEAR: Final[dict[int, dict[str, str]]] = {
    2020: {"democrat": "Joe Biden", "republican": "Donald Trump"},
    2024: {"democrat": "Kamala Harris", "republican": "Donald Trump"},
}

CANDIDATE_STANCE_ALIASES: Final[dict[str, str]] = {
    "pro trump": "Pro-Trump",
    "anti trump": "Anti-Trump",
    "pro biden": "Pro-Biden",
    "anti biden": "Anti-Biden",
    "pro kamala": "Pro-Harris",
    "anti kamala": "Anti-Harris",
    "pro harris": "Pro-Harris",
    "anti harris": "Anti-Harris",
    "neither": "Neither",
    "neutral": "Neither",
}

TOPIC_ORDER: Final[tuple[str, ...]] = (
    "Democratic concerns",
    "Wokeness",
    "Economy",
    "Immigration",
    "Abortion",
    "Guns control",
    "National security and foreign policy",
    "Voter trust",
    "Courts",
    "Healthcare",
    "Climate change",
    "Social media",
    "Education",
    "Policing and protest",
    "Crypto Currency",
)

TOPIC_COLORS: Final[dict[str, str]] = {
    "Democratic concerns": "#f4a261",
    "Wokeness": "#4771b2",
    "Economy": "#82c990",
    "Immigration": "#b2184b",
    "Abortion": "#6a4c93",
    "Guns control": "#e9c46a",
    "National security and foreign policy": "#d8e681",
    "Voter trust": "#f6d55c",
    "Courts": "#f28e85",
    "Healthcare": "#2a9d8f",
    "Climate change": "#4db6ac",
    "Social media": "#8d99ae",
    "Education": "#90be6d",
    "Policing and protest": "#ef476f",
    "Crypto Currency": "#8338ec",
}

STATE_NAME_TO_ABBR: Final[dict[str, str]] = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC",
    "Washington DC": "DC", "Washington, DC": "DC",
}
STATE_ABBR_TO_NAME: Final[dict[str, str]] = {v: k for k, v in STATE_NAME_TO_ABBR.items()}
STATE_ABBR_TO_NAME["DC"] = "District of Columbia"


def canonical_topic(label: object) -> str:
    if label is None:
        return "Not specified"
    text = str(label).strip()
    if not text or text.casefold() in {"nan", "none", "no close match"}:
        return "Not specified"
    if text.casefold().startswith("wrong topic"):
        return "Not specified"
    return TOPIC_CANONICAL_MAP.get(text, text)


def canonical_stance(label: object) -> str | None:
    if label is None:
        return None
    text = str(label).strip()
    if not text or text.casefold() in {"nan", "none", "null", "no close match"}:
        return None
    # Most aliases are exact labels, but Neutral often appears with arbitrary casing.
    return STANCE_LABEL_ALIASES.get(text, STANCE_LABEL_ALIASES.get(text.casefold(), text))


def canonical_candidate_stance(label: object) -> str | None:
    if label is None:
        return None
    text = str(label).strip()
    if not text or text.casefold() in {"nan", "none", "null"}:
        return None
    key = text.casefold().replace("_", "-").replace("-", " ")
    key = " ".join(key.split())
    return CANDIDATE_STANCE_ALIASES.get(key, text)


def state_abbr(value: object) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).replace(".", "").replace(",", " ").split())
    upper = text.upper()
    if len(upper) == 2 and upper in STATE_ABBR_TO_NAME:
        return upper
    if text.casefold() in {"washington dc", "district of columbia", "dc"}:
        return "DC"
    for name, abbr in STATE_NAME_TO_ABBR.items():
        if text.casefold() == name.casefold():
            return abbr
    return None
