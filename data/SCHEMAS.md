# Public data schemas

## `posts/posts_YYYY.csv.gz`

| field | type | description |
|---|---|---|
| `post_id` | string | deterministic HMAC pseudonym, namespaced by year |
| `page_id` | string | deterministic HMAC pseudonym, stable across years |
| `creation_time` | UTC timestamp | original publication time |
| `like_count` | non-negative integer | Facebook like count |
| `reaction_count` | non-negative integer | total Facebook reaction count |

## `labels/labels_YYYY.jsonl.gz`

One JSON object per post:

```json
{"post_id":"post2024_abcd...","topic":"Economy","stance":"Negative Economic Outlook","candidate_stance":"Pro-Trump"}
```

`stance` and `candidate_stance` may be `null`.

## `speeches/speeches_<candidate>_YYYY.csv.gz`

| field | type | description |
|---|---|---|
| `speech_id` | string | anonymized speech identifier |
| `paragraph_id` | string | anonymized paragraph identifier |
| `candidate` | string | public candidate name |
| `date` | date | speech date |
| `topic` | string | final canonical topic |
| `stance` | string/null | topic stance when defined |

## `geography/page_state_exposure_YYYY.csv.gz`

| field | type | description |
|---|---|---|
| `page_id` | string | same anonymized page key as post data |
| `state` | string | U.S. state name |
| `state_abbr` | string | two-letter abbreviation |
| `impression_value` | float | pre-normalization value from the private exposure table |
| `impression_share` | float | normalized page-level state share |

## `geography/election_results_YYYY.csv`

| field | type | description |
|---|---|---|
| `state` | string | U.S. state name |
| `state_abbr` | string | two-letter abbreviation |
| `democrat_pct` | float | Democratic vote percentage |
| `republican_pct` | float | Republican vote percentage |
| `dem_minus_rep` | float | Democratic minus Republican margin |

## `speeches/stance_counts_<candidate>_YYYY.csv.gz`

Daily aggregate speech-stance counts used to color candidate nodes in Figures 2
and 3. This file is derived from the separate stance inputs used by
`07_correlations.ipynb` and contains no speech text or original identifiers.

| field | type | description |
|---|---|---|
| `date` | date | speech date |
| `candidate` | string | public candidate name |
| `topic` | string | final canonical topic |
| `stance` | string | one of the manuscript stance labels or `Neutral` |
| `paragraph_count` | non-negative number | paragraphs assigned to that topic–stance pair |
