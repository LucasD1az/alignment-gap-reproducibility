# Using the pipeline with other data

The public figure code does not perform topic or stance classification. New input data must already contain the classifications needed by the analysis.

## Option 1: provide public-schema files directly

This is the simplest and most portable route. Create the files described in `data/SCHEMAS.md` and point a copied configuration file to them:

```yaml
extends: analysis.yml
paths:
  posts: my_data/posts
  labels: my_data/labels
  speeches: my_data/speeches
  geography: my_data/geography
  derived: my_data/derived
  results: my_results
```

Then run:

```bash
python scripts/validate_public_data.py --config config/my_analysis.yml
python scripts/reproduce_all.py --config config/my_analysis.yml
```

The current analysis expects election-year files for 2016, 2020, and 2024 and candidate speech/geography files for 2020 and 2024. To analyze different elections, update `periods`, `CANDIDATES_BY_YEAR`, and figure-specific year/topic configuration.

## Option 2: prepare public files from private classified inputs

Place private files under `data/temp/`. `config/analysis.yml` lists all accepted filename patterns under `input_patterns`. Run:

```bash
python scripts/check_temp_inputs.py
```

The main inputs are:

- raw post tables with post ID, page ID or username, creation time, likes, and total reactions;
- first-pass topic labels keyed by the raw post ID;
- topic-stance labels keyed by the raw post ID;
- democracy subtopic and democracy-stance labels when applicable;
- candidate-stance labels for 2020 and 2024;
- classified candidate speech paragraphs and separate speech-stance aggregates;
- page mapping, page-state exposure, and election-result files for 2020 and 2024.

Column-name aliases and legacy filenames are handled by `src/alignment_gap/prepare.py`. Use `scripts/check_temp_inputs.py --json` to see which concrete file was selected for each input.

## Pseudonymous identifiers

Set a private stable salt before preparation:

```bash
export ANONYMIZATION_SALT="a-long-random-private-value"
```

PowerShell:

```powershell
$env:ANONYMIZATION_SALT = "a-long-random-private-value"
```

Then run:

```bash
python scripts/prepare_public_data.py
```

The preparation step creates deterministic HMAC identifiers for posts, pages, speeches, and paragraphs. The same raw page receives the same pseudonymous page ID across years. The salt and private maps must never be committed.

## Validate and inspect

```bash
python scripts/validate_public_data.py
```

Inspect compressed CSV files without extracting them:

```bash
python -c "import pandas as pd; print(pd.read_csv('data/posts/posts_2024.csv.gz', nrows=5).to_string(index=False))"
```

Inspect JSONL labels:

```bash
python -c "import pandas as pd; print(pd.read_json('data/labels/labels_2024.jsonl.gz', lines=True, compression='gzip').head().to_string(index=False))"
```

## Configuration changes

Copy `config/analysis.yml` rather than editing the manuscript configuration in place. A custom YAML file can use:

```yaml
extends: analysis.yml
```

Only overridden fields need to be repeated.
