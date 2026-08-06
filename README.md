# The Alignment Gap: How Public Resonance Shaped Three U.S. Presidential Elections (2016–2024)

Lucas Díaz Celauro, Sebastián Pinto, Sofia del Pozo, Alireza Hashemi, Matteo Serafino, Pablo Balenzuela, Hernán A. Makse

This repository contains the anonymized public data and Python code needed to reproduce the five main figures of the manuscript. The complete research dataset is bundled with the repository and is used directly as the executable example. Because the full workflow runs in a practical amount of time on a standard desktop, no separate simulated or subsampled demonstration dataset is provided.

The repository excludes Facebook post text, page names, usernames, links, original Meta identifiers, speech text, and the topic/stance classification code. Public identifiers are pseudonymous and the figure scripts read only the public data folders.

## Quick start

Create and activate a virtual environment:

```bash
python -m venv .venv
```

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

Install, validate the bundled research data, and reproduce all main figures:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/validate_public_data.py
python scripts/reproduce_all.py
```

A successful run ends with:

```text
Full reproduction completed in <seconds> seconds.
Runtime report: <repository>/results/reproduction_runtime.json
```

## System requirements

### Operating systems and Python

| Environment | Status |
|---|---|
| Ubuntu 24.04, Python 3.10–3.12 | Tested automatically with GitHub Actions |
| Windows 10/11, Python 3.10 or newer | Supported; PowerShell commands are provided below |
| macOS, Python 3.10 or newer | Expected to work, but not currently tested in continuous integration |

Python 3.10 or newer is required. The software is CPU-only and does not require a GPU or other non-standard hardware.

Recommended hardware:

- 8 GB RAM minimum; 16 GB recommended;
- a modern multi-core desktop CPU;
- at least 2 GB of free disk space for the environment and generated outputs.

Static Plotly map export uses Kaleido and may require a local Chrome/Chromium installation. Interactive HTML maps are always produced even when static map export is unavailable.

### Software dependencies

The dependency bounds are recorded in `pyproject.toml`:

| Package | Minimum version |
|---|---:|
| NumPy | 1.26 |
| pandas | 2.1 |
| SciPy | 1.11 |
| Matplotlib | 3.8 |
| PyYAML | 6.0 |
| Plotly | 5.20 |
| Kaleido | 1.0 |
| Pillow | 10.0 |
| pytest (development) | 8.0 |
| Ruff (development) | 0.6 |

Print the exact operating-system, Python, hardware, and package versions used on a machine with:

```bash
python scripts/report_environment.py --output results/environment.json
```

## Installation

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Measure installation time on the author workstation with:

```powershell
Measure-Command { python -m pip install -e ".[dev]" } | Select-Object TotalSeconds
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Measure installation time with:

```bash
time python -m pip install -e ".[dev]"
```

Installation normally takes under a minute on a desktop computer with a broadband connection. The exact time depends mainly on package download speed. During testing, it took 59 seconds.

## Full-data executable example

The bundled anonymized research dataset serves as both the analysis input and the executable example. This avoids presenting simulated outputs that have no substantive interpretation and avoids introducing sampling differences relative to the manuscript.

Validate the input files:

```bash
python scripts/validate_public_data.py
```

Expected terminal output:

```text
Public data validation passed.
```

Run the complete workflow:

```bash
python scripts/reproduce_all.py
```

The command reports elapsed time for each figure and writes a machine-readable record to:

```text
results/reproduction_runtime.json
```

The expected runtime on a modern desktop is a few minutes. It took 3 minutes and 31 seconds during testing.

## Reproduce the manuscript figures

The complete workflow is:

```bash
python scripts/validate_public_data.py
python scripts/reproduce_all.py
```

Individual figures can be generated with:

```bash
python scripts/reproduce_figure_1.py
python scripts/reproduce_figure_2.py
python scripts/reproduce_figure_3.py
python scripts/reproduce_figure_4.py
python scripts/reproduce_figure_5.py
```

Figure 1 exports the data tables used to compose the final alluvial graphic externally. Figures 2–5 export the independent components used in the manuscript, together with intermediate CSV and JSON audit files. Detailed outputs, formulas, thresholds, and notebook-to-script decisions are documented in [`docs/FIGURE_REPRODUCTION.md`](docs/FIGURE_REPRODUCTION.md).

Expected top-level output directories are:

```text
results/figure_1/
results/figure_2/
results/figure_3/
results/figure_4/
results/figure_5/
results/reproduction_runtime.json
```

## Use with other data

There are two supported entry points.

### Already anonymized and classified data

Prepare files matching [`data/SCHEMAS.md`](data/SCHEMAS.md), place them in your own directories, copy `config/analysis.yml`, and change the entries under `paths`. Then run:

```bash
python scripts/validate_public_data.py --config config/my_analysis.yml
python scripts/reproduce_all.py --config config/my_analysis.yml
```

### Private classified source files

The repository can construct the public schemas from private source files placed under `data/temp/`. The input data must already contain topic, topic-stance, and candidate-stance classifications; the language-model classification pipeline is not part of this repository.

```bash
python scripts/check_temp_inputs.py
python scripts/prepare_public_data.py
python scripts/validate_public_data.py
```

Set a stable secret before preparation so that pseudonymous identifiers remain deterministic:

**PowerShell**

```powershell
$env:ANONYMIZATION_SALT = "replace-with-a-long-private-random-value"
python scripts/prepare_public_data.py
```

**Linux/macOS**

```bash
export ANONYMIZATION_SALT="replace-with-a-long-private-random-value"
python scripts/prepare_public_data.py
```

Never commit the salt, raw text, original identifiers, or private ID maps. Exact accepted filenames, required columns, and customization guidance are provided in [`docs/USING_YOUR_DATA.md`](docs/USING_YOUR_DATA.md).

## Public data

For each election year, the post table contains only:

```text
post_id, page_id, creation_time, like_count, reaction_count
```

The matching JSONL record contains:

```json
{"post_id":"post2024_…","topic":"Economy","stance":"Negative Economic Outlook","candidate_stance":"Pro-Trump"}
```

`stance` and `candidate_stance` are `null` when they do not apply. Speech tables contain pseudonymous speech and paragraph identifiers, candidate, date, topic, and stance, without transcript text. Geography tables contain pseudonymous page identifiers, state exposure shares, and public election results.

See [`data/SCHEMAS.md`](data/SCHEMAS.md) and `data/manifest.json` for exact schemas, counts, and checksums.

## Tests and continuous integration

Run the local test suite with:

```bash
pytest
```

GitHub Actions uses Ubuntu 24.04 and Python 3.10, 3.11, and 3.12. It runs the tests and validates the bundled public data on every supported Python version. The complete full-data reproduction is also executed on Python 3.11.

## Repository layout

```text
.
├── config/
│   └── analysis.yml              # manuscript reproduction configuration
├── data/
│   ├── posts/                    # anonymized public posts
│   ├── labels/                   # topic and stance labels
│   ├── speeches/                 # speech topic/stance data without text
│   ├── geography/                # page exposure and election results
│   └── temp/                     # private preparation inputs; ignored by Git
├── docs/
├── src/alignment_gap/
├── scripts/
├── tests/
└── results/                      # generated outputs; ignored by Git
```

## License

The software is licensed under the [Apache License 2.0](LICENSE). The public derived datasets are included for research transparency and reproducibility; users remain responsible for complying with any terms applicable to the underlying source data.

## Citation and archival record

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The final immutable software release and DOI will be added after the reviewed repository is imported into Code Ocean. Until then, cite the repository URL and the associated manuscript.

Repository: <https://github.com/LucasD1az/alignment-gap-reproducibility>

## Contact

For questions about the code or data release, open a GitHub issue or contact the corresponding author listed in the manuscript.
