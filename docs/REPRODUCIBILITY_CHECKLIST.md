# Nature code and software checklist mapping

| Checklist item | Repository location or status |
|---|---|
| Source code | `src/alignment_gap/`, `scripts/` |
| Dataset to exercise the software | Complete anonymized research dataset in `data/posts/`, `data/labels/`, `data/speeches/`, and `data/geography/` |
| Separate small demonstration dataset | Not provided by design; the complete dataset is bundled and the full workflow runs in a practical amount of time |
| System requirements and dependency versions | `README.md`, `pyproject.toml` |
| Installation instructions and timing command | `README.md` |
| Instructions to run the software on data | `README.md`, `scripts/validate_public_data.py`, `scripts/reproduce_all.py` |
| Expected output | `README.md`, `docs/FIGURE_REPRODUCTION.md` |
| Expected runtime and machine-readable report | `README.md`, `docs/ENVIRONMENT_AND_RUNTIME.md`, `results/reproduction_runtime.json` after execution |
| Instructions for other data | `docs/USING_YOUR_DATA.md` |
| Full reproduction instructions | `README.md`, `docs/FIGURE_REPRODUCTION.md` |
| License | `LICENSE`, Apache-2.0 |
| Open repository link | `README.md`, `CITATION.cff` |
| Exact execution environment | `scripts/report_environment.py` |
| Continuous integration | `.github/workflows/tests.yml` |
| Immutable DOI | To be added after the final repository is imported into Code Ocean |

## Deliberate deviation from the checklist wording

The checklist asks for a small simulated or real dataset for demonstration. This repository instead includes the complete anonymized research dataset and uses it directly as the executable example. A reduced sample was not included because it can alter correlations, lags, geographic estimates, and the appearance of the manuscript figures, while the complete workflow remains practical to run on a standard desktop. This choice should be disclosed if editors interpret the small-dataset item strictly.
