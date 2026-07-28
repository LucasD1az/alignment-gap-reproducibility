# The Alignment Gap — reproducibility repository

Clean public pipeline for reproducing the five main figures of:

> **The Alignment Gap: How Public Resonance Shaped Three U.S. Presidential Elections (2016–2024)**

The repository deliberately excludes the text of Facebook posts, page names,
usernames, links, and the topic/stance classification code. It starts from the
already classified private files in `data/temp/`, creates a minimal anonymized
public release, and reproduces the analyses from those public files. See
[`MIGRATION_NOTES.md`](MIGRATION_NOTES.md) for the decisions used to consolidate
the original notebooks and [`data/SCHEMAS.md`](data/SCHEMAS.md) for exact public
schemas.

## What is public

For each election year, the released post table contains only:

```text
post_id, page_id, creation_time, like_count, reaction_count
```

Both identifiers are deterministic HMAC identifiers that differ from the Meta
identifiers. The matching labels file contains:

```json
{"post_id":"post2024_…","topic":"Economy","stance":"Negative Economic Outlook","candidate_stance":"Pro-Trump"}
```

`stance` and `candidate_stance` are `null` when they do not apply or were not
available.

The public speech tables contain:

```text
speech_id, paragraph_id, candidate, date, topic, stance
```

No post or speech text is written to the public folders.

## Repository layout

```text
.
├── config/analysis.yml            # periods, thresholds, input patterns
├── data/
│   ├── temp/                       # private inputs; ignored by Git
│   ├── posts/                      # anonymized posts_YYYY.csv.gz
│   ├── labels/                     # labels_YYYY.jsonl.gz
│   ├── speeches/                   # anonymized paragraph classifications
│   ├── geography/                  # page exposure and election results
│   └── manifest.json               # counts, checksums and release manifest
├── src/alignment_gap/              # reusable analysis package
├── scripts/                        # one command per preparation/figure step
├── results/                        # generated figures; ignored by Git
└── tests/
```

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Static Plotly export for Figure 5 uses Kaleido. Depending on the Kaleido
version and operating system, a local Chrome installation may also be needed.
The script always writes an interactive HTML version even when static export is
not available.

## Reproduce the figures

```bash
python scripts/reproduce_all.py
```

Or run individual figures:

```bash
python scripts/reproduce_figure_1.py
python scripts/reproduce_figure_2.py
python scripts/reproduce_figure_3.py
python scripts/reproduce_figure_4.py
python scripts/reproduce_figure_5.py
```

### Figure 2 display and thresholds

Figure 2 labels every topic once on the public-reaction plane and colors all
three layers with their own aggregate stance bias. Node sizes, label sizes and
thresholds are editable under `figure_2` in `config/analysis.yml`. The defaults
that reproduce the final plotting notebook are:

- intra-layer significance: `p < 0.01`;
- inter-layer significance used to build the notebook link tables: `p < 0.05`;
- inter-layer minimum `|rho|`: `0.30` in 2020 and `0.45` in 2024;
- within-candidate minimum `|rho|`: `0.50` in 2020 and `0.70` in 2024;
- within-public minimum `|rho|`: `0.10` in 2020 and `0.40` in 2024.

To enforce the stricter manuscript wording for inter-layer links, change
`figure_2.inter_alpha` from `0.05` to `0.01`.

### Figure 1

The paper graphic was assembled outside Python. The script exports:

```text
results/figure_1/figure_1_2016.csv
results/figure_1/figure_1_2020.csv
results/figure_1/figure_1_2024.csv
results/figure_1/figure_1_all_years.csv
```

The tables contain topic, stance, post count, likes, total reactions, shares and
ordering. Topics below 3% of annual likes are combined as `Others`, matching the
current paper workflow.

### Figures 2 and 3

Figure 2 uses the separate candidate speech-stance aggregates described above
for node colors. The reaction layer is the centered seven-day mean of the daily
likes-per-post signal for each topic. The candidate layers are the centered seven-day mean
daily number of speech paragraphs per topic. Correlations are Spearman
correlations.

Figure 2 reproduces the final two-dimensional multilayer layout used in
`07_speeches_bis.ipynb`: three elliptical topic rings on perspective planes,
black within-layer links, and purple directed links between the same topic in
adjacent layers. Inter-layer links retain the maximum absolute correlation over
non-zero lags within ±14 days only when the selected lag is significant. The
2020 and 2024 topic order and all intra/inter thresholds are centralized in
`config/analysis.yml`.

Figure 3 evaluates the complete rectangular candidate–public correlation
blocks, including comparisons between different topics, and then applies the
figure-specific significance and magnitude masks.

Every matrix, lag table, edge table and node table used for the plots is also
written under the corresponding `results/figure_*/data/` directory.

### Figure 4

The candidate support difference is:

```text
(Pro-Democrat likes + Anti-Trump likes)
- (Pro-Trump likes + Anti-Democrat likes)
------------------------------------------------
all candidate-classified likes, including Neither
```

The candidate and topic signals are constructed from centered seven-day sums.
The radar windows are explicit in `config/analysis.yml`.

`06_time_series.ipynb` contains one figure-specific exception: for
`Democratic concerns`, it treats `Democrats threaten democracy` as the positive
pole and `Republicans threaten democracy` as the negative pole. The
reproduction keeps that inversion local to Figure 4 through
`figure_4.democratic_concerns_notebook_orientation`; Figures 2, 3, and 5 retain
the manuscript-wide stance convention.

### Figure 5

Each post inherits the state exposure distribution of its page. As in
`geography_v3.ipynb`, page-state shares below `0.001` are set to zero without
renormalizing the remaining shares. Post likes are then distributed across
states. For every topic and state:

```text
stance bias = (pro likes - anti likes) / (pro + anti + neutral likes)
```

The election axis uses the Democratic minus Republican percentage-point margin,
so positive values indicate Democratic-leaning states. Figure 5 is exported as
separate components rather than one assembled panel:

```text
results/figure_5/figure_5_electoral_map_2020.*
results/figure_5/figure_5_electoral_map_2024.*
results/figure_5/figure_5_immigration_zscore_2020.*
results/figure_5/figure_5_immigration_zscore_2024.*
results/figure_5/figure_5_scatter_grid_2020.{pdf,png,svg}
results/figure_5/figure_5_scatter_grid_2024.{pdf,png,svg}
```

The map components are always written as interactive HTML and, when Kaleido is
available, as static PNG/PDF/SVG. The two scatter grids are Matplotlib ports of
`scatter_topics_grid_publication`, including the shared axes, stance-colored
points, fitted line, Pearson coefficient, and swing-state annotations in the
Wokeness panel.

## Reproducibility decisions

- The manuscript is the primary source for public topic names, stance meanings,
  formulas and figure thresholds.
- First-pass topic labels are used; the abandoned general second-pass topic
  classification is not part of the pipeline.
- The democracy-subtopic output is retained only to construct the final public
  topics described above.
- Pickles are accepted only as private preparation inputs. All public data use
  compressed CSV or JSONL.
- Figure scripts read only from the anonymized public folders, never from
  `data/temp`.

## Tests

```bash
pytest
```

The tests cover deterministic anonymization, lag selection, candidate support,
state-weighted stance bias, and the minimal public schemas.

## Before public release

1. Replace the placeholder `LICENSE` with the license agreed by all authors.
2. Review `data/manifest.json` and the public files.
3. Confirm that `data/temp`, `.anonymization_salt`, and `private_maps` are not
   tracked by Git.
4. Add the final repository URL and paper DOI to `CITATION.cff` when available.

### Correlation audit files

Figure 2 writes the complete matrices used by the original
`export_three_layer_correlations` workflow to `results/figure_2/data/`: intra-layer
rho/p matrices and inter-layer rho, p, lag, sample-size, selection-source, and
all-pair link tables. The matching-topic link files are the filtered inputs used
for the plotted arrows.

### Figure 3 output files

The manuscript panel was composed externally, so the reproduction script does
not create one combined image. For each year it writes the two original
components separately:

```text
results/figure_3/figure_3_heatmap_2020.{pdf,png,svg}
results/figure_3/figure_3_subnetwork_2020.{pdf,png,svg}
results/figure_3/figure_3_heatmap_2024.{pdf,png,svg}
results/figure_3/figure_3_subnetwork_2024.{pdf,png,svg}
```

The heatmap follows the pseudo-triangular layout of
`plot_three_layer_block_heatmap_from_export`: diagonal intra-layer blocks and
only the two candidate–public blocks below the diagonal. The subnetwork follows
`plot_three_subnet_topic_network_matplotlib`, including four nodes per layer,
stance-bias colors, topic-volume sizes, and lag-directed inter-layer arrows.
