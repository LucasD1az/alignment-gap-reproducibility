# Migration notes from the analysis notebooks

This repository is a deliberately smaller public reproduction layer. It does
not preserve the exploratory notebook history cell by cell. Instead, it turns
the final analysis into explicit preparation and figure stages.

## Source-of-truth rule

The manuscript is authoritative for final public topic names, stance meanings,
formulas, campaign windows and figure thresholds. The notebooks are used to
recover file formats, rolling-window conventions and implementation details.
When they conflict, the public pipeline follows the manuscript unless the final
notebook workflow clearly requires otherwise.

Important consequences:

- first-pass topic classification is authoritative;
- the abandoned general second-pass topic correction is excluded;
- the democracy subtopic classifier is retained only to split the legacy
  `Danger to democracy` macro-topic into the final public topics;
- `Parties, leadership and democratic responsibility` is published as
  `Democratic concerns`;
- the Democratic-concerns stance follows the manuscript convention in every
  figure: `Republicans threaten democracy` is +1 and `Democrats threaten
  democracy` is −1; the opposite helper mapping left in `06_time_series.ipynb`
  is treated as an implementation error rather than part of the final method;
- `Healthcare/Science`, `Taxes`, and `LGBT issues` are published as
  `Healthcare`, `Economy`, and `Wokeness`, respectively;
- correlations use centered seven-day means and Spearman correlation: daily
  paragraph counts for speech layers and daily likes per post for the public
  reaction layer, with non-zero lags searched within ±14 days;
- the election margin in Figure 5 is Democratic minus Republican vote share.

## Notebook-to-module map

| Previous material | Public replacement |
|---|---|
| `00_preprocess_datasets.ipynb` | private input contract plus `prepare.py` |
| `01_topic_classification.ipynb` | excluded from this release; consumes its first-pass output |
| `02_second_pass_classification.ipynb` | excluded |
| `03_danger_to_democracy.ipynb` | democracy-subtopic merge in `prepare.py` |
| `04_stance.ipynb` | stance merge and canonicalization in `prepare.py` |
| `03_topics_in_time.ipynb`, `06_time_series.ipynb` | `series.py`, Figure 1 and Figure 4 modules |
| `07_correlations.ipynb`, `07_speeches_bis*.ipynb` | `correlations.py`, `layers.py`, Figures 2 and 3 |
| `geography_v3.ipynb` | `figure5.py` plus geography preparation |
| validation, polls, keywords, subtopics, sentiment notebooks | intentionally deferred |

## What is intentionally not public yet

- raw Facebook text and page metadata;
- classification prompts and model execution;
- topic/subtopic validation;
- Table 1 and supplementary figures;
- exploratory robustness checks and poll comparisons;
- private raw-to-public ID maps.

## Geographic estimator

The manuscript writes the geographic calculation probabilistically. With the
available page-level ad shares and deterministic post labels, the implemented
estimator is the direct weighted form of that equation: each post's likes are
allocated across states using its page's normalized ad-impression distribution,
then aggregated by topic and stance. No MCMC sampler is required for this point
estimate.

## Release checks

The preparation command writes `data/manifest.json`, including row counts,
file sizes, and SHA-256 checksums. The validation command checks exact schemas,
one-to-one post/label keys, ID prefixes, forbidden fields, duplicate IDs, and
page exposure normalization.
## v0.1.2 corrections

- Fixed Figure 3 hierarchical clustering for pandas/NumPy configurations that expose `DataFrame.values` as read-only. The distance matrix is now created as an explicit writable NumPy copy and clustered using absolute correlation, as in the original notebook.
- Replaced the provisional three-dimensional Figure 2 implementation with a static port of the final `plot_multilayer_topic_network` layout from `07_speeches_bis.ipynb`.
- Figure 2 now uses explicit year-specific topic rings, same-topic inter-layer lagged correlations only, significance `p < 0.01`, and the original year-specific intra/inter magnitude thresholds.
- Node sizes now use unsmoothed total likes or paragraph counts; centered seven-day series are used only for correlations.
- The public correlation layer now reconstructs the notebook's daily `likes / posts` series before applying the centered seven-day mean; it no longer correlates total daily likes.
- Candidate speech series now use the notebook's centered seven-day mean rather than a rolling sum.


## Correlation reproduction update (v0.1.4)

Figures 2 and 3 now reproduce the effective correlation workflow of
`export_three_layer_correlations` in `07_correlations.ipynb`, including sparse
speech dates, missing public topic-days, centered rolling means with
`min_periods=1`, legacy date-index alignment after lag slicing, significant-first
lag selection with fallback, and full all-topic-pair inter-layer exports. See
`PATCH_NOTES_v0.1.4.md` for details.

## v0.1.5: candidate stance source and Figure 3 composition

The candidate stance bias used for node colors is sourced from the separate
speech stance time-series loaded by `load_stance_candidatestance_wide` in
`07_correlations.ipynb`, rather than inferred from the topic-count time series.
The public pipeline now stores those counts in a minimal daily long table.

The paper's Figure 3 was manually composed from two independent exports. The
repository now reproduces those exports separately: a pseudo-triangular block
heatmap and a four-topic reduced network for each election year.

## v0.1.6: Figure 4 polarity and decomposed Figure 5

`06_time_series.ipynb` contains a Figure-4-specific override for the legacy
`Parties, leadership and democratic responsibility` topic: it assigns
`Democrats threaten democracy` to the positive pole and `Republicans threaten
democracy` to the negative pole. The reproduction now preserves that inversion
only in Figure 4; the remaining figures keep the manuscript-wide convention.

Figure 5 is now exported as the separate pieces used for manual composition in
the paper: an electoral map and an Immigration stance-bias z-score map for each
year, plus one four-topic scatter grid for each year. The geographic aggregation
now also mirrors `geography_v3.ipynb` by applying the 0.001 page-state threshold
without renormalizing the retained weights. Prepared exposure tables preserve
both the original state value and the normalized page-state share.

## v0.1.8: corrected Figure 4 orientation and separate exports

Figure 4 now follows the manuscript polarity for Democratic concerns, removing
the former `democratic_concerns_notebook_orientation` configuration switch.
The support/stance time series and the four radar profiles are exported as
separate files for each year, matching the workflow based on
`plot_support_bump_stance_v2` and `plot_multiple_stance_radars` in
`06_time_series.ipynb`. The older v0.1.6 note above documents a superseded
intermediate reproduction decision.
