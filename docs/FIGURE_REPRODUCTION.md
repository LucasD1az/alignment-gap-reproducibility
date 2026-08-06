# Figure reproduction details

This document records the outputs and central methodological settings used by the public reproduction scripts. The manuscript is the primary source for public topic names, stance definitions, formulas, and final figure interpretation. `config/analysis.yml` is the executable source for periods, thresholds, and plotting parameters.

## Figure 1

The final alluvial graphic was assembled outside Python. The script exports:

```text
results/figure_1/figure_1_2016.csv
results/figure_1/figure_1_2020.csv
results/figure_1/figure_1_2024.csv
results/figure_1/figure_1_all_years.csv
```

The tables contain topic, stance, post count, likes, total reactions, shares, and ordering. Topics below 3% of annual likes are grouped as `Others`.

## Figures 2 and 3

The public-reaction layer is the centered seven-day mean of daily likes per post for each topic. Candidate layers are centered seven-day means of daily speech-paragraph counts. Correlations are Spearman correlations.

Figure 2 uses the final multilayer layout: three elliptical topic rings on perspective planes, within-layer links, and lag-directed links between matching topics in adjacent layers. Candidate node colors use the separate speech-stance aggregate files. Correlation periods, lag range, significance levels, topic order, and magnitude thresholds are in `config/analysis.yml`.

Figure 3 evaluates the complete rectangular candidate–public blocks, including different-topic comparisons. It writes the two externally composable pieces separately:

```text
results/figure_3/figure_3_heatmap_2020.{pdf,png,svg}
results/figure_3/figure_3_subnetwork_2020.{pdf,png,svg}
results/figure_3/figure_3_heatmap_2024.{pdf,png,svg}
results/figure_3/figure_3_subnetwork_2024.{pdf,png,svg}
```

The heatmap uses diagonal within-layer blocks and the two candidate–public blocks below the diagonal. The subnetwork contains four topics per layer, topic-volume node sizes, stance-bias colors, and lag-directed inter-layer arrows.

Every correlation matrix, lag table, link table, and node table used by Figures 2 and 3 is written under the corresponding `results/figure_*/data/` directory.

## Figure 4

Candidate support is computed as:

```text
(Pro-Democrat likes + Anti-Trump likes)
- (Pro-Trump likes + Anti-Democrat likes)
------------------------------------------------
all candidate-classified likes, including Neither
```

Topic stance bias is:

```text
(pro likes - anti likes) / (pro + anti + neutral likes)
```

The repository uses the manuscript-wide Democratic concerns convention:

- positive: `Republicans threaten democracy`;
- negative: `Democrats threaten democracy`.

The script writes the time-series and radar components separately:

```text
results/figure_4/figure_4_support_stance_2020.*
results/figure_4/figure_4_radars_2020.*
results/figure_4/figure_4_support_stance_2024.*
results/figure_4/figure_4_radars_2024.*
```

Reference dates and radar windows are explicit in `config/analysis.yml`.

## Figure 5

Each post inherits the state exposure distribution of its page. Page-state shares below `0.001` are set to zero without renormalizing the remaining shares, matching the final geography notebook. Post likes are distributed across states, and state-topic stance bias is computed from pro, anti, and neutral likes.

The election axis is Democratic percentage minus Republican percentage. The script writes independent components:

```text
results/figure_5/figure_5_electoral_map_2020.*
results/figure_5/figure_5_electoral_map_2024.*
results/figure_5/figure_5_immigration_zscore_2020.*
results/figure_5/figure_5_immigration_zscore_2024.*
results/figure_5/figure_5_scatter_grid_2020.{pdf,png,svg}
results/figure_5/figure_5_scatter_grid_2024.{pdf,png,svg}
```

Map components are always written as interactive HTML. Static PNG/PDF/SVG map export is attempted when Kaleido and a compatible Chrome installation are available.

## Reproducibility decisions

- First-pass topic labels are used; the abandoned general second-pass classification is not part of the analysis.
- Democracy subtopic labels are retained only to construct the final public democracy-related topics.
- Pickles are accepted only as private preparation inputs. Public files use compressed CSV or JSONL.
- Figure scripts read only the public anonymized folders and never read `data/temp/`.
- `MIGRATION_NOTES.md` documents consolidation decisions and known differences from exploratory notebooks.
