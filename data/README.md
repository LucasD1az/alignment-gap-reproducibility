# Public data layout

- `posts/`: anonymized post metadata and engagement counts.
- `labels/`: topic, topic-stance, and candidate-stance labels keyed by public post ID.
- `speeches/`: paragraph-level candidate, date, topic, and stance data without transcript text.
- `geography/`: anonymized page-to-state exposure shares and clean election results.
- `temp/`: private preparation inputs; ignored by Git except for its README.
- `derived/`: optional intermediate files; ignored by Git.

The complete anonymized research dataset is bundled and is used directly to validate the software and reproduce the manuscript figures. No separate simulated or subsampled demonstration dataset is maintained.

See the root `README.md`, `data/SCHEMAS.md`, and `docs/USING_YOUR_DATA.md` for schemas and preparation instructions.
