# Private temporary inputs

Everything in this directory is ignored by Git, except this file. Put the
original Meta-derived files and classification outputs here before running
`python scripts/prepare_public_data.py`.

The complete expected tree and accepted filename variants are documented in
the root `README.md` and in `config/analysis.yml`.

Never commit:

- raw post text;
- names or usernames;
- links;
- the generated `.anonymization_salt`;
- `private_maps/`, which links original and public identifiers.


For candidate node colors, also place the separate speech-stance files used in
`07_correlations.ipynb` here, preferably under `speeches/` with names such as
`2024_speech_stance_trump_time_series.pkl`. Accepted alternatives are declared
in `config/analysis.yml`.
