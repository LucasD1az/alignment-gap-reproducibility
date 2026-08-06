#!/usr/bin/env python3
from __future__ import annotations

import argparse

from alignment_gap.config import load_config
from alignment_gap.validation import validate_public_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the public data schemas and identifier integrity.")
    parser.add_argument("--config", default=None, help="Configuration path; defaults to config/analysis.yml")
    args = parser.parse_args()
    errors = validate_public_data(load_config(args.config))
    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("Public data validation passed.")


if __name__ == "__main__":
    main()
