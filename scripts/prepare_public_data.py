#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time

from alignment_gap.config import load_config
from alignment_gap.prepare import prepare_all
from alignment_gap.validation import assert_public_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Create anonymized public datasets from data/temp.")
    parser.add_argument("--config", default=None, help="Path to analysis.yml")
    args = parser.parse_args()
    config = load_config(args.config)
    started = time.perf_counter()
    manifest = prepare_all(config)
    assert_public_data(config)
    print(json.dumps(manifest, indent=2))
    print(f"Public data prepared and validated in {time.perf_counter() - started:.2f} seconds.")


if __name__ == "__main__":
    main()
