#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from alignment_gap.config import load_config
from alignment_gap.preflight import check_private_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether the private data/temp inputs are present.")
    parser.add_argument("--config", default=None, help="Path to analysis.yml")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()
    config = load_config(args.config)
    rows = check_private_inputs(config)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            marker = "OK" if row["status"] == "found" else ("MISSING" if row["required"] else "OPTIONAL")
            year = f" [{row['year']}]" if row["year"] else ""
            detail = row["path"] or " | ".join(row["expected"])
            print(f"{marker:8} {row['input']}{year}: {detail}")
    missing = [row for row in rows if row["required"] and row["status"] == "missing"]
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
