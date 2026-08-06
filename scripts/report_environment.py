#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from alignment_gap.environment import collect_environment, write_environment_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Print Python, OS, hardware, and dependency versions.")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()
    report = collect_environment()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        path = write_environment_report(report, args.output)
        print(f"Environment report written to {path}")


if __name__ == "__main__":
    main()
