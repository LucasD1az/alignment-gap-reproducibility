#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from alignment_gap.config import load_config, repo_path
from alignment_gap.environment import collect_environment
from alignment_gap.figure1 import reproduce_figure_1
from alignment_gap.figure2 import reproduce_figure_2
from alignment_gap.figure3 import reproduce_figure_3
from alignment_gap.figure4 import reproduce_figure_4
from alignment_gap.figure5 import reproduce_figure_5
from alignment_gap.validation import assert_public_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce all main-paper figures from public data.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip", nargs="*", default=[], choices=["1", "2", "3", "4", "5"])
    args = parser.parse_args()
    config = load_config(args.config)
    assert_public_data(config)
    functions = {
        "1": reproduce_figure_1,
        "2": reproduce_figure_2,
        "3": reproduce_figure_3,
        "4": reproduce_figure_4,
        "5": reproduce_figure_5,
    }
    started_all = time.perf_counter()
    timings: dict[str, float] = {}
    outputs: list[str] = []
    for number, function in functions.items():
        if number in args.skip:
            continue
        print(f"[Figure {number}]")
        started = time.perf_counter()
        paths = function(config)
        timings[number] = round(time.perf_counter() - started, 3)
        for path in paths:
            path = Path(path)
            try:
                display = path.resolve().relative_to(Path(config["_repo_root"]).resolve()).as_posix()
            except ValueError:
                display = str(path)
            outputs.append(display)
            print(f"  {display}")
        print(f"  elapsed: {timings[number]:.2f} s")

    total = time.perf_counter() - started_all
    report = {
        "config": str(config["_config_path"]),
        "figure_seconds": timings,
        "total_seconds": round(total, 3),
        "outputs": outputs,
        "environment": collect_environment(config["_repo_root"]),
    }
    results = repo_path(config, "results")
    results.mkdir(parents=True, exist_ok=True)
    report_path = results / "reproduction_runtime.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Full reproduction completed in {total:.2f} seconds.")
    print(f"Runtime report: {report_path}")


if __name__ == "__main__":
    main()
