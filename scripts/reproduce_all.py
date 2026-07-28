#!/usr/bin/env python3
from __future__ import annotations

import argparse

from alignment_gap.config import load_config
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
    for number, function in functions.items():
        if number in args.skip:
            continue
        print(f"[Figure {number}]")
        for path in function(config):
            print(f"  {path}")


if __name__ == "__main__":
    main()
