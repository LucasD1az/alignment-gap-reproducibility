#!/usr/bin/env python3
from alignment_gap.config import load_config
from alignment_gap.figure3 import reproduce_figure_3

if __name__ == "__main__":
    for path in reproduce_figure_3(load_config()):
        print(path)
