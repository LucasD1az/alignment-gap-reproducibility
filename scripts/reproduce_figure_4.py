#!/usr/bin/env python3
from alignment_gap.config import load_config
from alignment_gap.figure4 import reproduce_figure_4

if __name__ == "__main__":
    for path in reproduce_figure_4(load_config()):
        print(path)
