#!/usr/bin/env python3
from alignment_gap.config import load_config
from alignment_gap.validation import validate_public_data

if __name__ == "__main__":
    errors = validate_public_data(load_config())
    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("Public data validation passed.")
