.PHONY: install check-inputs prepare validate reproduce test clean

install:
	python -m pip install -e ".[dev]"

check-inputs:
	python scripts/check_temp_inputs.py

prepare:
	python scripts/prepare_public_data.py

validate:
	python scripts/validate_public_data.py

reproduce:
	python scripts/reproduce_all.py

test:
	pytest

clean:
	rm -rf results/figure_* data/derived/*
