.PHONY: install environment check-inputs prepare validate reproduce verify test clean

install:
	python -m pip install -e ".[dev]"

environment:
	python scripts/report_environment.py --output results/environment.json

check-inputs:
	python scripts/check_temp_inputs.py

prepare:
	python scripts/prepare_public_data.py

validate:
	python scripts/validate_public_data.py

reproduce:
	python scripts/reproduce_all.py

verify: validate reproduce

test:
	pytest

clean:
	rm -rf results/figure_* results/reproduction_runtime.json data/derived/*
