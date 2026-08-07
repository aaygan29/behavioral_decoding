.PHONY: help install install-all test test-fast lint format demo clean

PYTHON ?= python3

help:
	@echo "install      install the package and dev tools"
	@echo "install-all  also install fmri, eeg, face, and vision extras"
	@echo "test         run the full test suite (~4 min)"
	@echo "test-fast    skip the slow multi-seed tests"
	@echo "lint         ruff check"
	@echo "format       ruff format and import sort"
	@echo "demo         run the synthetic ground-truth check"
	@echo "clean        remove caches and build artefacts"

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-all:
	$(PYTHON) -m pip install -e ".[dev,all]"

test:
	$(PYTHON) -m pytest

test-fast:
	$(PYTHON) -m pytest -k "not across_seeds and not nested_cv"

lint:
	$(PYTHON) -m ruff check src tests scripts

format:
	$(PYTHON) -m ruff format src tests scripts
	$(PYTHON) -m ruff check --select I --fix src tests scripts

demo:
	$(PYTHON) scripts/run_demo.py --quick

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage build dist
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
