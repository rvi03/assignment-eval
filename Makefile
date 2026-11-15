.DEFAULT_GOAL := help
PYTHON := .venv/bin/python

.PHONY: help install clean test cli-help

##@ Setup

install: ## Create .venv with Python 3.12 and install dependencies (uv sync)
	@command -v uv >/dev/null 2>&1 || { echo "Error: uv is not installed. Install it from https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
	uv sync

clean: ## Remove .venv and generated caches
	rm -rf .venv .pytest_cache
	find . -type d -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} +

##@ Development

test: $(PYTHON) ## Run the test suite
	$(PYTHON) -m pytest

##@ Evaluator

cli-help: $(PYTHON) ## Show the evaluator's command-line help
	$(PYTHON) -m evaluator --help

##@ Help

help: ## Show this list (the default target)
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target>\n"} /^##@/ {printf "\n%s\n", substr($$0, 5)} /^[a-zA-Z_-]+:.*## / {printf "  %-10s  %s\n", $$1, $$2}' $(MAKEFILE_LIST)

$(PYTHON):
	@echo "Error: .venv not found. Run 'make install' first."; exit 1
