NAME		= call_me_maybe

PYTHON		= uv run python
ENTRY		= -m src

all: run

install:
	uv sync

# Run the main script under Python's built-in debugger.
debug:
	$(PYTHON) -m pdb $(ENTRY)

# Forward any extra CLI flags: `make run ARGS="--input foo.json"`.
run:
	$(PYTHON) $(ENTRY) $(ARGS)

clean:
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .mypy_cache .pytest_cache .ruff_cache

re: clean install

# Mandatory lint target (subject IV.2): flake8 + the specified mypy flags.
lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs

# Optional, stricter variant (subject IV.2).
lint-strict:
	uv run flake8 .
	uv run mypy . --strict

# Not graded (subject IV.3), but kept for local development.
test:
	uv run pytest

docs:
	@echo "Open README.md"

help:
	@echo "Available targets:"
	@echo "  install      Install dependencies (uv sync)"
	@echo "  run          Run the project (ARGS=... to forward CLI flags)"
	@echo "  debug        Run the project under pdb"
	@echo "  clean        Remove Python/mypy/pytest caches"
	@echo "  re           clean + install"
	@echo "  lint         Run flake8 + mypy (mandatory flags)"
	@echo "  lint-strict  Run flake8 + mypy --strict"
	@echo "  test         Run pytest (not graded, recommended)"

.PHONY: all install run debug clean re lint lint-strict test docs help
