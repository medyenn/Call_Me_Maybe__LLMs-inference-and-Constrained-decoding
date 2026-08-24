NAME		= call_me_maybe
PYTHON		= uv run python
ENTRY		= -m src

all: run

install:
	@uv sync

debug:
	@$(PYTHON) -m pdb $(ENTRY)

run:
	@$(PYTHON) $(ENTRY) $(ARGS)

clean:
	@rm -rf __pycache__
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@rm -rf .mypy_cache .pytest_cache .ruff_cache

re: clean install

lint:
	@uv run flake8 src
	@uv run mypy src --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs

test:
	@uv run pytest

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
