
NAME		= call_me_maybe

PYTHON		= uv run python
ENTRY		= -m src

all: run

install:
	uv sync

debug:
	python3 run pdb

run:
	$(PYTHON) $(ENTRY)

clean:
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

fclean: clean
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache

re: fclean install

format:
	uv run black src

lint:
	uv run flake8 src

typecheck:
	uv run mypy src

test:
	uv run pytest

bonus:
	$(PYTHON) -m bonus

docs:
	@echo "Open README.md"

help:
	@echo "Available targets:"
	@echo "  install    Install dependencies"
	@echo "  run        Run the project"
	@echo "  clean      Remove Python cache"
	@echo "  fclean     Remove all generated cache"
	@echo "  re         Reinstall dependencies"
	@echo "  test       Run tests"
	@echo "  lint       Run flake8"
	@echo "  typecheck  Run mypy"

.PHONY: all install run clean fclean re