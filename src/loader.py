import sys
import pydantic
from .models import FunctionCatalog, PromptItem
from pathlib import Path
import json


def json_loader(path: Path) -> dict | list:
    """..."""

    if path.exists():
        with open(path) as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                print(f'Invalid JSON format: {e.msg}', file=sys.stderr)
                sys.exit(1)
    print(f'The Path "{path}" Doese NOT Exist!', file=sys.stderr)
    sys.exit(1)


def fcts_loader(path: Path) -> FunctionCatalog:
    """..."""

    fcts = json_loader(path)
    try:
        return FunctionCatalog(functions=fcts)
    except pydantic.ValidationError as e:
        print(f'Invalid Input: {e.msg}', file=sys.stderr)
        sys.exit(1)


def prompt_loader(path: Path) -> list[PromptItem]:
    """..."""

    prompts = json_loader(path)
    try:
        return PromptItem(prompt=prompts)
    except pydantic.ValidationError:
        print('\nInvalid Input:\n')
        sys.exit(1)


def vocab_loader():
    ...
