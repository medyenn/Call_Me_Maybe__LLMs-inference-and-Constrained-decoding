import sys
import pydantic
from .models import Config, FunctionCatalog, PromptItem
from pathlib import Path
import json


def json_loader(path: Path) -> dict | list:
    """..."""

    if path.exists():
        with open(Config.functions_definition) as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print('err', file=sys.stderr)
                sys.exit(1)
    print('err', file=sys.stderr)
    sys.exit(1)


def fcts_loader(path: Path) -> FunctionCatalog:
    """..."""

    fcts = json_loader(path)
    try:
        catalog = FunctionCatalog(functions=fcts)
    except pydantic.ValidationError:
        print('err')
        sys.exit(1)
    else:
        return catalog


def prompt_loader(path: Path) -> list[PromptItem]:
    """..."""

    prompts = json_loader(path)
    try:
        item = PromptItem(prompts)
    except pydantic.ValidationError:
        print('err')
        sys.exit(1)
    else:
        return item


def vocab_loader():
    ...
