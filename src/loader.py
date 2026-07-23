import sys
import pydantic
import json
from pathlib import Path
from .models import FunctionCatalog, PromptItem
from typing import Any


def json_loader(path: Path) -> dict | list:
    """..."""

    if path.exists():
        with path.open() as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                print(f'Invalid JSON format: {e.msg}', file=sys.stderr)
                sys.exit(1)
            except OSError as e:
                print(f'{e}')
    print(f'The Path "{path}" Doese NOT Exist!', file=sys.stderr)
    sys.exit(1)


def fcts_loader(path: Path) -> FunctionCatalog:
    """..."""

    fcts = json_loader(path)
    try:
        return FunctionCatalog(functions=fcts)
    except pydantic.ValidationError as e:
        print(f'Invalid Input in "{path}":\n{e}', file=sys.stderr)
        sys.exit(1)


def prompt_loader(path: Path) -> list[PromptItem]:
    """..."""

    prompts = json_loader(path)
    try:
        return [PromptItem(**item) for item in prompts]
    except pydantic.ValidationError as e:
        print(f'Invalid Input in "{path}":\n{e}', file=sys.stderr)
        sys.exit(1)


class VocabularyManager(pydantic.BaseModel):
    id_to_token: dict[int, str]
    token_to_id: dict[str, int]


def vocab_loader(sdk: Any) -> VocabularyManager:
    """Load the tokenizer vocabulary via the SDK and
        build id<->token lookups."""
    vocab_path = Path(sdk.get_path_to_vocab_file())
    raw_vocab = json_loader(vocab_path)

    token_to_id: dict[str, int] = raw_vocab
    id_to_token: dict[int, str] = {v: k for k, v in raw_vocab.items()}

    return VocabularyManager(token_to_id=token_to_id, id_to_token=id_to_token)
