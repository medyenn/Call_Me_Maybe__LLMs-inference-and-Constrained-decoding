import json
from pathlib import Path
from typing import Any, Protocol

import pydantic

from .exceptions import InputFileError, SchemaValidationError
from .models import FunctionCatalog, PromptItem


def json_loader(path: Path) -> Any:
    """Read and parse a JSON file from disk.

    Args:
        path: Path to the JSON file to read.

    Returns:
        The parsed JSON value (list, dict, str, etc. -- whatever the
        file contains).

    Raises:
        InputFileError: if the file does not exist, cannot be read
            (permissions, I/O error), or does not contain valid JSON.
    """
    if not path.exists():
        raise InputFileError(f'The path "{path}" does not exist.')

    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise InputFileError(
            f'Invalid JSON in "{path}": {e.msg} (line {e.lineno}, '
            f"column {e.colno})."
        ) from e
    except OSError as e:
        raise InputFileError(f'Could not read "{path}": {e}') from e


def fcts_loader(path: Path) -> FunctionCatalog:
    """Load and validate the function catalog from
    functions_definition.json.

    Args:
        path: Path to the function definitions JSON file.

    Returns:
        A validated FunctionCatalog.

    Raises:
        InputFileError: if the file is missing or not valid JSON.
        SchemaValidationError: if the JSON does not describe a valid
            function catalog (missing keys, empty catalog, duplicate
            function names, etc.).
    """
    raw = json_loader(path)
    try:
        return FunctionCatalog(functions=raw)
    except pydantic.ValidationError as e:
        raise SchemaValidationError(
            f'Invalid function catalog in "{path}":\n{e}'
        ) from e


def prompt_loader(path: Path) -> list[PromptItem]:
    """Load and validate the list of prompts from
    function_calling_tests.json.

    Args:
        path: Path to the prompts JSON file.

    Returns:
        A list of validated PromptItem objects (may be empty, which
        is a valid input -> an empty output array, not an error).

    Raises:
        InputFileError: if the file is missing, not valid JSON, or is
            not a JSON array.
        SchemaValidationError: if an entry is missing the "prompt"
            key or has a non-string value for it.
    """
    raw = json_loader(path)
    if not isinstance(raw, list):
        raise InputFileError(
            f'"{path}" must contain a JSON array of prompt objects.'
        )
    try:
        return [PromptItem(**item) for item in raw]
    except (pydantic.ValidationError, TypeError) as e:
        raise SchemaValidationError(
            f'Invalid prompt entry in "{path}":\n{e}'
        ) from e


class VocabularyManager(pydantic.BaseModel):
    """Bidirectional mapping between token IDs and their literal
    string representation, as produced by the model's tokenizer.

    This is built once (per Phase 10.3's performance guidance) and
    reused for every generation step, across every prompt.
    """

    id_to_token: dict[int, str]
    token_to_id: dict[str, int]


class _VocabSource(Protocol):
    """The subset of the llm_sdk SDK that vocab_loader depends on --
    kept narrow and explicit so this module never needs to know about
    the rest of Small_LLM_Model's public interface."""

    def get_path_to_vocab_file(self) -> str: ...


def vocab_loader(sdk: _VocabSource) -> VocabularyManager:
    """Load the tokenizer vocabulary via the SDK and build id<->token
    lookups.

    Args:
        sdk: An llm_sdk model instance exposing
            `get_path_to_vocab_file()`.

    Returns:
        A VocabularyManager with both directions of the mapping.

    Raises:
        InputFileError: if the vocab file cannot be found or parsed.
        SchemaValidationError: if the vocab file's shape is not the
            expected `{token_string: token_id}` mapping.
    """
    vocab_path = Path(sdk.get_path_to_vocab_file())
    raw_vocab = json_loader(vocab_path)

    if not isinstance(raw_vocab, dict):
        raise SchemaValidationError(
            f'Vocabulary file "{vocab_path}" does not contain a '
            "JSON object mapping token strings to token IDs."
        )

    token_to_id: dict[str, int] = raw_vocab
    id_to_token: dict[int, str] = {v: k for k, v in raw_vocab.items()}

    return VocabularyManager(token_to_id=token_to_id, id_to_token=id_to_token)
