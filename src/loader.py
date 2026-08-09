import json
from pathlib import Path
from typing import Any, Protocol

import pydantic

from .models import FunctionCatalog, PromptItem


def json_loader(path: Path) -> Any:
    """Read and parse a JSON file from disk.

    Args:
        path: Path to the JSON file to read.

    Returns:
        The parsed JSON value (list, dict, str, etc. -- whatever the
        file contains).

    Raises:
        FileNotFoundError: if the file does not exist.
        OSError: if the file exists but cannot be read (permissions,
            it's a directory, disk error, etc.).
        ValueError: if the file does not contain valid JSON.
    """
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f'File not found: "{path}"') from None
    except json.JSONDecodeError as e:
        raise ValueError(
            f'Invalid JSON in "{path}": {e.msg} '
            f"(line {e.lineno}, column {e.colno})."
        ) from e
    except OSError as e:
        # Catches every other OS-level failure (permissions, path is a
        # directory, disk error, etc.) with one clear message.
        reason = e.strerror or str(e)
        raise OSError(f'Could not read "{path}": {reason}') from e


def fcts_loader(path: Path) -> FunctionCatalog:
    """Load and validate the function catalog from
    functions_definition.json.

    Args:
        path: Path to the function definitions JSON file.

    Returns:
        A validated FunctionCatalog.

    Raises:
        FileNotFoundError / OSError: if the file is missing or unreadable.
        ValueError: if the JSON is malformed, or does not describe a
            valid function catalog (missing keys, empty catalog,
            duplicate function names, etc.).
    """
    raw = json_loader(path)
    try:
        return FunctionCatalog(functions=raw)
    except pydantic.ValidationError as e:
        raise ValueError(f'Invalid function catalog in "{path}":\n{e}') from e


def prompt_loader(path: Path) -> list[PromptItem]:
    """Load and validate the list of prompts from
    function_calling_tests.json.

    Args:
        path: Path to the prompts JSON file.

    Returns:
        A list of validated PromptItem objects (may be empty, which
        is a valid input -> an empty output array, not an error).

    Raises:
        FileNotFoundError / OSError: if the file is missing or unreadable.
        ValueError: if the file is not valid JSON, is not a JSON
            array, or one of its entries is missing the "prompt" key
            or has a non-string value for it.
    """
    raw = json_loader(path)
    if not isinstance(raw, list):
        raise ValueError(
            f'"{path}" must contain a JSON array of prompt objects.'
        )
    try:
        return [PromptItem(**item) for item in raw]
    except (pydantic.ValidationError, TypeError) as e:
        raise ValueError(f'Invalid prompt entry in "{path}":\n{e}') from e


def _gpt2_byte_decoder() -> dict[str, int]:
    """Build the inverse of the standard GPT-2 byte<->unicode map.

    `vocab.json` does not store literal text: like every byte-level
    BPE tokenizer descended from GPT-2 (Qwen included), each token
    string is a *re-encoding* of raw bytes into a printable unicode
    alphabet, so every byte has a visible character (e.g. the space
    byte 0x20 is written as "Ġ", newline as "Ċ", etc.). Reversing that
    map lets us turn a vocab entry back into the real bytes it stands
    for.

    Returns:
        A mapping from each of those 256 display characters back to
        the raw byte value (0-255) it encodes.
    """
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    display_chars = byte_values[:]
    n = 0
    for b in range(256):
        if b not in byte_values:
            byte_values.append(b)
            display_chars.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(byte_values, display_chars)}


_BYTE_DECODER = _gpt2_byte_decoder()


def _decode_vocab_token(token: str) -> str:
    """Turn one raw `vocab.json` entry into the literal text it
    represents (e.g. "Ġcat" -> " cat"), by reversing the byte-level
    BPE display encoding and decoding the recovered bytes as UTF-8.

    Args:
        token: The raw token string exactly as it appears in
            `vocab.json`.

    Returns:
        The decoded text. Falls back to the original string unchanged
        for entries that aren't part of the byte-level alphabet (e.g.
        literal special tokens some tokenizers add directly).
    """
    try:
        raw_bytes = bytes(_BYTE_DECODER[ch] for ch in token)
    except KeyError:
        return token
    return raw_bytes.decode("utf-8", errors="replace")


class VocabularyManager(pydantic.BaseModel):
    """Bidirectional mapping between token IDs and their literal
    string representation, as produced by the model's tokenizer.

    `id_to_token` holds the *decoded* text each token represents
    (real spaces, real punctuation) -- this is what the grammar/
    schema resolver and the generation loop actually walk over, so
    generated text never contains raw byte-level-BPE artifacts like
    "Ġ". `token_to_id` keeps the raw `vocab.json` keys, since nothing
    downstream needs the decoded form for that direction.

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
        FileNotFoundError / OSError: if the vocab file cannot be found
            or read.
        ValueError: if the vocab file isn't valid JSON, or its shape
            is not the expected `{token_string: token_id}` mapping.
    """
    vocab_path = Path(sdk.get_path_to_vocab_file())
    raw_vocab = json_loader(vocab_path)

    if not isinstance(raw_vocab, dict):
        raise ValueError(
            f'Vocabulary file "{vocab_path}" does not contain a '
            "JSON object mapping token strings to token IDs."
        )

    token_to_id: dict[str, int] = raw_vocab
    id_to_token: dict[int, str] = {
        token_id: _decode_vocab_token(token_str)
        for token_str, token_id in raw_vocab.items()
    }

    return VocabularyManager(token_to_id=token_to_id, id_to_token=id_to_token)
