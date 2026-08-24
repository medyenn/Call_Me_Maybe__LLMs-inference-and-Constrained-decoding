import json
from pathlib import Path
from typing import Any, Protocol

import pydantic

from .models import FunctionCatalog, PromptItem


def json_loader(path: Path) -> Any:
    """Read and parse a JSON file from disk."""
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
        reason = e.strerror or str(e)
        raise OSError(f'Could not read "{path}": {reason}') from e


def fcts_loader(path: Path) -> FunctionCatalog:
    """Load and validate the function catalog."""
    raw = json_loader(path)
    try:
        return FunctionCatalog(functions=raw)
    except pydantic.ValidationError as e:
        raise ValueError(f'Invalid function catalog in "{path}":\n{e}') from e


def prompt_loader(path: Path) -> list[PromptItem]:
    """Load and validate the list of prompts."""
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
    """Inverse of the standard GPT-2 byte<->unicode display map, so a
    raw vocab.json entry can be turned back into the bytes it encodes.
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
    """Turn one raw vocab.json entry into the literal text it
    represents (e.g. "Ġcat" -> " cat").
    """
    try:
        raw_bytes = bytes(_BYTE_DECODER[ch] for ch in token)
    except KeyError:
        return token
    return raw_bytes.decode("utf-8", errors="replace")


class VocabularyManager(pydantic.BaseModel):
    """Bidirectional token ID <-> string mapping. `id_to_token` holds
    the decoded text (real spaces, no BPE artifacts like "Ġ") that the
    grammar and generation loop actually walk over.
    """

    id_to_token: dict[int, str]
    token_to_id: dict[str, int]


class _VocabSource(Protocol):
    def get_path_to_vocab_file(self) -> str: ...


def vocab_loader(sdk: _VocabSource) -> VocabularyManager:
    """Load the tokenizer vocabulary via the SDK and build id<->token
    lookups.
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
