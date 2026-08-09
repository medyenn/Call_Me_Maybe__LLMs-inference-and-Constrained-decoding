"""Output Assembler & Output Writer (Phase 10 components G and H).

The constrained-decoding engine already guarantees that whatever text
it produces is syntactically valid JSON that matches the grammar it
was walking. This module is the "belt and suspenders" layer: it
parses that text, re-validates it against the *actual* FunctionCatalog
one more time, and only then turns it into the exact 3-key record the
subject requires (V.4) -- catching any lingering engine bug before it
ever reaches the deliverable file.
"""

import json
import sys
from pathlib import Path
from typing import Any

from .models import (
    FunctionCallResult,
    FunctionCatalog,
    PYTHON_TYPE_BY_PARAM_TYPE)


def _value_matches_declared_type(value: object, param_type: str) -> bool:
    """Check a Python value against one of our declared parameter
    types.

    "number" accepts both `int` and `float` (JSON's "number" does not
    distinguish the two -- see subject V.4.1's own `2.0`/`3.0` example,
    which is a formatting choice, not a hard type split). "integer" is
    a stricter subset requiring a whole number. `bool` is excluded from
    both, since `bool` is a subclass of `int` in Python and must never
    silently satisfy a numeric parameter.

    Args:
        value: The already-parsed Python value (from `json.loads`).
        param_type: One of "string", "number", "integer", "boolean".

    Returns:
        True if `value`'s Python type is an acceptable match.
    """
    if param_type == "boolean":
        return isinstance(value, bool)
    if isinstance(value, bool):
        return False
    if param_type == "integer":
        return isinstance(value, int)
    if param_type == "number":
        return isinstance(value, (int, float))
    if param_type == "string":
        return isinstance(value, str)
    return False


def _strip_copied_quotes(value: str) -> str:
    """Drop a matching pair of leading/trailing quote marks from a
    generated string value.

    The model sometimes copies a quoted word straight out of the
    natural-language prompt (e.g. "Reverse the string 'hello'")
    quote marks and all, instead of just the word itself. Those marks
    were never meant to be part of the *value* -- JSON's own double
    quotes already delimit the string -- so a leading/trailing `'` or
    `"` pair on an otherwise-plain value is stripped as a copy-paste
    artifact rather than kept as real content.

    Args:
        value: The raw string value parsed out of the generated JSON.

    Returns:
        `value` with one matching pair of surrounding quote marks
        removed, or unchanged if there is no such matching pair.
    """
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _coerce_value(value: Any, param_type: str) -> object:
    """Normalize an already type-checked parameter value into the
    exact Python representation its declared type calls for.

    "number" is always emitted as a `float` (so a generated `5`
    becomes `5.0`), matching subject V.4.1's own example, even though
    JSON itself doesn't distinguish `5` from `5.0`. "integer" stays a
    plain `int`. "string" has copy-pasted quote marks stripped (see
    `_strip_copied_quotes`). "boolean" passes through unchanged.

    Args:
        value: The already type-checked value.
        param_type: One of "string", "number", "integer", "boolean".

    Returns:
        The normalized value.
    """
    if param_type == "number":
        return float(value)
    if param_type == "integer":
        return int(value)
    if param_type == "string":
        return _strip_copied_quotes(value)
    return value


def assemble_result(
    prompt: str,
    generated_text: str,
    catalog: FunctionCatalog,
) -> FunctionCallResult:
    """Turn the decoder's raw `{"name": ..., "parameters": {...}}` text
    for one prompt into a validated FunctionCallResult.

    Args:
        prompt: The original natural-language request, copied
            verbatim into the output record.
        generated_text: The raw JSON text produced by
            `decoder.generate_function_call` for this prompt.
        catalog: The real function catalog to validate against.

    Returns:
        A FunctionCallResult with exactly the three required keys.

    Raises:
        ValueError: if the generated text does not parse, does not
            have the expected shape, names a function that does not
            exist in `catalog`, is missing a required parameter, has
            an extra/unknown parameter, or has a parameter whose
            value does not match its declared type.
    """
    try:
        parsed = json.loads(generated_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Generated output for prompt {prompt!r} is not valid "
            f"JSON despite constrained decoding: {e}"
        ) from e

    if not isinstance(parsed, dict) or set(parsed) != {"name", "parameters"}:
        raise ValueError(
            f"Generated output for prompt {prompt!r} does not have "
            f'the expected {{"name", "parameters"}} shape: {parsed!r}'
        )

    name = parsed["name"]
    parameters = parsed["parameters"]

    function_def = catalog.get(name)
    if function_def is None:
        raise ValueError(
            f"Generated function name {name!r} for prompt {prompt!r} "
            "does not exist in the function catalog."
        )

    if not isinstance(parameters, dict):
        raise ValueError(
            f'Generated "parameters" for prompt {prompt!r} is not '
            f"a JSON object: {parameters!r}"
        )

    expected_keys = set(function_def.parameters)
    actual_keys = set(parameters)
    if actual_keys != expected_keys:
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        raise ValueError(
            f"Parameters for {name!r} (prompt {prompt!r}) do not match "
            f"the declared signature. Missing: {sorted(missing)}. "
            f"Extra: {sorted(extra)}."
        )

    normalized_parameters: dict[str, Any] = {}
    for key, value in parameters.items():
        expected_type = function_def.parameters[key].type
        if not _value_matches_declared_type(value, expected_type):
            raise ValueError(
                f"Parameter {key!r} of {name!r} (prompt {prompt!r}) "
                f"should be of type {expected_type!r} "
                f"({PYTHON_TYPE_BY_PARAM_TYPE[expected_type].__name__}), "
                f"got {value!r} ({type(value).__name__})."
            )
        normalized_parameters[key] = _coerce_value(value, expected_type)

    return FunctionCallResult(
        prompt=prompt, name=name, parameters=normalized_parameters)


def write_results(
        results: list[FunctionCallResult], output_path: Path) -> None:
    """Serialize the full list of results to `output_path` as a JSON
    array, creating parent directories if needed.

    Args:
        results: The validated results, in the same order as the
            input prompts.
        output_path: Where to write the final JSON file.

    Raises:
        OSError: if the parent directory cannot be created, or the
            file cannot be written (permissions, disk errors, etc.).
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        reason = e.strerror or str(e)
        raise OSError(
            f'Could not create output directory '
            f'"{output_path.parent}": {reason}'
        ) from e

    payload = [result.model_dump() for result in results]

    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
    except OSError as e:
        reason = e.strerror or str(e)
        raise OSError(
            f'Could not write output file "{output_path}": {reason}'
        ) from e


def report_skipped_prompt(prompt: str, error: Exception) -> None:
    """Log (to stderr, never crashing the run) that one prompt was
    skipped from the output.

    Design decision (documented in README): a single prompt that
    cannot be resolved into a valid, schema-compliant function call
    does not abort the whole batch. It is skipped with a clear
    warning, and every other prompt is still processed and written.

    Args:
        prompt: The prompt that was skipped.
        error: The exception explaining why.
    """
    print(f"Skipping prompt {prompt!r}: {error}", file=sys.stderr)
