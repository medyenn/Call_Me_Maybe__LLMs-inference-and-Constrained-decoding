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

from .exceptions import SchemaValidationError, InputFileError
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
        SchemaValidationError: if the generated text does not parse,
            does not have the expected shape, names a function that
            does not exist in `catalog`, is missing a required
            parameter, has an extra/unknown parameter, or has a
            parameter whose value does not match its declared type.
    """
    try:
        parsed = json.loads(generated_text)
    except json.JSONDecodeError as e:
        raise SchemaValidationError(
            f"Generated output for prompt {prompt!r} is not valid "
            f"JSON despite constrained decoding: {e}"
        ) from e

    if not isinstance(parsed, dict) or set(parsed) != {"name", "parameters"}:
        raise SchemaValidationError(
            f"Generated output for prompt {prompt!r} does not have "
            f'the expected {{"name", "parameters"}} shape: {parsed!r}'
        )

    name = parsed["name"]
    parameters = parsed["parameters"]

    function_def = catalog.get(name)
    if function_def is None:
        raise SchemaValidationError(
            f"Generated function name {name!r} for prompt {prompt!r} "
            "does not exist in the function catalog."
        )

    if not isinstance(parameters, dict):
        raise SchemaValidationError(
            f'Generated "parameters" for prompt {prompt!r} is not '
            f"a JSON object: {parameters!r}"
        )

    expected_keys = set(function_def.parameters)
    actual_keys = set(parameters)
    if actual_keys != expected_keys:
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        raise SchemaValidationError(
            f"Parameters for {name!r} (prompt {prompt!r}) do not match "
            f"the declared signature. Missing: {sorted(missing)}. "
            f"Extra: {sorted(extra)}."
        )

    for key, value in parameters.items():
        expected_type = function_def.parameters[key].type
        if not _value_matches_declared_type(value, expected_type):
            raise SchemaValidationError(
                f"Parameter {key!r} of {name!r} (prompt {prompt!r}) "
                f"should be of type {expected_type!r} "
                f"({PYTHON_TYPE_BY_PARAM_TYPE[expected_type].__name__}), "
                f"got {value!r} ({type(value).__name__})."
            )

    return FunctionCallResult(prompt=prompt, name=name, parameters=parameters)


def write_results(
        results: list[FunctionCallResult], output_path: Path) -> None:
    """Serialize the full list of results to `output_path` as a JSON
    array, creating parent directories if needed.

    Args:
        results: The validated results, in the same order as the
            input prompts.
        output_path: Where to write the final JSON file.

    Raises:
        InputFileError: if the parent directory cannot be created, or
            the file cannot be written (permissions, disk errors).
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise InputFileError(
            f'Could not create output directory "{output_path.parent}": {e}'
        ) from e

    payload = [result.model_dump() for result in results]

    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
    except OSError as e:
        raise InputFileError(
            f'Could not write output file "{output_path}": {e}'
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
