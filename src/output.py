"""Parses and validates the decoder's generated text into the final
3-key output record, or a best-effort fallback when generation didn't
finish, then writes the results file.
"""

import json
import re
import sys
from pathlib import Path
from typing import cast, Any

from .models import (
    FunctionCallResult,
    FunctionCatalog,
    PYTHON_TYPE_BY_PARAM_TYPE)


def _value_matches_declared_type(value: object, param_type: str) -> bool:
    """Check a parsed value against one of our declared parameter types."""
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
    """Drop a matching pair of leading/trailing quote marks, a common
    artifact of the model copying a quoted word out of the prompt.
    """
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _coerce_value(value: object, param_type: str) -> object:
    """Normalize a type-checked value into its declared type."""

    if param_type == "number":
        return float(cast(float, value))

    if param_type == "integer":
        return int(cast(int, value))

    if param_type == "string":
        return _strip_copied_quotes(cast(str, value))

    return value


def assemble_result(
    prompt: str,
    generated_text: str,
    catalog: FunctionCatalog,
) -> FunctionCallResult:
    """Turn the decoder's raw JSON text for one prompt into a
    validated FunctionCallResult.
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

    normalized_parameters: dict = {}
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


def _extract_partial_name(partial_text: str) -> str | None:
    """Best-effort recovery of a `"name": "..."` value from
    possibly-incomplete text.
    """
    match = re.search(r'"name"\s*:\s*"([^"]*)"', partial_text)
    return match.group(1) if match else None


def _extract_partial_value(partial_text: str, key: str) -> Any:
    """Best-effort recovery of a complete `"<key>": <value>` pair's
    value from possibly-incomplete text; a value cut off mid-generation
    won't match, which is the correct outcome.
    """
    pattern = (
        rf'"{re.escape(key)}"\s*:\s*'
        r'("(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?|true|false)'
    )
    match = re.search(pattern, partial_text)
    if match is None:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def assemble_fallback_result(
    prompt: str,
    catalog: FunctionCatalog,
    partial_text: str,
) -> FunctionCallResult:
    """Build a best-effort FunctionCallResult for a prompt that
    couldn't be resolved into a complete, valid function call, so
    every prompt still produces exactly one output record.
    """
    function_def = catalog.get(_extract_partial_name(partial_text) or "")
    if function_def is None:
        return FunctionCallResult(prompt=prompt, name="unknown", parameters={})

    parameters: dict = {}
    for key, spec in function_def.parameters.items():
        raw_value = _extract_partial_value(partial_text, key)
        if raw_value is not None and _value_matches_declared_type(
                raw_value, spec.type):
            parameters[key] = _coerce_value(raw_value, spec.type)
        else:
            parameters[key] = "unknown"

    return FunctionCallResult(
        prompt=prompt, name=function_def.name, parameters=parameters)


def write_results(
        results: list[FunctionCallResult], output_path: Path) -> None:
    """Serialize the results to `output_path` as a JSON array."""
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


def report_fallback_prompt(prompt: str, error: Exception) -> None:
    """Log to stderr that one prompt needed a best-effort fallback."""
    print(
        f"Falling back to a best-effort result for prompt {prompt!r}: "
        f"{error}",
        file=sys.stderr,
    )
