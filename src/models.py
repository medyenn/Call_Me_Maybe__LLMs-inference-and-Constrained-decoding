from pydantic import BaseModel, PrivateAttr, StrictBool, StrictInt, StrictStr
from pydantic import field_validator, model_validator
from pathlib import Path
from typing import Literal, Union

ParamType = Literal["string", "number", "integer", "boolean"]

# Single source of truth (Phase 11) for the type-keyword -> Python type
# mapping. Both the constrained-decoding engine's schema resolver
# (src/decoder.py) and the Output Assembler's final validation pass
# (src/output.py) must agree on this exact mapping, or "my decoder
# allowed X but my validator rejects X" bugs become possible.
PYTHON_TYPE_BY_PARAM_TYPE: dict[ParamType, type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
}

# What a single parameter value is allowed to look like once parsed out
# of the model's generated JSON. Order matters for pydantic's type
# coercion: booleans must be checked before ints (bool is a subclass of
# int in Python), and ints before floats, so a value that is genuinely
# an integer isn't silently widened to a float.
ParamValue = Union[StrictBool, StrictInt, float, StrictStr]


class Config(BaseModel):
    """Resolved, validated paths the pipeline runs on.

    This is the single source of truth for default paths. The CLI layer
    only decides *whether* the user overrode a value; the actual defaults
    live here and nowhere else.
    """

    functions_definition: Path = Path("data/input/functions_definition.json")
    input: Path = Path("data/input/function_calling_tests.json")
    # Subject V.4 is the authoritative output contract and names this
    # file "function_calling_results.json" (IV.3.2's own usage example
    # uses a different, illustrative filename for a *custom* --output
    # value -- see README "Design Decisions" for the full reasoning).
    output: Path = Path("data/output/function_calling_results.json")


class ParameterSpec(BaseModel):
    """The declared type of a single function parameter or return value."""

    type: ParamType


class FunctionDefinition(BaseModel):
    """One callable function as declared in functions_definition.json."""

    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    returns: ParameterSpec

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, value: str) -> str:
        """Reject a function whose name is empty or only whitespace."""
        if not value.strip():
            raise ValueError("function name must not be blank")
        return value


class FunctionCatalog(BaseModel):
    """The full set of functions the model is allowed to call."""

    functions: list[FunctionDefinition]

    _by_name: dict[str, FunctionDefinition] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _validate_and_index(self) -> "FunctionCatalog":
        """Reject an empty/duplicate-name catalog and build the
        name -> FunctionDefinition index used by `get()`.

        Raises:
            ValueError: if the catalog has zero functions, or two
                functions share the same name.
        """
        if not self.functions:
            raise ValueError(
                "functions_definition.json declares zero functions; "
                "at least one function is required."
            )
        names = [fn.name for fn in self.functions]
        if len(names) != len(set(names)):
            duplicates = {n for n in names if names.count(n) > 1}
            raise ValueError(
                f"functions_definition.json has duplicate function "
                f"name(s): {sorted(duplicates)}"
            )
        self._by_name = {fn.name: fn for fn in self.functions}
        return self

    def get(self, name: str) -> FunctionDefinition | None:
        """Look up a function definition by name."""
        return self._by_name.get(name)


class PromptItem(BaseModel):
    """One natural-language request from function_calling_tests.json."""

    prompt: str


class FunctionCallResult(BaseModel):
    """One entry of the final output file, matching V.4 exactly:
    exactly the three keys `prompt`, `name`, `parameters`."""

    prompt: str
    name: str
    parameters: dict[str, ParamValue]
