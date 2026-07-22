from pydantic import BaseModel, StrictBool, StrictStr, model_validator
from pathlib import Path
from typing import Literal, Union


ParamType = Literal["string", "number", "integer", "boolean"]
ParamValue = Union[StrictBool, float, StrictStr]


class Config(BaseModel):
    """Resolved, validated paths the pipeline runs on.

    This is the single source of truth for default paths. The CLI layer
    only decides *whether* the user overrode a value; the actual defaults
    live here and nowhere else.
    """

    functions_definition: Path = Path("data/input/functions_definition.json")
    input: Path = Path("data/input/function_calling_tests.json")
    output: Path = Path("data/output/function_calls.json")


class ParameterSpec(BaseModel):
    """The declared type of a single function parameter or return value."""

    type: ParamType


class FunctionDefinition(BaseModel):
    """One callable function as declared in functions_definition.json."""

    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    returns: ParameterSpec


class FunctionCatalog(BaseModel):
    """The full set of functions the model is allowed to call."""

    functions: list[FunctionDefinition]

    @model_validator(mode="after")
    def _build_lookup(self) -> "FunctionCatalog":
        """Build a name -> FunctionDefinition index for O(1) lookup."""
        self._by_name = {f.name: f for f in self.functions}
        return self

    def get(self, name: str) -> FunctionDefinition | None:
        """Look up a function definition by name."""
        return self._by_name.get(name)


class PromptItem(BaseModel):
    """One natural-language request from function_calling_tests.json."""

    prompt: str


class FunctionCallResult(BaseModel):
    """One entry of the final output file, matching V.4 exactly."""

    prompt: str
    name: str
    parameters: dict[str, ParamValue]


class GenerationConfig(BaseModel):
    ...
