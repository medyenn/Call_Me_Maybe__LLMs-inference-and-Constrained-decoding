from pydantic import BaseModel, PrivateAttr, StrictBool, StrictInt, StrictStr
from pydantic import field_validator, model_validator
from pathlib import Path
from typing import Literal, Union

ParamType = Literal["string", "number", "integer", "boolean"]

PYTHON_TYPE_BY_PARAM_TYPE: dict[ParamType, type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
}

ParamValue = Union[StrictBool, StrictInt, float, StrictStr]


class Config(BaseModel):
    """Resolved, validated paths and settings the pipeline runs on."""

    functions_definition: Path = Path("data/input/functions_definition.json")
    input: Path = Path("data/input/function_calling_tests.json")
    output: Path = Path("data/output/function_calling_results.json")
    time_budget_seconds: float = 240.0


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
        if not value.strip():
            raise ValueError("function name must not be blank")
        return value


class FunctionCatalog(BaseModel):
    """The full set of functions the model is allowed to call."""

    functions: list[FunctionDefinition]

    _by_name: dict[str, FunctionDefinition] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _validate_and_index(self) -> "FunctionCatalog":
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
        return self._by_name.get(name)


class PromptItem(BaseModel):
    """One natural-language request from function_calling_tests.json."""

    prompt: str


class FunctionCallResult(BaseModel):
    """One output record: exactly `prompt`, `name`, `parameters`."""

    prompt: str
    name: str
    parameters: dict[str, ParamValue]
