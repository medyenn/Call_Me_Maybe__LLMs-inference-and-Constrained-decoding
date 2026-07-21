from abc import BaseModel


class ParameterSpec(BaseModel):
    ...


class FunctionDefinition(BaseModel):
    ...


class FunctionCatalog(BaseModel):
    ...


class PromptRequest(BaseModel):
    ...


class FunctionCallResult(BaseModel):
    ...


class GenerationConfig(BaseModel):
    ...
