class CallMeMaybeError(Exception):
    """Base class for every error raised by this project's own code.

    The CLI entrypoint catches this single type (and only this type)
    at the top level, so every failure path in the pipeline surfaces a
    clear, human-readable message instead of a raw traceback, while
    still letting truly unexpected bugs propagate as-is.
    """


class InputFileError(CallMeMaybeError):
    """Raised when a required input file is missing, unreadable, or
    contains malformed/invalid JSON.

    Covers both `functions_definition.json` and
    `function_calling_tests.json` (subject Ch. V.2): missing file,
    invalid JSON syntax, and I/O errors (permissions, etc.).
    """


class SchemaValidationError(CallMeMaybeError):
    """Raised when parsed JSON does not match the expected data shape.

    Covers: a function definition missing a required key, a prompt
    entry with a non-string "prompt" value, an empty function
    catalog, or a generated function call that (despite constrained
    decoding) fails the final defensive validation pass against the
    real `FunctionCatalog`.
    """


class GenerationError(CallMeMaybeError):
    """Raised when the constrained-decoding engine cannot produce a
    valid function call.

    Covers: no legal token surviving the logit mask at some step (a
    grammar/schema bug), and hitting the maximum token budget without
    completing a well-formed object.
    """
