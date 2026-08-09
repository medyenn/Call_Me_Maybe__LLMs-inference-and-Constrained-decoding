"""Constrained-decoding engine (Phase 10 components F.1-F.5).

Builds and drives a hand-rolled JSON grammar + function-schema state
machine, used to mask an LLM's logits at every generation step so that
only tokens keeping the output on track for a valid, schema-compliant
`{"name": ..., "parameters": {...}}` object can ever be selected.

No JSON/grammar library is used: `Pos` + `state_tracker` implement the
JSON pushdown-automaton by hand (F.1), `SchemaState` +
`schema_resolver` layer the function-schema constraints on top of it
(F.2), and `build_vocab_trie` + `logit_masker` turn "which token
strings are legal right now" into "which token IDs are legal right
now" without re-simulating the whole vocabulary from scratch at every
step (F.3, Phase 10.3 performance notes).
"""

from dataclasses import dataclass, field, replace
from typing import Any, Protocol, Tuple
from enum import Enum, auto

from .models import FunctionCatalog, FunctionDefinition, ParamType

_LITERALS = {"t": "true", "f": "false", "n": "null"}
_ROOT_KEYS = ("name", "parameters")
# The only characters JSON allows directly after a backslash inside a
# string (RFC 8259 sec. 7). Without this check the grammar happily
# accepts e.g. `\q`, which `json.loads` then rejects downstream --
# turning a silent generation-time bug into a random-looking
# "invalid escape" crash at output-assembly time instead.
_LEGAL_ESCAPE_CHARS = set('"\\/bfnrtu')

# Grammar / State Tracker


class Pos(Enum):
    ROOT = auto()
    ROOT_DONE = auto()
    OBJ_KEY_OR_CLOSE = auto()
    OBJ_KEY = auto()
    OBJ_COLON = auto()
    OBJ_VALUE = auto()
    OBJ_COMMA_OR_CLOSE = auto()
    ARR_VALUE_OR_CLOSE = auto()
    ARR_VALUE = auto()
    ARR_COMMA_OR_CLOSE = auto()
    IN_STRING = auto()
    IN_STRING_ESCAPE = auto()
    IN_LITERAL = auto()
    IN_NUMBER = auto()


StateItem = Pos | tuple[Pos, str]
StateStack = list[StateItem]
TrieNode = dict[str | None, Any]


def _is_value_start(char: str) -> bool:
    """Return True if `char` could legally begin a JSON value (an
    object, array, string, number, or literal)."""
    return char == '"' or char in "{[-" or char.isdigit() or char in _LITERALS


def _descend(stack: StateStack, char: str) -> StateStack | None:
    """Push the state(s) entered by starting a new JSON value with
    `char` (e.g. `{` pushes an "expect key or close" frame).

    Args:
        stack: The current state stack (mutated in place).
        char: The character that starts the new value.

    Returns:
        The mutated stack, or None if `char` cannot start any value.
    """
    if char == "{":
        stack.append(Pos.OBJ_KEY_OR_CLOSE)
    elif char == "[":
        stack.append(Pos.ARR_VALUE_OR_CLOSE)
    elif char == '"':
        stack.append(Pos.IN_STRING)
    elif char == "-":
        stack.append((Pos.IN_NUMBER, "start_needs_digit"))
    elif char == "0":
        stack.append((Pos.IN_NUMBER, "leading_zero"))
    elif char.isdigit():
        stack.append((Pos.IN_NUMBER, "digits"))
    elif char in _LITERALS:
        stack.append((Pos.IN_LITERAL, _LITERALS[char][1:]))
    else:
        return None
    return stack


def state_tracker(state: StateStack, char: str) -> StateStack | None:
    """Advance the pure JSON-grammar state machine by one character.

    This is Phase 10's F.1 "Grammar / State Tracker": it knows nothing
    about function schemas, only about what generic JSON syntax
    permits next (braces balanced, strings properly quoted, numbers
    well-formed, etc).

    Args:
        state: The current stack of `Pos` states (and, for numbers
            and literals, `(Pos, extra)` tuples carrying just enough
            extra context to track partial progress).
        char: The next character being considered.

    Returns:
        The new state stack if `char` is a legal continuation, or
        None if it would break JSON syntax.
    """
    stack = list(state)
    top = stack[-1]

    if isinstance(top, tuple):
        tag, extra = top
        if tag == Pos.IN_LITERAL:
            expected_char, rest = extra[0], extra[1:]
            if char != expected_char:
                return None
            if rest:
                stack[-1] = (Pos.IN_LITERAL, rest)
            else:
                stack.pop()
            return stack
        if tag == Pos.IN_NUMBER:
            if extra == "start_needs_digit":
                if char == "0":
                    stack[-1] = (Pos.IN_NUMBER, "leading_zero")
                    return stack
                if char.isdigit():
                    stack[-1] = (Pos.IN_NUMBER, "digits")
                    return stack
                return None
            if extra == "leading_zero":
                if char == ".":
                    stack[-1] = (Pos.IN_NUMBER, "dot_needs_digit")
                    return stack
                stack.pop()
                return state_tracker(stack, char)
            if extra == "digits":
                if char.isdigit():
                    return stack
                if char == ".":
                    stack[-1] = (Pos.IN_NUMBER, "dot_needs_digit")
                    return stack
                stack.pop()
                return state_tracker(stack, char)
            if extra == "dot_needs_digit":
                if char.isdigit():
                    stack[-1] = (Pos.IN_NUMBER, "frac_digits")
                    return stack
                return None
            if extra == "frac_digits":
                if char.isdigit():
                    return stack
                stack.pop()
                return state_tracker(stack, char)
        return None

    if top == Pos.IN_STRING:
        if char == "\\":
            stack[-1] = Pos.IN_STRING_ESCAPE
            return stack
        if char == '"':
            stack.pop()
            return stack
        return stack

    if top == Pos.IN_STRING_ESCAPE:
        if char not in _LEGAL_ESCAPE_CHARS:
            return None
        stack[-1] = Pos.IN_STRING
        return stack

    if char in " \t\n\r":
        return stack

    if top == Pos.ROOT:
        if _is_value_start(char):
            stack[-1] = Pos.ROOT_DONE
            return _descend(stack, char)
        return None
    if top == Pos.ROOT_DONE:
        return None
    if top == Pos.OBJ_KEY_OR_CLOSE:
        if char == '"':
            stack[-1] = Pos.OBJ_COLON
            stack.append(Pos.IN_STRING)
            return stack
        if char == "}":
            stack.pop()
            return stack
        return None
    if top == Pos.OBJ_KEY:
        if char == '"':
            stack[-1] = Pos.OBJ_COLON
            stack.append(Pos.IN_STRING)
            return stack
        return None
    if top == Pos.OBJ_COLON:
        if char == ":":
            stack[-1] = Pos.OBJ_VALUE
            return stack
        return None
    if top == Pos.OBJ_VALUE:
        if _is_value_start(char):
            stack[-1] = Pos.OBJ_COMMA_OR_CLOSE
            return _descend(stack, char)
        return None
    if top == Pos.OBJ_COMMA_OR_CLOSE:
        if char == ",":
            stack[-1] = Pos.OBJ_KEY
            return stack
        if char == "}":
            stack.pop()
            return stack
        return None
    if top == Pos.ARR_VALUE_OR_CLOSE:
        if _is_value_start(char):
            stack[-1] = Pos.ARR_COMMA_OR_CLOSE
            return _descend(stack, char)
        if char == "]":
            stack.pop()
            return stack
        return None
    if top == Pos.ARR_VALUE:
        if _is_value_start(char):
            stack[-1] = Pos.ARR_COMMA_OR_CLOSE
            return _descend(stack, char)
        return None
    if top == Pos.ARR_COMMA_OR_CLOSE:
        if char == ",":
            stack[-1] = Pos.ARR_VALUE
            return stack
        if char == "]":
            stack.pop()
            return stack
        return None
    return None


# Schema Constraint Resolver


@dataclass
class SchemaState:
    """Everything the Schema Constraint Resolver (F.2) needs to track
    on top of the raw JSON grammar state: which function has been
    selected (if any), which of its parameters are still required,
    the declared type of whichever parameter value is in progress, and
    a small text buffer used to recognise function/parameter names
    (and the two fixed root keys) character by character.

    Attributes:
        catalog: The function catalog this generation is constrained
            against.
        selected: The `FunctionDefinition` chosen once the "name"
            field has been fully written, else None.
        remaining_params: Parameter names of `selected` not yet
            written into the "parameters" object.
        current_param_type: Declared type of the parameter value
            currently being generated, else None.
        params_depth: Stack depth (`len(f1_state)`) at which the
            "parameters" object's contents live, once its `{` has
            been seen.
        collecting: What kind of string is currently being
            accumulated into `buffer` -- "root_key", "name", "key",
            or None.
        buffer: The characters collected so far for `collecting`.
        root_key_index: How many of the two required root keys
            (`_ROOT_KEYS`) have been fully written so far.
        expect_params_object: True right after the `"parameters":`
            key/colon, before its opening `{` has been seen.
    """

    catalog: FunctionCatalog
    selected: FunctionDefinition | None = None
    remaining_params: set[str] = field(default_factory=set)
    current_param_type: ParamType | None = None
    params_depth: int | None = None
    collecting: str | None = None
    buffer: str = ""
    root_key_index: int = 0
    expect_params_object: bool = False


def _value_start_allowed(param_type: str, char: str) -> bool:
    """Return True if `char` could legally begin a value of
    `param_type` ("string", "number", "integer", or "boolean")."""
    if param_type == "string":
        return char == '"'
    if param_type in ("number", "integer"):
        return char.isdigit() or char == "-"
    if param_type == "boolean":
        return char in ("t", "f")
    return False


def schema_resolver(
    f1_state: StateStack,
    f2: SchemaState,
    char: str,
) -> Tuple[StateStack, SchemaState] | None:
    """Advance both the JSON grammar state and the schema state by one
    character, enforcing structural JSON validity AND schema
    compliance together (Phase 5.2 / 10-F.1+F.2 combined).

    On top of `state_tracker`'s pure-syntax rules, this additionally
    restricts: the two root keys to exactly `name` then `parameters`
    (in order, no repeats); `name`'s string value to a prefix of a
    real function name; parameter keys to a prefix of one of the
    selected function's still-remaining parameter names; parameter
    values to characters legal for that parameter's declared type; and
    the "parameters"/root object from closing before every required
    key has been written.

    Args:
        f1_state: Current JSON-grammar state stack (as used by
            `state_tracker`).
        f2: Current `SchemaState`.
        char: The next character being considered.

    Returns:
        The `(new_f1_state, new_f2)` pair if `char` is legal under
        both JSON syntax and the function schema, or None otherwise.
    """
    prev_top = f1_state[-1]
    in_params = f2.params_depth is not None and (
        len(f1_state) == f2.params_depth)
    at_root = len(f1_state) == 2

    if prev_top == Pos.ROOT and char != "{":
        return None
    if (
        at_root
        and char == '"'
        and prev_top in (
            Pos.OBJ_KEY_OR_CLOSE, Pos.OBJ_KEY, Pos.OBJ_COMMA_OR_CLOSE)
        and f2.root_key_index >= len(_ROOT_KEYS)
    ):
        return None
    if (
        f2.collecting == "root_key"
        and prev_top == Pos.IN_STRING
        and char != '"'
    ):
        expected = _ROOT_KEYS[f2.root_key_index]
        if not expected.startswith(f2.buffer + char):
            return None
    if char == "}" and at_root and f2.root_key_index < len(_ROOT_KEYS):
        return None
    if (
        f2.expect_params_object
        and prev_top == Pos.OBJ_VALUE
        and char not in " \t\n\r"
        and char != "{"
    ):
        return None
    if (
        prev_top == Pos.OBJ_VALUE
        and at_root
        and f2.selected is None
        and f2.root_key_index == 1
        and char not in " \t\n\r"
        and char != '"'
    ):
        return None
    if (
        prev_top == Pos.IN_STRING
        and char == "\\"
        and f2.collecting in ("root_key", "name", "key")
    ):
        # Function/parameter/root-key identifiers never contain
        # characters that need escaping. Without this guard, entering
        # IN_STRING_ESCAPE here would leave `f2.buffer` one character
        # short (the escaped char is intentionally not buffered) and
        # the identifier match below would silently desync from what
        # was actually generated.
        return None
    if (prev_top == Pos.IN_STRING and char != '"'
            and f2.collecting in ("name", "key")):
        candidate = f2.buffer + char
        pool = (
            [fn.name for fn in f2.catalog.functions]
            if f2.collecting == "name"
            else list(f2.remaining_params)
        )
        if not any(name.startswith(candidate) for name in pool):
            return None
    if (
        prev_top == Pos.OBJ_VALUE
        and in_params
        and f2.current_param_type is not None
        and char not in " \t\n\r"
        and not _value_start_allowed(f2.current_param_type, char)
    ):
        return None
    if (
        char == '"'
        and in_params
        and (prev_top in
                (Pos.OBJ_KEY_OR_CLOSE, Pos.OBJ_KEY, Pos.OBJ_COMMA_OR_CLOSE))
        and not f2.remaining_params
    ):
        return None

    new_f1 = state_tracker(f1_state, char)
    if new_f1 is None:
        return None

    if (
        f2.params_depth is not None
        and len(new_f1) < f2.params_depth
        and f2.remaining_params
    ):
        return None
    if (
        char == ","
        and f2.params_depth is not None
        and len(new_f1) == f2.params_depth
        and not f2.remaining_params
    ):
        return None
    if (char == "," and len(new_f1) == 2
            and f2.root_key_index >= len(_ROOT_KEYS)):
        return None

    f2 = replace(f2, remaining_params=set(f2.remaining_params))

    if (
        at_root
        and char == '"'
        and prev_top in (
            Pos.OBJ_KEY_OR_CLOSE, Pos.OBJ_KEY, Pos.OBJ_COMMA_OR_CLOSE)
    ):
        f2.collecting, f2.buffer = "root_key", ""
    if (prev_top == Pos.IN_STRING and char == '"'
            and f2.collecting == "root_key"):
        if f2.buffer != _ROOT_KEYS[f2.root_key_index]:
            return None
        f2.root_key_index += 1
        if _ROOT_KEYS[f2.root_key_index - 1] == "parameters":
            f2.expect_params_object = True
        f2.collecting, f2.buffer = None, ""
    if prev_top == Pos.OBJ_VALUE and char == '"' and f2.selected is None:
        f2.collecting, f2.buffer = "name", ""
    if (prev_top in (Pos.OBJ_KEY_OR_CLOSE, Pos.OBJ_KEY)
            and char == '"' and in_params):
        f2.collecting, f2.buffer = "key", ""
    if prev_top == Pos.IN_STRING and char != '"' and f2.collecting:
        f2.buffer += char
    if prev_top == Pos.IN_STRING and char == '"':
        if f2.collecting == "name":
            fn = f2.catalog.get(f2.buffer)
            if fn is None:
                return None
            f2.selected = fn
            f2.remaining_params = set(fn.parameters.keys())
            f2.collecting, f2.buffer = None, ""
        elif f2.collecting == "key":
            key = f2.buffer
            if key not in f2.remaining_params:
                return None
            f2.current_param_type = (
                f2.selected.parameters[key].type)  # type: ignore[union-attr]
            f2.remaining_params.discard(key)
            f2.collecting, f2.buffer = None, ""
    if char == "{" and f2.selected is not None and f2.params_depth is None:
        f2.params_depth = len(new_f1)
        f2.expect_params_object = False
    if prev_top == Pos.OBJ_COMMA_OR_CLOSE and char == "," and in_params:
        f2.current_param_type = None

    return new_f1, f2


# Logit Masker (trie-based)


def build_vocab_trie(id_to_token: dict[int, str]) -> TrieNode:
    """Build a char-trie over the vocabulary ONCE, outside the
    generation loop. node[ch] -> child node; node[None] -> token ids
    whose full string ends exactly at this node.

    Args:
        id_to_token: The full token-ID -> token-string mapping.

    Returns:
        The trie's root node.
    """
    root: TrieNode = {}
    for token_id, token_str in id_to_token.items():
        node = root
        for ch in token_str:
            node = node.setdefault(ch, {})
        node.setdefault(None, []).append(token_id)
    return root


def _state_key(
    f1_state: StateStack,
    f2: SchemaState,
) -> tuple[Any, ...]:
    """A hashable snapshot of (f1, f2) — cache key so an identical
    (state, char) transition is never computed twice."""
    return (
        tuple(f1_state),
        id(f2.catalog),
        f2.selected.name if f2.selected else None,
        frozenset(f2.remaining_params),
        f2.current_param_type,
        f2.params_depth,
        f2.collecting,
        f2.buffer,
        f2.root_key_index,
        f2.expect_params_object,
    )


def logit_masker(
    logits: list[float],
    trie: TrieNode,
    f1_state: StateStack,
    f2: SchemaState,
    transition_cache: dict[Any, Any] | None = None,
) -> list[float]:
    """F.3 — set every illegal token's logit to -inf, by walking the
    vocab trie instead of simulating each token independently.

    Args:
        logits: The raw logits for the next token, one per vocab
            entry (unmodified; a new list is returned).
        trie: The vocabulary trie from `build_vocab_trie`.
        f1_state: Current JSON-grammar state stack.
        f2: Current `SchemaState`.
        transition_cache: If passed in and reused across steps, an
            identical (state, char) transition is never recomputed
            across the whole generation run for one prompt.

    Returns:
        A new logits list where every token ID that would be an
        illegal continuation has been set to negative infinity.
    """
    if transition_cache is None:
        transition_cache = {}

    legal_ids: list[int] = []

    def walk(
        node: TrieNode,
        state1: StateStack,
        state2: SchemaState,
    ) -> None:
        terminal = node.get(None)
        if terminal:
            legal_ids.extend(terminal)
        for ch, child in node.items():
            if ch is None:
                continue
            key = (_state_key(state1, state2), ch)
            if key in transition_cache:
                result = transition_cache[key]
            else:
                result = schema_resolver(state1, state2, ch)
                transition_cache[key] = result
            if result is not None:
                walk(child, *result)

    walk(trie, f1_state, f2)

    masked = list(logits)
    legal_set = set(legal_ids)
    for token_id in range(len(masked)):
        if token_id not in legal_set:
            masked[token_id] = float("-inf")
    return masked


# Token Selector


def token_selector(masked_logits: list[float]) -> int:
    """Greedily pick the next token ID from already-masked logits
    (Phase 6.4: deterministic selection among legal tokens only, since
    correctness -- not creative variety -- is the goal here).

    Args:
        masked_logits: Logits with every illegal token already set to
            negative infinity (see `logit_masker`).

    Returns:
        The token ID with the highest surviving logit.

    Raises:
        RuntimeError: if every single token is illegal. This
            should never happen if the grammar/schema resolver is
            correct; it indicates an internal bug rather than a
            normal per-prompt failure.
    """
    best_id = max(range(len(masked_logits)), key=lambda i: masked_logits[i])
    if masked_logits[best_id] == float("-inf"):
        raise RuntimeError(
            "No legal token survived masking (internal grammar bug)."
        )
    return best_id


# Generation Loop Driver


class _GenerationSDK(Protocol):
    """The subset of the llm_sdk SDK's public interface that the
    generation loop depends on -- documented here as a Protocol-style
    contract rather than imported directly, so this module never
    depends on llm_sdk's internal types (only the given public
    interface: encode, get_logits_from_input_ids)."""

    def encode(self, text: str) -> Any:
        ...

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        ...


def generate_function_call(
    sdk: _GenerationSDK,
    vocab: Any,
    catalog: FunctionCatalog,
    prompt: str,
    max_new_tokens: int = 150,
) -> str:
    """F.5 — build the trie ONCE, then loop: get logits, mask, select,
    advance state, feed the chosen token back to the model.

    Args:
        sdk: An llm_sdk model instance (only its public `encode` and
            `get_logits_from_input_ids` methods are used).
        vocab: A `loader.VocabularyManager`-shaped object exposing
            `id_to_token`.
        catalog: The function catalog to constrain generation against.
        prompt: The natural-language prompt (already including the
            function menu) to feed the model.
        max_new_tokens: Safety cap on generated tokens, so a
            hypothetical grammar bug can never turn into an infinite
            loop (Phase 10-F.5's "never crashes / never hangs").

    Returns:
        The raw generated text: valid JSON, by construction, of the
        shape `{"name": ..., "parameters": {...}}`.

    Raises:
        RuntimeError: if no legal token exists at some step, or
            the token budget is exhausted before completion.
    """
    trie = build_vocab_trie(vocab.id_to_token)
    transition_cache: dict[Any, Any] = {}

    input_ids = sdk.encode(prompt)
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]

    f1: StateStack = [Pos.ROOT]
    f2 = SchemaState(catalog=catalog)
    generated_text = ""

    for _ in range(max_new_tokens):
        if f1 == [Pos.ROOT_DONE]:
            return generated_text

        logits = sdk.get_logits_from_input_ids(input_ids)
        masked = logit_masker(logits, trie, f1, f2, transition_cache)
        token_id = token_selector(masked)
        token_str = vocab.id_to_token[token_id]

        for char in token_str:
            result = schema_resolver(f1, f2, char)
            if result is None:
                raise RuntimeError(
                    f"Masker selected illegal token {token_str!r} "
                    "(internal grammar bug)."
                )
            f1, f2 = result

        generated_text += token_str
        input_ids = input_ids + [token_id]

    raise RuntimeError(
        f"Hit max_new_tokens={max_new_tokens} without completing a "
        "well-formed function call."
    )
