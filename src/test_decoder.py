from dataclasses import dataclass, field, replace
from enum import Enum, auto

from .models import FunctionCatalog, FunctionDefinition, ParamType

_LITERALS = {"t": "true", "f": "false", "n": "null"}
_ROOT_KEYS = ("name", "parameters")


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


def initial_state() -> list:
    return [Pos.ROOT]


def _is_value_start(char: str) -> bool:
    return char == '"' or char in "{[-" or char.isdigit() or char in _LITERALS


def _descend(stack: list, char: str):
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


def state_tracker(state: list, char: str):
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
    catalog: FunctionCatalog
    selected: FunctionDefinition | None = None
    remaining_params: set = field(default_factory=set)
    current_param_type: ParamType | None = None
    params_depth: int | None = None
    collecting: str | None = None
    buffer: str = ""
    root_key_index: int = 0
    expect_params_object: bool = False


def initial_schema_state(catalog: FunctionCatalog) -> SchemaState:
    return SchemaState(catalog=catalog)


def _value_start_allowed(param_type: str, char: str) -> bool:
    if param_type == "string":
        return char == '"'
    if param_type in ("number", "integer"):
        return char.isdigit() or char == "-"
    if param_type == "boolean":
        return char in ("t", "f")
    return False


def schema_resolver(f1_state: list, f2: SchemaState, char: str):
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
        and char not in ('"', "\\")
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
    if prev_top == Pos.IN_STRING and char != '"' and f2.collecting in ("name", "key"):
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
        and prev_top in (Pos.OBJ_KEY_OR_CLOSE, Pos.OBJ_KEY, Pos.OBJ_COMMA_OR_CLOSE)
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
    if char == "," and len(new_f1) == 2 and f2.root_key_index >= len(_ROOT_KEYS):
        return None

    f2 = replace(f2, remaining_params=set(f2.remaining_params))

    if (
        at_root
        and char == '"'
        and prev_top in (
            Pos.OBJ_KEY_OR_CLOSE, Pos.OBJ_KEY, Pos.OBJ_COMMA_OR_CLOSE)
    ):
        f2.collecting, f2.buffer = "root_key", ""
    if prev_top == Pos.IN_STRING and char == '"' and f2.collecting == "root_key":
        if f2.buffer != _ROOT_KEYS[f2.root_key_index]:
            return None
        f2.root_key_index += 1
        if _ROOT_KEYS[f2.root_key_index - 1] == "parameters":
            f2.expect_params_object = True
        f2.collecting, f2.buffer = None, ""
    if prev_top == Pos.OBJ_VALUE and char == '"' and f2.selected is None:
        f2.collecting, f2.buffer = "name", ""
    if prev_top in (Pos.OBJ_KEY_OR_CLOSE, Pos.OBJ_KEY) and char == '"' and in_params:
        f2.collecting, f2.buffer = "key", ""
    if prev_top == Pos.IN_STRING and char not in ('"', "\\") and f2.collecting:
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
            f2.current_param_type = f2.selected.parameters[key].type
            f2.remaining_params.discard(key)
            f2.collecting, f2.buffer = None, ""
    if char == "{" and f2.selected is not None and f2.params_depth is None:
        f2.params_depth = len(new_f1)
        f2.expect_params_object = False
    if prev_top == Pos.OBJ_COMMA_OR_CLOSE and char == "," and in_params:
        f2.current_param_type = None

    return new_f1, f2


# Logit Masker (trie-based)


def build_vocab_trie(id_to_token: dict) -> dict:
    """Build a char-trie over the vocabulary ONCE, outside the
    generation loop. node[ch] -> child node; node[None] -> token ids
    whose full string ends exactly at this node."""
    root: dict = {}
    for token_id, token_str in id_to_token.items():
        node = root
        for ch in token_str:
            node = node.setdefault(ch, {})
        node.setdefault(None, []).append(token_id)
    return root


def _state_key(f1_state: list, f2: SchemaState):
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
    logits: list,
    trie: dict,
    f1_state: list,
    f2: SchemaState,
    transition_cache: dict | None = None,
) -> list:
    """F.3 — set every illegal token's logit to -inf, by walking the
    vocab trie instead of simulating each token independently.
    `transition_cache`, if passed in and reused across steps, means an
    identical (state, char) pair is never recomputed across the whole run."""
    if transition_cache is None:
        transition_cache = {}

    legal_ids: list = []

    def walk(node, state1, state2):
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


def token_selector(masked_logits: list) -> int:
    best_id = max(range(len(masked_logits)), key=lambda i: masked_logits[i])
    if masked_logits[best_id] == float("-inf"):
        raise RuntimeError("No legal token survived masking — grammar bug.")
    return best_id


# Generation Loop Driver


def generate_function_call(
    sdk, vocab, catalog: FunctionCatalog, prompt: str, max_new_tokens: int = 150
) -> str:
    """F.5 — build the trie ONCE, then loop: get logits, mask, select,
    advance state, feed the chosen token back to the model."""
    trie = build_vocab_trie(vocab.id_to_token)
    transition_cache: dict = {}

    input_ids = sdk.encode(prompt)
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]

    f1, f2 = initial_state(), initial_schema_state(catalog)
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
                    f"masker selected illegal token {token_str!r} — bug."
                )
            f1, f2 = result

        generated_text += token_str
        input_ids = input_ids + [token_id]

    raise RuntimeError(f"hit max_new_tokens={max_new_tokens} without finishing")
