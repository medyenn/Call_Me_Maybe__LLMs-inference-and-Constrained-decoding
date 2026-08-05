from dataclasses import dataclass, field, replace
from enum import Enum, auto

from .models import FunctionCatalog, FunctionDefinition, ParamType

_LITERALS = {'t': 'true', 'f': 'false', 'n': 'null'}
_ROOT_KEYS = ("name", "parameters")


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
    """The F1 state before any characters have been generated."""
    return [Pos.ROOT]


def _is_value_start(char: str) -> bool:
    return char == '"' or char in '{[-' or char.isdigit() or char in _LITERALS


def _descend(stack: list, char: str):
    """`char` legally begins a value; the caller already updated the top
    of the stack to what it becomes once this value finishes. Push what
    it means to be 'inside' this new value."""
    if char == '{':
        stack.append(Pos.OBJ_KEY_OR_CLOSE)
    elif char == '[':
        stack.append(Pos.ARR_VALUE_OR_CLOSE)
    elif char == '"':
        stack.append(Pos.IN_STRING)
    elif char == '-':
        stack.append((Pos.IN_NUMBER, "start_needs_digit"))
    elif char == '0':
        stack.append((Pos.IN_NUMBER, "leading_zero"))  # "01" is invalid JSON
    elif char.isdigit():
        stack.append((Pos.IN_NUMBER, "digits"))
    elif char in _LITERALS:
        stack.append((Pos.IN_LITERAL, _LITERALS[char][1:]))
    else:
        return None
    return stack


def state_tracker(state: list, char: str):
    """F.1 — consume one character. Returns the new state, or None if
    `char` is not valid JSON syntax here."""
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
                if char == '0':
                    stack[-1] = (Pos.IN_NUMBER, "leading_zero")
                    return stack
                if char.isdigit():
                    stack[-1] = (Pos.IN_NUMBER, "digits")
                    return stack
                return None
            if extra == "leading_zero":
                if char == '.':
                    stack[-1] = (Pos.IN_NUMBER, "dot_needs_digit")
                    return stack
                stack.pop()
                return state_tracker(stack, char)
            if extra == "digits":
                if char.isdigit():
                    return stack
                if char == '.':
                    stack[-1] = (Pos.IN_NUMBER, "dot_needs_digit")
                    return stack
                stack.pop()
                return state_tracker(stack, char)
            if extra == "dot_needs_digit":
                if char.isdigit():
                    stack[-1] = (Pos.IN_NUMBER, "frac_digits")
                    return stack
                return None  # e.g. "23." with no digit after '.' -> invalid
            if extra == "frac_digits":
                if char.isdigit():
                    return stack
                stack.pop()
                return state_tracker(stack, char)
        return None

    if top == Pos.IN_STRING:
        if char == '\\':
            stack[-1] = Pos.IN_STRING_ESCAPE
            return stack
        if char == '"':
            stack.pop()
            return stack
        return stack

    if top == Pos.IN_STRING_ESCAPE:
        stack[-1] = Pos.IN_STRING
        return stack

    if char in ' \t\n\r':
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
        if char == '}':
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
        if char == ':':
            stack[-1] = Pos.OBJ_VALUE
            return stack
        return None

    if top == Pos.OBJ_VALUE:
        if _is_value_start(char):
            stack[-1] = Pos.OBJ_COMMA_OR_CLOSE
            return _descend(stack, char)
        return None

    if top == Pos.OBJ_COMMA_OR_CLOSE:
        if char == ',':
            stack[-1] = Pos.OBJ_KEY
            return stack
        if char == '}':
            stack.pop()
            return stack
        return None

    if top == Pos.ARR_VALUE_OR_CLOSE:
        if _is_value_start(char):
            stack[-1] = Pos.ARR_COMMA_OR_CLOSE
            return _descend(stack, char)
        if char == ']':
            stack.pop()
            return stack
        return None

    if top == Pos.ARR_VALUE:
        if _is_value_start(char):
            stack[-1] = Pos.ARR_COMMA_OR_CLOSE
            return _descend(stack, char)
        return None

    if top == Pos.ARR_COMMA_OR_CLOSE:
        if char == ',':
            stack[-1] = Pos.ARR_VALUE
            return stack
        if char == ']':
            stack.pop()
            return stack
        return None

    return None


@dataclass
class SchemaState:
    catalog: FunctionCatalog
    selected: FunctionDefinition | None = None
    remaining_params: set = field(default_factory=set)
    current_param_type: ParamType | None = None
    params_depth: int | None = None     # F1 stack depth of "parameters"'s {}
    collecting: str | None = None       # "root_key" | "name" | "key" | None
    buffer: str = ""
    root_key_index: int = 0             # how many of _ROOT_KEYS are done
    expect_params_object: bool = False  # True right after "parameters" closes


def initial_schema_state(catalog: FunctionCatalog) -> SchemaState:
    """The F2 state before any characters have been generated."""
    return SchemaState(catalog=catalog)


def _value_start_allowed(param_type: str, char: str) -> bool:
    if param_type == "string":
        return char == '"'
    if param_type in ("number", "integer"):
        return char.isdigit() or char == '-'
    if param_type == "boolean":
        return char in ('t', 'f')
    return False


def schema_resolver(f1_state: list, f2: SchemaState, char: str):
    """F.2 — advance F1 and F2 together by one character.
    Returns (new_f1_state, new_f2_state), or None if illegal."""
    prev_top = f1_state[-1]
    in_params = f2.params_depth is not None and len(f1_state) == f2.params_depth
    at_root = len(f1_state) == 2  # depth of the top-level object's keys

    # --- pre-checks: can reject what F1 alone would allow ---

    if prev_top == Pos.ROOT and char != '{':
        return None  # top-level value must be an object, not bare str/num/etc

    if (at_root and char == '"'
            and prev_top in (Pos.OBJ_KEY_OR_CLOSE, Pos.OBJ_KEY, Pos.OBJ_COMMA_OR_CLOSE)
            and f2.root_key_index >= len(_ROOT_KEYS)):
        return None  # no more root keys allowed

    if (f2.collecting == "root_key" and prev_top == Pos.IN_STRING
            and char not in ('"', '\\')):
        expected = _ROOT_KEYS[f2.root_key_index]
        if not expected.startswith(f2.buffer + char):
            return None  # force literal "name" then "parameters"

    if char == '}' and at_root and f2.root_key_index < len(_ROOT_KEYS):
        return None  # can't close root object before both keys exist

    if (f2.expect_params_object and prev_top == Pos.OBJ_VALUE
            and char not in ' \t\n\r' and char != '{'):
        return None  # "parameters"'s value must be an object

    if (prev_top == Pos.OBJ_VALUE and at_root and f2.selected is None
            and f2.root_key_index == 1 and char not in ' \t\n\r' and char != '"'):
        return None  # "name"'s value must be a string

    if (prev_top == Pos.IN_STRING and char not in ('"', '\\')
            and f2.collecting in ("name", "key")):
        candidate = f2.buffer + char
        pool = ([fn.name for fn in f2.catalog.functions] if f2.collecting == "name"
                else list(f2.remaining_params))
        if not any(name.startswith(candidate) for name in pool):
            return None  # narrow function-name / param-key prefix

    if (prev_top == Pos.OBJ_VALUE and in_params and f2.current_param_type is not None
            and char not in ' \t\n\r' and not _value_start_allowed(f2.current_param_type, char)):
        return None  # param value must start with a char matching its type

    if (char == '"' and in_params
            and prev_top in (Pos.OBJ_KEY_OR_CLOSE, Pos.OBJ_KEY, Pos.OBJ_COMMA_OR_CLOSE)
            and not f2.remaining_params):
        return None  # no schema params left to open a key for

    new_f1 = state_tracker(f1_state, char)
    if new_f1 is None:
        return None

    if f2.params_depth is not None and len(new_f1) < f2.params_depth and f2.remaining_params:
        return None  # can't close "parameters" while required params missing

    if (char == ',' and f2.params_depth is not None
            and len(new_f1) == f2.params_depth and not f2.remaining_params):
        return None  # trailing comma dead-end inside "parameters"

    if char == ',' and len(new_f1) == 2 and f2.root_key_index >= len(_ROOT_KEYS):
        return None  # trailing comma dead-end at root level

    # --- updates: react to what F1 just accepted ---
    f2 = replace(f2, remaining_params=set(f2.remaining_params))

    if (at_root and char == '"'
            and prev_top in (Pos.OBJ_KEY_OR_CLOSE, Pos.OBJ_KEY, Pos.OBJ_COMMA_OR_CLOSE)):
        f2.collecting, f2.buffer = "root_key", ""

    if prev_top == Pos.IN_STRING and char == '"' and f2.collecting == "root_key":
        if f2.buffer != _ROOT_KEYS[f2.root_key_index]:
            return None  # must be the FULL literal, not a valid-so-far prefix
        f2.root_key_index += 1
        if _ROOT_KEYS[f2.root_key_index - 1] == "parameters":
            f2.expect_params_object = True
        f2.collecting, f2.buffer = None, ""

    if prev_top == Pos.OBJ_VALUE and char == '"' and f2.selected is None:
        f2.collecting, f2.buffer = "name", ""

    if prev_top in (Pos.OBJ_KEY_OR_CLOSE, Pos.OBJ_KEY) and char == '"' and in_params:
        f2.collecting, f2.buffer = "key", ""

    if prev_top == Pos.IN_STRING and char not in ('"', '\\') and f2.collecting:
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
                return None  # must be a COMPLETE, still-remaining param name
            f2.current_param_type = f2.selected.parameters[key].type
            f2.remaining_params.discard(key)
            f2.collecting, f2.buffer = None, ""

    if char == '{' and f2.selected is not None and f2.params_depth is None:
        f2.params_depth = len(new_f1)
        f2.expect_params_object = False

    if prev_top == Pos.OBJ_COMMA_OR_CLOSE and char == ',' and in_params:
        f2.current_param_type = None

    return new_f1, f2


# =====================================================================
# F.3 — Logit Masker
# =====================================================================

def _is_token_legal(f1_state: list, f2: SchemaState, token_str: str) -> bool:
    for char in token_str:
        result = schema_resolver(f1_state, f2, char)
        if result is None:
            return False
        f1_state, f2 = result
    return True


def logit_masker(logits: list, id_to_token: dict, f1_state: list, f2: SchemaState) -> list:
    """F.3 — set every illegal token's logit to -inf."""
    masked = list(logits)
    for token_id, token_str in id_to_token.items():
        if not _is_token_legal(f1_state, f2, token_str):
            masked[token_id] = float("-inf")
    return masked


# =====================================================================
# F.4 — Token Selector
# =====================================================================

def token_selector(masked_logits: list) -> int:
    """F.4 — pick the highest-scoring surviving token id."""
    best_id = max(range(len(masked_logits)), key=lambda i: masked_logits[i])
    if masked_logits[best_id] == float("-inf"):
        raise RuntimeError("No legal token survived masking — grammar bug.")
    return best_id


# =====================================================================
# F.5 — Generation Loop Driver
# =====================================================================

def generate_function_call(sdk, vocab, catalog: FunctionCatalog, prompt: str, max_new_tokens: int = 500) -> str:
    """F.5 — run constrained generation, return the raw JSON text.
    `sdk` needs encode(text)->ids and get_logits_from_input_ids(ids)->logits.
    `vocab` needs `.id_to_token: dict[int, str]`."""
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
        masked = logit_masker(logits, vocab.id_to_token, f1, f2)
        token_id = token_selector(masked)
        token_str = vocab.id_to_token[token_id]

        for char in token_str:
            result = schema_resolver(f1, f2, char)
            if result is None:
                raise RuntimeError(f"masker selected illegal token {token_str!r} — bug.")
            f1, f2 = result

        generated_text += token_str
        input_ids = input_ids + [token_id]

    raise RuntimeError(f"hit max_new_tokens={max_new_tokens} without finishing")
