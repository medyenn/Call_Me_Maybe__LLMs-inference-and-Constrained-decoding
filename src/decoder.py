from enum import Enum, auto
from dataclasses import dataclass, field
from .models import FunctionCatalog, FunctionDefinition, ParamType


_LITERALS = {'t': 'true', 'f': 'false', 'n': 'null'}


class SchemaPhase(Enum):
    """Where we are in the FIXED shape: {"name": ..., "parameters": {...}}."""
    NAME_VALUE = auto()     # currently typing characters of the function name
    PARAM_KEY = auto()      # currently typing a parameter key's characters
    PARAM_VALUE = auto()    # currently typing a parameter value
    BETWEEN = auto()        # not currently inside name/key/value (e.g. right
    # after a key closes, waiting for ':' — F1 already
    # enforces the punctuation, F2 has nothing to add here)
    DONE = auto()  # function fully resolved, nothing left to constrain


@dataclass
class SchemaState:
    """F2's state, tracked alongside (not instead of) F1's stack."""
    catalog: FunctionCatalog
    phase: SchemaPhase = SchemaPhase.NAME_VALUE

    name_prefix: str = ""  # chars typed so far of the function name
    selected: FunctionDefinition | None = None

    remaining_params: set[str] = field(default_factory=set)
    key_prefix: str = ""  # chars typed so far of current param key
    current_param_type: ParamType | None = None
    value_prefix: str = ""  # chars typed so far of current param VALUE


class Pos(Enum):
    """One flat list of every position the tracker can be in.
    The whole tracker state is just a stack (list) of these."""
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
    """The tracker state before any characters have been generated."""
    return [Pos.ROOT]


def initial_schema_state(catalog: FunctionCatalog) -> SchemaState:
    """The F2 state before any characters have been generated."""
    return SchemaState(catalog=catalog)


def _is_value_start(char: str) -> bool:
    """Whether `char` can legally begin a JSON value."""
    return char == '"' or char in '{[-' or char.isdigit() or char in _LITERALS


def _descend(stack: list, char: str):
    """`char` legally begins a value; the caller has ALREADY updated the
    top of the stack to what it becomes once this value finishes.
    Push what it means to now be 'inside' this new value."""
    if char == '{':
        stack.append(Pos.OBJ_KEY_OR_CLOSE)
    elif char == '[':
        stack.append(Pos.ARR_VALUE_OR_CLOSE)
    elif char == '"':
        stack.append(Pos.IN_STRING)
    elif char == '-':
        stack.append((Pos.IN_NUMBER, "start_needs_digit"))
    elif char.isdigit():
        stack.append((Pos.IN_NUMBER, "digits"))
    elif char in _LITERALS:
        stack.append((Pos.IN_LITERAL, _LITERALS[char][1:]))
    else:
        return None
    return stack


def step(state: list, char: str):
    """Consume one character. Returns the new state if legal, else None."""
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
                if char.isdigit():
                    stack[-1] = (Pos.IN_NUMBER, "digits")
                    return stack
                return None

            if extra == "digits":
                if char.isdigit():
                    return stack
                if char == '.':
                    stack[-1] = (Pos.IN_NUMBER, "dot_needs_digit")
                    return stack
                stack.pop()
                return step(stack, char)

            if extra == "dot_needs_digit":
                if char.isdigit():
                    stack[-1] = (Pos.IN_NUMBER, "frac_digits")
                    return stack
                return None

            if extra == "frac_digits":
                if char.isdigit():
                    return stack
                stack.pop()
                return step(stack, char)

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


def schema_step(state: SchemaState, char: str) -> SchemaState | None:
    """Consume one character from F2's point of view. Mirrors F1's step()."""

    if state.phase == SchemaPhase.NAME_VALUE:
        candidates = [
            fn.name for fn in state.catalog.functions
            if fn.name.startswith(state.name_prefix + char)
        ]
        if not candidates:
            return None
        # TODO: build and return a new SchemaState with name_prefix extended
        # by `char`. Don't mutate `state` in place — same rule as F1.
        ...

    if state.phase == SchemaPhase.PARAM_KEY:
        # TODO: same prefix-narrowing idea as NAME_VALUE, but filtered
        # against `state.remaining_params` instead of the whole catalog.
        ...

    if state.phase == SchemaPhase.PARAM_VALUE:
        # TODO: use `state.current_param_type` to decide which characters
        # are legal. Think about: for type "number", which characters does
        # F1 already allow that you now need to further restrict/allow
        # (hint — for "number" you don't actually need to restrict anything
        # beyond what F1 does; F1's number grammar IS the number grammar.
        # For "string"/"boolean", is there anything F2 needs to check here
        # at all, or does F1 already fully cover it? Think carefully —
        # this phase may end up being closer to "just track the prefix,
        # don't reject anything extra" for some types.)
        ...

    if state.phase == SchemaPhase.BETWEEN:
        return state  # F1 owns punctuation; F2 has nothing to add

    return None  # DONE phase: nothing further expected


if __name__ == "__main__":
    def run(text):
        state = initial_state()
        for c in text:
            state = step(state, c)
            # print(state)
            if state is None:
                print(f"{text!r}: REJECTED at {c!r}")
                return
        print(f"{text!r}: OK, final state = {state}")

    run('{"Name": "Enix", "Age": 23, "rich": true}')
    run('{"a": 23.5}')
    run('{"a": 23.}')
    run('{"a": -5}')
    run('{"a": -.5}')
    run('{"active": true, "note": null}')
    run('{"a": 1,}')
#  5 — Generation Loop Driver
