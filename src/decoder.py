from enum import Enum, auto
from dataclasses import dataclass


class RootPhase(Enum):
    """where we are before/after the single top-level value."""
    VALUE = auto()  # nothing generated yet: any value-start char is legal
    DONE = auto()   # top-lvl value complete: nothing (but whitespace) is legal


class FrameKind(Enum):
    """What kind of structural scope this stack frame represents."""
    ROOT = auto()
    OBJECT = auto()
    ARRAY = auto()


class ObjectPhase(Enum):
    """where we are inside an open {...}"""
    KEY_OR_CLOSE = auto()    # just opened `{`: `"` (key) or `}` (empty object)
    KEY = auto()             # just consumed `,`: `"` only, no `}` allowed
    COLON = auto()           # just closed a key string: need `:`
    VALUE = auto()           # just consumed `:`: need any value-start char
    COMMA_OR_CLOSE = auto()  # just closed a value: need `,` or `}`


class ArrayPhase(Enum):
    """where we are inside an open [...]"""
    VALUE_OR_CLOSE = auto()  # just opened `[`: any value-start char, or `]`
    VALUE = auto()           # just consumed `,`: value-start char only, no `]`
    COMMA_OR_CLOSE = auto()  # just closed a value: need `,` or `]`


class Mode(Enum):
    """are we mid-token or between tokens"""
    STRUCTURAL = auto()  # not inside a string/number/literal right now
    IN_STRING = auto()
    IN_NUMBER = auto()
    IN_LITERAL = auto()


@dataclass
class Frame:
    kind: FrameKind
    phase: ObjectPhase | ArrayPhase | RootPhase


@dataclass
class TrackerState:
    mode: Mode
    stack: list[Frame]


def _is_value_start(char: str) -> bool:
    """Whether 'char' can legally begin a JSON value."""
    ...


def _value_completed(stack: list[Frame]) -> TrackerState:
    """Update the top-of-stack frame's phase after a value has just
    finished, given the (already possibly-popped) stack."""
    ...


def initial_state() -> TrackerState:
    """The tracker state before any characters have been generated."""
    return TrackerState(
        mode=Mode.STRUCTURAL,
        stack=[Frame(
            kind=FrameKind.ROOT,
            phase=RootPhase)])


if __name__ == '__main__':
    s = initial_state()
    print(s)


# def is_string_legal(state: State, stack: list, candidate: str) -> bool:
#     """Check whether every character of `candidate`, applied in sequence
#     from (state, stack), stays legal."""
#     ...


# def step(state: State, stack: list, char: str) -> tuple[State, list] | None:
#     """Given current state+stack and one candidate character, return the
#     new (state, stack) if char is legal here, or None if it's not."""
#     ...


#  1 — Grammar / State Tracker
def state_tracker():
    ...


#  2 — Schema Constraint Resolver
def schema_resolver():
    ...


#  3 — Logit Masker
def logit_masker():
    ...


#  4 — Token Selector
def token_selector():
    ...


#  5 — Generation Loop Driver
