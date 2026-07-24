import enum


class State(enum):
    EXPECT_VALUE_START = ''
    # — legal next chars: " (begin a string), a digit or - (begin a number),
    # t/f (begin true/false), { (begin an object)
    EXPECT_OBJECT_KEY_OR_CLOSE = ''
    # — right after { or after , inside an object: legal is " (begin a key)
    # or, only if the object is empty so far, }
    IN_STRING = ''
    # — inside a "...", tracking escape sequences(a \ means the next char is
    # escaped and shouldn't be treated as the closing quote)
    AFTER_KEY_STRING = ''
    # — just closed a key's string: only : is legal
    AFTER_COLON = ''
    # — same as EXPECT_VALUE_START, just reached via a different path
    IN_NUMBER = ''
    # — inside a numeric literal (digits, optionally one ., optionally
    # exponent — decide now how much number-grammar complexity you need
    IN_LITERAL = ''
    # (true/false) — partway through matching a fixed keyword; only the next
    # specific character of that keyword is legal
    AFTER_VALUE = ''
    # — just closed a value (string/number/literal/nested object): legal is ,
    # or the closing bracket matching whatever's on top of the stack
    DONE = ''
    # — stack is empty and the top-level object closed;
    # nothing further is legal


def is_string_legal(state: State, stack: list, candidate: str) -> bool:
    """Check whether every character of `candidate`, applied in sequence
    from (state, stack), stays legal."""
    ...


def step(state: State, stack: list, char: str) -> tuple[State, list] | None:
    """Given current state+stack and one candidate character, return the
    new (state, stack) if char is legal here, or None if it's not."""
    ...


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
