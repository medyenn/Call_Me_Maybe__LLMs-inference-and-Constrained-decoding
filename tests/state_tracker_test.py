"""Unit tests for `decoder.state_tracker` -- the pure JSON grammar
(Phase 10-F.1), independent of any function schema."""

from src.decoder import Pos, state_tracker


def test_open_brace_from_root_is_legal() -> None:
    result = state_tracker([Pos.ROOT], "{")
    assert result == [Pos.ROOT_DONE, Pos.OBJ_KEY_OR_CLOSE]


def test_bare_letter_cannot_start_any_json_value() -> None:
    # "x" is not the start of a string, number, object, array, or
    # literal (true/false/null) -- pure JSON grammar rejects it,
    # independently of any function-schema constraint.
    assert state_tracker([Pos.ROOT], "x") is None


def test_array_is_a_legal_json_value_at_the_pure_grammar_level() -> None:
    # The *pure* JSON grammar allows "[" to start a value here; it is
    # the schema layer (schema_resolver), not state_tracker, that
    # additionally requires the root value to be an object.
    assert state_tracker([Pos.ROOT], "[") is not None


def test_empty_object_closes_immediately() -> None:
    state = [Pos.ROOT_DONE, Pos.OBJ_KEY_OR_CLOSE]
    result = state_tracker(state, "}")
    assert result == [Pos.ROOT_DONE]


def test_quote_after_open_brace_starts_a_key() -> None:
    state = [Pos.ROOT_DONE, Pos.OBJ_KEY_OR_CLOSE]
    result = state_tracker(state, '"')
    assert result == [Pos.ROOT_DONE, Pos.OBJ_COLON, Pos.IN_STRING]


def test_closing_brace_not_legal_right_after_a_key() -> None:
    # {"a  <- expecting more string content or a closing quote, not "}"
    state = [Pos.ROOT_DONE, Pos.OBJ_COLON, Pos.IN_STRING]
    assert state_tracker(state, "}") == state  # just more string content


def test_colon_required_after_key_string_closes() -> None:
    state = [Pos.ROOT_DONE, Pos.OBJ_COLON, Pos.IN_STRING]
    after_quote = state_tracker(state, '"')
    assert after_quote == [Pos.ROOT_DONE, Pos.OBJ_COLON]
    assert state_tracker(after_quote, ":") == [Pos.ROOT_DONE, Pos.OBJ_VALUE]
    assert state_tracker(after_quote, "x") is None


def test_comma_or_close_after_a_value() -> None:
    # ..."a": 1  <- next must be "," or "}"
    state = state_tracker([Pos.ROOT_DONE, Pos.OBJ_VALUE], "1")
    assert state is not None
    assert state_tracker(state, ",")[-1] == Pos.OBJ_KEY
    assert state_tracker(state, "}") is not None
    assert state_tracker(state, "x") is None


def test_whitespace_is_always_legal_between_tokens() -> None:
    state = [Pos.ROOT_DONE, Pos.OBJ_KEY_OR_CLOSE]
    for ws in " \t\n\r":
        assert state_tracker(state, ws) == state


def test_number_grammar_rejects_leading_zero_with_extra_digit() -> None:
    # "01" is not valid JSON; "0" then "." is (e.g. "0.5").
    state = state_tracker([Pos.ROOT], "0")
    assert state is not None
    assert state_tracker(state, "1") is None  # "01" illegal
    assert state_tracker(state, ".") is not None  # "0." in progress


def test_number_grammar_requires_digit_after_dot() -> None:
    state = state_tracker([Pos.ROOT], "1")
    state = state_tracker(state, ".")
    assert state is not None
    assert state_tracker(state, "x") is None
    assert state_tracker(state, "5") is not None


def test_true_literal_must_match_exactly() -> None:
    state = state_tracker([Pos.ROOT], "t")
    for ch in "rue":
        assert state is not None
        state = state_tracker(state, ch)
    assert state == [Pos.ROOT_DONE]

    bad = state_tracker([Pos.ROOT], "t")
    assert state_tracker(bad, "x") is None
