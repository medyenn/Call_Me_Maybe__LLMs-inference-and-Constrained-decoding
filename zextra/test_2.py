import sys
sys.path.insert(0, ".")

from src.decoder import state_tracker, Pos, generate_function_call
from src.models import FunctionDefinition, ParameterSpec, FunctionCatalog
from src.loader import VocabularyManager

# --- test: only legal JSON escape chars accepted after backslash ---
state = [Pos.IN_STRING]
ok = state_tracker(state, "\\")
assert ok is not None
state = ok
# 'n' is legal (\n)
assert state_tracker(state, "n") is not None
# 'q' is NOT a legal JSON escape char
state2 = state_tracker([Pos.IN_STRING], "\\")
assert state_tracker(state2, "q") is None
print("escape-char grammar: OK")

# --- full simulated generation with a fake byte-level-BPE SDK ---
catalog = FunctionCatalog(functions=[
    FunctionDefinition(
        name="fn_greet",
        description="greet",
        parameters={"name": ParameterSpec(type="string")},
        returns=ParameterSpec(type="string"),
    ),
])

TARGET = '{"name": "fn_greet", "parameters": {"name": "shrek"}}'

# Build a tiny fake vocab: one token per character of TARGET (decoded
# form, as VocabularyManager now stores it), each with a unique id.
chars = list(dict.fromkeys(TARGET))  # unique chars, preserve order
id_to_token = {i: ch for i, ch in enumerate(chars)}
token_to_id = {ch: i for i, ch in enumerate(chars)}
vocab = VocabularyManager(id_to_token=id_to_token, token_to_id=token_to_id)


class FakeSDK:
    """Greedily emits TARGET one character-token at a time; logits favor
    whichever character comes next in TARGET (but the masker is what
    actually enforces legality -- this just needs to give it a nonzero
    score so it survives masking)."""

    def __init__(self, target):
        self.target = target
        self.pos = 0

    def encode(self, text):
        return [0]

    def get_logits_from_input_ids(self, input_ids):
        logits = [0.0] * len(id_to_token)
        if self.pos < len(self.target):
            want_char = self.target[self.pos]
            logits[token_to_id[want_char]] = 100.0
        self.pos += 1
        return logits


sdk = FakeSDK(TARGET)
generated = generate_function_call(sdk, vocab, catalog, "irrelevant prompt")
print("generated:", repr(generated))
assert generated == TARGET
print("FULL GENERATION LOOP: OK, no raw byte-BPE artifacts, valid JSON")

import json
parsed = json.loads(generated)
assert parsed == {"name": "fn_greet", "parameters": {"name": "shrek"}}
print("ALL DECODER TESTS OK")
