# scratch_profile.py
import time

from llm_sdk.llm_sdk import Small_LLM_Model

from src.models import Config
from src.test_decoder import (
    initial_state,
    initial_schema_state,
    build_vocab_trie,
    logit_masker,
    schema_resolver,
)
from src.loader import fcts_loader, vocab_loader

print("Loading model...")
sdk = Small_LLM_Model()

config = Config()
vocab = vocab_loader(sdk)
catalog = fcts_loader(config.functions_definition)
trie = build_vocab_trie(vocab.id_to_token)

prompt = "What is the sum of 2 and 3?"
input_ids = sdk.encode(prompt)
if hasattr(input_ids, "tolist"):
    input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]

f1, f2 = initial_state(), initial_schema_state(catalog)
cache = {}
generated_text = ""

for step in range(30):  # just enough steps to see the pattern, not a full run
    t0 = time.time()
    logits = sdk.get_logits_from_input_ids(input_ids)
    model_time = time.time() - t0

    t0 = time.time()
    masked = logit_masker(logits, trie, f1, f2, cache)
    mask_time = time.time() - t0

    token_id = max(range(len(masked)), key=lambda i: masked[i])
    token_str = vocab.id_to_token[token_id]

    for ch in token_str:
        f1, f2 = schema_resolver(f1, f2, ch)
    input_ids = input_ids + [token_id]
    generated_text += token_str

    print(
        f"step {step:2d}  seq_len={len(input_ids):3d}  "
        f"model={model_time:.3f}s  mask={mask_time:.3f}s  "
        f"token={token_str!r}  text_so_far={generated_text!r}"
    )
