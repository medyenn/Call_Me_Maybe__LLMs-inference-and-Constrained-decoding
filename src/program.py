import time

from llm_sdk.llm_sdk import Small_LLM_Model

from models import Config
from loader import fcts_loader, vocab_loader
from decoder import generate_function_call

# --- 1. Build the SDK once ---
print("Loading model...")
t0 = time.time()
sdk = Small_LLM_Model()
print(f"Model loaded in {time.time() - t0:.1f}s")

# --- 2. Load the real vocabulary ---
vocab = vocab_loader(sdk)
print(f"Vocab loaded: {len(vocab.id_to_token)} tokens")

# --- 3. Load the real function catalog from the subject's example file ---
config = Config()
catalog = fcts_loader(config.functions_definition)
print(f"Catalog loaded: {[fn.name for fn in catalog.functions]}")

# --- 4. Run constrained generation on one trivial prompt ---
prompt = "What is the sum of 2 and 3?"

print(f"\nGenerating for prompt: {prompt!r}")
t0 = time.time()
text = generate_function_call(sdk, vocab, catalog, prompt)
elapsed = time.time() - t0

print(f"\nGenerated in {elapsed:.1f}s:")
print(text)
