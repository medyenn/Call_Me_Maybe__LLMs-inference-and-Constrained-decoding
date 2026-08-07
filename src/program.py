import time

from llm_sdk.llm_sdk import Small_LLM_Model

from src.models import Config
from src.loader import fcts_loader, vocab_loader
from src.test_decoder import generate_function_call


def build_prompt(user_question: str, catalog) -> str:
    functions_desc = "\n".join(
        f"- {fn.name}({', '.join(fn.parameters)}): {fn.description}"
        for fn in catalog.functions
    )
    return (
        f"You have access to these functions:\n{functions_desc}\n\n"
        f"Given the following request, respond with a "
        f"single JSON function call.\nRequest: {user_question}"
    )


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
prompt = "What is the result of adding 2 and 3?"

print(f"\nGenerating for prompt: {prompt!r}")
t0 = time.time()
text = generate_function_call(sdk, vocab, catalog, build_prompt(prompt, catalog))
elapsed = time.time() - t0

print(f"\nGenerated in {elapsed:.1f}s:")
print(text)
