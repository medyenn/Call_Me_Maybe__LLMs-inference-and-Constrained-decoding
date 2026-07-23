from .loader import vocab_loader  # fcts_loader, prompt_loader
# from .models import Config
from llm_sdk.llm_sdk import Small_LLM_Model
# uv run python -m src


def main():

    sdk = Small_LLM_Model()
    vocab = vocab_loader(sdk)
    print(len(vocab.id_to_token), "tokens loaded")
    for token_id in [0, 1, 100, 5000]:
        print(token_id, "->", repr(vocab.id_to_token[token_id]))
    # config = Config(
    #     functions_definition='data/input/functions_definition.json',
    #     input='data/input/function_calling_tests.json'
    #     )
    # print((5 * '\n'), fcts_loader(config.functions_definition))
    # print((5 * '\n'),  prompt_loader(config.input))


main()
#  Models Tests
# from pydantic import ValidationError

# from src.models import (
#     FunctionCallResult,
#     FunctionCatalog,
#     FunctionDefinition,
#     ParameterSpec,
#     PromptItem,
# )

# # --- 1. Build a FunctionDefinition matching fn_add_numbers from the subject
# add_numbers = FunctionDefinition(
#     name="fn_add_numbers",
#     description="Add two numbers together.",
#     parameters={
#         "a": ParameterSpec(type="number"),
#         "b": ParameterSpec(type="number"),
#     },
#     returns=ParameterSpec(type="number"),
# )
# print("1. FunctionDefinition built OK:")
# print(add_numbers)
# print()

# # --- 2. Build a second one, e.g. fn_greet, to prove the dict/list shapes
# work
# greet = FunctionDefinition(
#     name="fn_greet",
#     description="Greet someone by name.",
#     parameters={"name": ParameterSpec(type="string")},
#     returns=ParameterSpec(type="string"),
# )

# # --- 3. Wrap both into a FunctionCatalog ---
# catalog = FunctionCatalog(functions=[add_numbers, greet])
# print("2. FunctionCatalog built OK, contains:")
# for fn in catalog.functions:
#     print(" -", fn.name)
# print()

# # --- 4. Test the .get() lookup method ---
# found = catalog.get("fn_add_numbers")
# missing = catalog.get("does_not_exist")
# print("3. catalog.get('fn_add_numbers') ->", found.name if found else None)
# print("   catalog.get('does_not_exist')  ->", missing)
# print()

# # --- 5. Build a PromptItem ---
# prompt_item = PromptItem(prompt="What is the sum of 2 and 3?")
# print("4. PromptItem built OK:", prompt_item)
# print()

# # --- 6. Build a FunctionCallResult matching V.4.1's example exactly ---
# result = FunctionCallResult(
#     prompt="What is the sum of 2 and 3?",
#     name="fn_add_numbers",
#     parameters={"a": 2.0, "b": 3.0},
# )
# print("5. FunctionCallResult built OK:")
# print(result)
# print()

# # --- 7. Deliberately break it: bad type string should raise ---
# print("6. Testing that an invalid type is REJECTED:")
# try:
#     bad = ParameterSpec(type="wrong")
#     print("   FAILED — this should not have succeeded:", bad)
# except ValidationError as e:
#     print("   Correctly raised ValidationError:")
#     print("  ", e)


# if __name__ == "__main__":
#     main()
