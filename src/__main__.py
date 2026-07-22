from .cli import main

if __name__ == "__main__":
    main()

# from llm_sdk.llm_sdk import Small_LLM_Model

# model = Small_LLM_Model()

# prompt = "What's 10 + 27? Give me only the exact result."

# inputs = model.encode(prompt)

# output = model._model.generate(
#     input_ids=inputs,
#     max_new_tokens=20,
# )

# print(model.decode(output[0]))
