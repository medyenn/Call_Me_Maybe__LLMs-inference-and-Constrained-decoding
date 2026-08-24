"""Wires the Config, loaders, decoder, and output modules into one run."""

import time

from typing import Any
from .decoder import GenerationIncomplete, generate_function_call
from .loader import fcts_loader, prompt_loader, vocab_loader
from .models import Config, FunctionCallResult, FunctionCatalog
from .output import (
    assemble_fallback_result,
    assemble_result,
    report_fallback_prompt,
    write_results,
)

PipelineError = (OSError, ValueError, RuntimeError)


def build_prompt(user_question: str, catalog: FunctionCatalog) -> str:
    """Build the natural-language prompt handed to the model."""
    functions_desc = "\n".join(
        f"- {fn.name}({', '.join(fn.parameters)}): {fn.description}"
        for fn in catalog.functions
    )
    return (
        f"You have access to these functions:\n{functions_desc}\n\n"
        f"Given the following request, respond with a single JSON "
        f"function call of the form "
        f'{{"name": "<function_name>", "parameters": {{...}}}}.\n'
        f"Example: \"Reverse 'abc'\" -> "
        f'{{"name":"fn_reverse_string","parameters":{{"s":"abc"}}}}.\n'
        f"Example: \"Replace all numbers in 'ab1 cd2' with X\" -> "
        f'{{"name":"fn_substitute_string_with_regex",'
        f'"parameters":{{"source_string":"ab1 cd2","regex":"\\\\d+",'
        f'"replacement":"X"}}}}.\n'
        f'"regex" must be a general pattern (e.g. "\\\\d+"), not a literal '
        f'from the text; "replacement" is the literal target text. '
        f"Do not copy quote marks into string values.\n\n"
        f"Request: {user_question}"
    )


def run_pipeline(config: Config, sdk: Any) -> list[FunctionCallResult]:
    """Run the full pipeline and write the output file. Every prompt
    yields one record; unresolved ones fall back to a best effort,
    all bounded by one shared wall-clock deadline for the batch.
    """
    catalog = fcts_loader(config.functions_definition)
    prompts = prompt_loader(config.input)
    vocab = vocab_loader(sdk)

    print(f"Loaded {len(catalog.functions)} function(s), "
          f"{len(prompts)} prompt(s), "
          f"{len(vocab.id_to_token)} vocab token(s).")

    results: list[FunctionCallResult] = []
    start = time.time()
    deadline = start + config.time_budget_seconds

    for i, item in enumerate(prompts, start=1):
        if time.time() >= deadline:
            result = assemble_fallback_result(item.prompt, catalog, "")
            report_fallback_prompt(
                item.prompt,
                RuntimeError(
                    "Time budget exhausted before this prompt could "
                    "be attempted."
                ),
            )
            results.append(result)
            print(f"[{i}/{len(prompts)}] {item.prompt!r} -> "
                  f"{result.name} (time budget exhausted)")
            continue

        generated_text = ""
        try:
            generated_text = generate_function_call(
                sdk, vocab, catalog, build_prompt(item.prompt, catalog),
                deadline=deadline,
            )
            result = assemble_result(item.prompt, generated_text, catalog)
        except GenerationIncomplete as e:
            result = assemble_fallback_result(
                item.prompt, catalog, e.partial_text)
            report_fallback_prompt(item.prompt, e)
        except (OSError, ValueError, RuntimeError) as e:
            result = assemble_fallback_result(
                item.prompt, catalog, generated_text)
            report_fallback_prompt(item.prompt, e)

        results.append(result)
        print(f"[{i}/{len(prompts)}] {item.prompt!r} -> {result.name}")

    write_results(results, config.output)

    elapsed = time.time() - start
    print(
        f"Processed {len(prompts)} prompt(s), wrote {len(results)} "
        f"result(s) to {config.output} in {elapsed:.1f}s."
    )
    return results
