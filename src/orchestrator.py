"""Orchestrator / Pipeline Controller (Phase 10 component I).

The only module that "knows about everything": it wires the Config,
the two loaders, the Vocabulary Manager, the constrained-decoding
engine, the Output Assembler, and the Output Writer into one run, and
reports progress/timing relevant to the subject's 5-minute budget
(V.5).
"""

import time

from .decoder import generate_function_call
from .exceptions import CallMeMaybeError
from .loader import fcts_loader, prompt_loader, vocab_loader
from .models import Config, FunctionCallResult, FunctionCatalog
from .output import assemble_result, report_skipped_prompt, write_results


def build_prompt(user_question: str, catalog: FunctionCatalog) -> str:
    """Build the natural-language prompt handed to the model.

    This text is only ever used for *content* decisions (which
    function, which values) -- it has no bearing on the *validity* of
    the output, which is guaranteed by the constrained-decoding engine
    regardless of how this prompt is phrased.

    Args:
        user_question: The natural-language request to fulfill.
        catalog: The available functions, listed so the model can
            pick among them.

    Returns:
        The full prompt text to feed the model.
    """
    functions_desc = "\n".join(
        f"- {fn.name}({', '.join(fn.parameters)}): {fn.description}"
        for fn in catalog.functions
    )
    return (
        f"You have access to these functions:\n{functions_desc}\n\n"
        f"Given the following request, respond with a single JSON "
        f"function call.\nRequest: {user_question}"
    )


def run_pipeline(config: Config, sdk: object) -> list[FunctionCallResult]:
    """Run the full pipeline end to end and write the output file.

    A single prompt that cannot be turned into a valid, schema
    compliant function call does not abort the batch: it is skipped
    with a warning (see `output.report_skipped_prompt`), and every
    other prompt is still processed and written (documented design
    decision -- see README "Design Decisions").

    Args:
        config: Resolved input/output paths.
        sdk: A ready-to-use llm_sdk model instance.

    Returns:
        The list of results that were successfully written, in the
        same relative order as the input prompts.

    Raises:
        CallMeMaybeError: if a fatal, non-recoverable error occurs
            (e.g. the function catalog itself cannot be loaded), or
            the output file cannot be written.
    """
    catalog = fcts_loader(config.functions_definition)
    prompts = prompt_loader(config.input)
    vocab = vocab_loader(sdk)

    print(f"Loaded {len(catalog.functions)} function(s), "
          f"{len(prompts)} prompt(s), "
          f"{len(vocab.id_to_token)} vocab token(s).")

    results: list[FunctionCallResult] = []
    start = time.time()

    for i, item in enumerate(prompts, start=1):
        try:
            generated_text = generate_function_call(
                sdk, vocab, catalog, build_prompt(item.prompt, catalog)
            )
            result = assemble_result(item.prompt, generated_text, catalog)
        except CallMeMaybeError as e:
            report_skipped_prompt(item.prompt, e)
            continue
        results.append(result)
        print(f"[{i}/{len(prompts)}] {item.prompt!r} -> {result.name}")

    write_results(results, config.output)

    elapsed = time.time() - start
    print(
        f"Processed {len(prompts)} prompt(s), wrote {len(results)} "
        f"result(s) to {config.output} in {elapsed:.1f}s."
    )
    return results
