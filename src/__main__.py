import sys
from argparse import ArgumentParser, Namespace

from llm_sdk.llm_sdk import Small_LLM_Model

from .models import Config
from .orchestrator import PipelineError, run_pipeline


def args_parser() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with functions_definition, input, and output attributes,
        each either a user-supplied string or None if omitted.
    """
    parser = ArgumentParser(
        prog="python -m src",
        description="Translate natural language "
        "prompts into structured function calls.",
    )
    parser.add_argument(
        "--functions_definition",
        default=None,
        type=str,
        help="Path to the function definitions JSON file "
        "(default: data/input/functions_definition.json)",
    )
    parser.add_argument(
        "--input",
        default=None,
        type=str,
        help="Path to the input prompts JSON file "
        "(default: data/input/function_calling_tests.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help="Path to write the output JSON file "
        "(default: data/output/function_calling_results.json)",
    )
    return parser.parse_args()


def build_config() -> Config:
    """Parse argv and build a validated Config, falling back to defaults
    for any flag the user did not supply.

    Returns:
        A validated Config instance.
    """
    namespace_obj = args_parser()
    overrides = {k: v for k, v in vars(namespace_obj).items() if v is not None}
    return Config(**overrides)


def main() -> None:
    """Entry point: parse args, run the full pipeline, and translate
    any failure -- ours or not -- into a clean message and a non-zero
    exit code instead of a raw traceback (subject IV.1: the program
    must never crash during review).
    """
    config = build_config()
    try:
        print("Loading model (this can take a while on first run)...")
        sdk = Small_LLM_Model()
        run_pipeline(config, sdk)
    except PipelineError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Last-resort safety net: nothing should ever reach here if
        # every component raises OSError/ValueError/RuntimeError as
        # intended, but this guarantees we still fail cleanly instead
        # of crashing with a raw traceback.
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
