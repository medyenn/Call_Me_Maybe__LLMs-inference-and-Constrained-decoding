import sys
from argparse import ArgumentParser, Namespace

from llm_sdk.llm_sdk import Small_LLM_Model

from .models import Config
from .orchestrator import PipelineError, run_pipeline


def args_parser() -> Namespace:
    """Parse command-line arguments."""
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
    parser.add_argument(
        "--time_budget_seconds",
        default=None,
        type=float,
        help="Wall-clock budget in seconds for the generation phase "
        "only, excluding model loading and output writing "
        "(default: 240.0)",
    )
    return parser.parse_args()


def build_config() -> Config:
    """Parse argv and build a validated Config."""
    namespace_obj = args_parser()
    overrides = {k: v for k, v in vars(namespace_obj).items() if v is not None}
    return Config(**overrides)


def main() -> None:
    """Entry point: run the pipeline, never crash on a known failure."""
    config = build_config()
    try:
        print("Loading model (this can take a while on first run)...")
        sdk = Small_LLM_Model()
        run_pipeline(config, sdk)
    except PipelineError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
