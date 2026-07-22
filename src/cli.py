from argparse import ArgumentParser, Namespace

from .models import Config


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with functions_definition, input, and output attributes,
        each either a user-supplied string or None if omitted.
    """
    parser = ArgumentParser(
        prog="python -m src",
        description="Translate natural language prompts into structured function calls.",
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
             "(default: data/output/function_calls.json)",
    )
    return parser.parse_args()


def build_config() -> Config:
    """Parse argv and build a validated Config, falling back to defaults
    for any flag the user did not supply.

    Returns:
        A validated Config instance.
    """
    namespace_obj = parse_args()
    overrides = {k: v for k, v in vars(namespace_obj).items() if v is not None}
    return Config(**overrides)


def main() -> None:
    """Entry point: build the config and (for now) display it."""
    config = build_config()
    print(config)


if __name__ == "__main__":
    main()