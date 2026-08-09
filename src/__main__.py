"""Thin entrypoint for `python -m src`.

Per the subject's own component architecture (Phase 10-A), this
module must contain no business logic itself -- it only delegates to
`cli.main()`, which parses arguments and hands off to the
Orchestrator.
"""

from .cli import main

if __name__ == "__main__":
    main()
