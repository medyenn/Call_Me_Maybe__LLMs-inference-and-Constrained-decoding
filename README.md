*This project has been created as part of the 42 curriculum by mennih.*

# call me maybe — Function Calling with Constrained Decoding

> Making a 500-million-parameter language model speak the one language computers actually understand: valid, schema-compliant JSON — every single time.

---

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Project Structure](#project-structure)
- [Instructions](#instructions)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Running the Project](#running-the-project)
  - [Makefile Targets](#makefile-targets)
  - [CLI Arguments](#cli-arguments)
- [Algorithm Explanation](#algorithm-explanation)
- [Design Decisions](#design-decisions)
- [Performance Analysis](#performance-analysis)
- [Challenges Faced](#challenges-faced)
- [Testing Strategy](#testing-strategy)
- [Example Usage](#example-usage)
- [Resources](#resources)
- [Authors](#authors)

---

## Description

**call me maybe** is a from-scratch implementation of **function calling** for Large
Language Models, built around a single core engineering constraint: no high-level
inference or structured-generation libraries (`transformers`, `outlines`, `dspy`,
`pytorch`, etc.) are used anywhere in the decision-making pipeline.

The goal of the project is to translate a natural-language request — e.g. *"What is
the sum of 2 and 3?"* — into a precise, machine-executable instruction:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": { "a": 2.0, "b": 3.0 }
}
```

The model never answers the question itself. It only ever decides **which function**
should be called and **with which typed arguments** — the actual execution of that
function is left to whatever system consumes this output.

The interesting part of this project is not the extraction itself, but the guarantee
behind it. `Qwen/Qwen3-0.6B` is a very small language model (0.6B parameters), and
small models are notoriously unreliable when simply *asked* to produce JSON — success
rates as low as 30% are common when relying on prompting alone. This project instead
implements **constrained decoding**: a technique that intervenes directly in the
model's token-generation loop, masking out every token that would make the output
invalid, so that **producing malformed JSON becomes mathematically impossible** rather
than merely unlikely. Accuracy (picking the *right* function and values) still depends
on the model's language understanding — but structural and schema **validity is
guaranteed by construction**, independent of the model's own reliability.

In short: this project builds, by hand, the same category of mechanism used in
production tool-calling systems — without leaning on any library that would do that
work for us.

---

## Features

- Reads a natural-language prompt set and a JSON function catalog, and resolves each
  prompt to a structured function call.
- **Constrained decoding engine** built from scratch on top of raw logits, enforcing:
  - Syntactic JSON validity (balanced braces, correctly placed commas/colons/quotes).
  - Schema compliance (only real function names, only declared parameters, only
    correctly typed values).
- **100% valid, parseable JSON output**, guaranteed at the token level — not through
  post-hoc validation and retries.
- Function selection performed entirely by the model's own logits (no keyword
  matching or heuristic routing).
- Fully typed codebase (`mypy`-clean) with `pydantic`-validated data models
  throughout.
- Defensive error handling — missing files, malformed JSON, and unexpected schemas
  are reported clearly instead of crashing the program.

---

## Project Structure

```
.
├── src/                          # Application source code
│   ├── __main__.py                # CLI entrypoint
│   ├── ...
├── llm_sdk/                      # Provided SDK (copied, used read-only / public API only)
├── data/
│   └── input/
│       ├── functions_definition.json
│       └── function_calling_tests.json
│   └── output/                   # Generated at runtime — not committed
├── tests/                        # Non-graded but recommended test suite
├── pyproject.toml
├── uv.lock
├── Makefile
├── .gitignore
└── README.md
```

*(Adjust this tree to match your actual final module layout before submitting.)*

---

## Instructions

### Requirements

- Python **3.10+**
- [`uv`](https://docs.astral.sh/uv/) for dependency and environment management
- The `llm_sdk` package (provided) placed alongside `src/`

### Installation

```bash
git clone <this-repository-url>
cd call-me-maybe
make install
```

`make install` (and the reviewer's/moulinette's plain `uv sync`) resolves and installs
all dependencies — `numpy`, `pydantic`, and the local `llm_sdk` package — into an
isolated environment, with no manual virtualenv setup required.

### Running the Project

```bash
uv run python -m src [--functions_definition <file>] [--input <file>] [--output <file>]
```

All three arguments are optional. If omitted, the program falls back to:

| Argument                 | Default                                     |
|---------------------------|----------------------------------------------|
| `--functions_definition`  | `data/input/functions_definition.json`       |
| `--input`                 | `data/input/function_calling_tests.json`     |
| `--output`                | `data/output/function_calling_results.json`  |

Example with explicit paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Makefile Targets

| Target        | Purpose                                                              |
|---------------|-----------------------------------------------------------------------|
| `make install`| Installs project dependencies via `uv`.                              |
| `make run`    | Runs the main program with default paths.                            |
| `make debug`  | Runs the main program under `pdb` for step-by-step debugging.        |
| `make clean`  | Removes `__pycache__`, `.mypy_cache`, and other generated artifacts. |
| `make lint`   | Runs `flake8 .` and `mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs`. |
| `make lint-strict` *(optional)* | Runs `flake8 .` and `mypy . --strict`.             |

### CLI Arguments

| Flag                      | Required | Description                                             |
|----------------------------|:--------:|-----------------------------------------------------------|
| `--functions_definition`  | No       | Path to the JSON file describing the available functions. |
| `--input`                  | No       | Path to the JSON file containing natural-language prompts. |
| `--output`                 | No       | Path where the resulting JSON array will be written.       |

---

## Algorithm Explanation

The core of this project is a **constrained decoding loop** that wraps the raw
`llm_sdk` generation primitives (`get_logits_from_input_ids`, `get_path_to_vocab_file`,
`encode`) with a layer that guarantees every generated token keeps the output on a
legal path toward a schema-compliant JSON object.

**1. Vocabulary indexing.**
At startup, the vocabulary file (`get_path_to_vocab_file()`) is loaded once and turned
into a `token_id → token_string` lookup table. Every legality check performed later in
the loop is, at bottom, a question about the *string* a candidate token would produce
— not about its numeric ID — so this table is the foundation everything else is built
on.

**2. Decoding state.**
Before generating a single token, the target JSON structure is known in advance: a
top-level object with exactly two required keys, `name` (a string, drawn from a
*closed set* of real function names) and `parameters` (an object whose required keys
and value types are dictated by whichever function ends up selected). This is tracked
as an explicit state machine — at every position, the state answers the question
*"what is currently legal here?"*: an opening brace, a specific key, a colon, a value
of a specific declared type, or a comma/closing-brace.

**3. Per-step masking.**
At every generation step:
1. `get_logits_from_input_ids` is called with the tokens generated so far.
2. Given the current decoding state, the set of *legal next tokens* is computed —
   tokens that, if appended, would keep the output both a valid JSON prefix **and**
   consistent with the expected schema at that position (e.g. only digit/`-`/`.`
   characters while inside a `number` value; only one of the real function names
   while filling the `name` field).
3. Every logit **not** in that legal set is set to negative infinity.
4. The next token is selected — deterministically (argmax/greedy) — from the
   surviving logits, since the goal here is reliability, not creative variety.
5. The decoding state advances based on the token actually chosen, and the loop
   repeats until the object is structurally complete.

Because illegal continuations are made *impossible to select* rather than merely
discouraged, the resulting text is guaranteed to parse and to match the target
function's schema — with zero reliance on the model "getting the format right" on its
own. Function *selection itself* is resolved the same way: the `name` field is
generated under a constraint restricting legal tokens to the (small, closed) set of
real function names, letting the model's own logits — not any heuristic or keyword
match — decide which one wins among the legal options.

*(Expand this section with your actual grammar/state representation, how you scoped
the legal-token search for performance, and any simplifications or extensions —
e.g. how nested objects, arrays, or booleans are handled if you support them.)*

---

## Design Decisions

- **Pydantic for every data model.** Function definitions, prompts, and output
  records are all represented as validated `pydantic` models rather than raw
  dictionaries, so malformed input files are caught at the boundary rather than
  causing obscure failures deep inside the decoding loop.
- **Greedy token selection.** Once logits are masked to the legal set, the highest-
  scoring remaining token is chosen deterministically. There is no notion of "more
  interesting" JSON — only correct or incorrect — so sampling-based creativity adds
  risk without benefit here.
- **A single source of truth for type mapping.** The mapping between a function
  definition's declared `type` (`number`, `string`, `boolean`, …) and its legal
  token-level grammar is kept in exactly one place, and reused both by the decoding
  engine (to constrain generation) and by the final output validator (to double-check
  the result) — avoiding drift between "what the decoder allows" and "what the
  validator accepts."
- **No hardcoded functions, names, or counts.** Nothing in the pipeline assumes a
  specific number of functions, specific parameter names, or specific types beyond
  what's declared in `functions_definition.json` at runtime — this file is explicitly
  allowed to change during peer review.
- **Ambiguous-prompt policy:** *(document here what your program does when a prompt
  doesn't clearly map to any function — e.g. does the LLM still pick its
  best-guess function under the same closed-set constraint, and how is that
  communicated in the output?)*
- **Nested/array parameter support:** *(document here whether your grammar supports
  nested objects or arrays as parameter types, or whether it is deliberately scoped
  to flat scalar types, and why.)*

*(Add or adjust bullets to reflect the decisions you actually made — this section is
one of the most heavily scrutinized during peer review, since it's where you
demonstrate genuine understanding rather than incidental correctness.)*

---

## Performance Analysis

| Metric                                   | Target (per subject) | Measured |
|--------------------------------------------|:---:|:---:|
| Function selection + argument accuracy     | ≥ 90% | *[fill in]* |
| Valid, schema-compliant JSON output        | 100%  | *[fill in]* |
| Full test-set processing time              | < 5 minutes | *[fill in]* |
| Program crashes on malformed/edge-case input | 0   | *[fill in]* |

**Methodology:** *(describe how you measured accuracy — e.g. a hand-labeled set of
expected `{name, parameters}` pairs compared against your program's output — and how
you measured timing, e.g. wall-clock time for the full example test set on your
development machine, including its specs.)*

**Observations:** *(note where accuracy is strongest/weakest — e.g. function
selection vs. numeric extraction vs. string extraction — and any performance
optimizations applied to stay within the time budget, such as caching the
vocabulary-to-legal-token mapping or narrowing the search space once inside a
closed-set field.)*

---

## Challenges Faced

*(Document the real difficulties you ran into and how you resolved them. Some
starting points worth reflecting on, since they're common pain points for this
specific project:)*

- Handling tokens that don't cleanly correspond to a single JSON character (numbers,
  punctuation, or key names split unpredictably across multiple tokens).
- Keeping the per-step legal-token computation fast enough to meet the 5-minute
  budget as the function catalog or prompt count grows.
- Deciding what to do when no function is a clean match for a prompt, without falling
  back to heuristic/keyword logic (which the subject explicitly forbids).
- *(add your own)*

---

## Testing Strategy

- **Unit-level, model-free tests:** the JSON grammar/state tracker and the
  schema-constraint resolver are tested against hand-written candidate strings and a
  small hand-built function catalog, with no model calls involved — this is the
  fastest and highest-value layer of testing, since bugs here are structural, not
  probabilistic.
- **Integration tests:** the full generation loop is run against a handful of
  hand-picked prompts covering each supported parameter type.
- **System tests:** the full CLI is run end-to-end against the actual
  `data/input/` files, and the output is checked against every rule in the subject's
  validation section (valid JSON, exact key set, names present in the catalog,
  parameter names/types matching the schema).
- **Edge-case sweep:** empty prompts, very large numbers, special characters,
  type-mismatched requests, ambiguous prompts, and multi-parameter functions are all
  exercised explicitly.

*(Describe your actual test suite location and how to run it, e.g. `uv run pytest`,
plus a summary of what it covers.)*

---

## Example Usage

Given `data/input/functions_definition.json`:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    },
    "returns": { "type": "number" }
  },
  {
    "name": "fn_reverse_string",
    "description": "Reverse a string and return the reversed result.",
    "parameters": {
      "s": { "type": "string" }
    },
    "returns": { "type": "string" }
  }
]
```

And `data/input/function_calling_tests.json`:

```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Reverse the string 'hello'" }
]
```

Running:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Produces `data/output/function_calling_results.json`:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 }
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": { "s": "hello" }
  }
]
```

---

## Resources

**References used to understand the underlying concepts:**

- [RFC 8259 — The JavaScript Object Notation (JSON) Data Interchange Format](https://datatracker.ietf.org/doc/html/rfc8259) — the formal grammar constrained decoding has to respect.
- [JSON Schema](https://json-schema.org/) — the standard vocabulary for describing JSON structure, conceptually mirrored by `functions_definition.json`.
- [Sennrich, Haddow & Birch — *Neural Machine Translation of Rare Words with Subword Units* (2016)](https://arxiv.org/abs/1508.07909) — the original Byte-Pair Encoding (BPE) paper underlying modern subword tokenization.
- [Hugging Face — Tokenizers documentation](https://huggingface.co/docs/tokenizers/index) — background on how vocabularies and subword tokenization work in practice.
- [Qwen/Qwen3-0.6B model card — Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B) — the model used in this project.
- [dottxt-ai/outlines](https://github.com/dottxt-ai/outlines) — a production structured-generation library; studied for conceptual understanding of logit-masking and FSM-based constrained decoding (not used as a dependency, per project requirements).
- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — a real-world reference point for the natural-language-to-structured-call problem this project solves from scratch.
- [Pydantic documentation](https://docs.pydantic.dev/) — used for all data validation in this project.
- [mypy documentation](https://mypy.readthedocs.io/) / [flake8 documentation](https://flake8.pycqa.org/) — used for static typing and style enforcement.
- [uv documentation](https://docs.astral.sh/uv/) — used for dependency and environment management.

**How AI was used in this project:**

AI assistance (Claude) was used during the *learning and design* phase of this
project — specifically to: explain foundational concepts (tokenization, logits,
softmax, autoregressive generation, JSON as a formal grammar, and constrained
decoding); break down the subject's requirements into an explicit checklist; and help
design the component architecture (data models, module responsibilities, build order,
and testing strategy) described above. **No implementation code was generated by AI.**
All Python code in `src/` was written, tested, and is fully understood by the
author(s) listed below.

*(Adjust this paragraph to precisely and honestly reflect how you personally used AI
tools throughout the project — this section is read closely during peer review.)*

---

## Author

- `Mohamed ENNIH`
