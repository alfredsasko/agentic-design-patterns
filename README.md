# Agentic Design Patterns

Practical coding exercises inspired by *Agentic Design Patterns: A Hands-On Guide to Building Intelligent Systems* by Antonio Gullí.

The repository is organized as a set of small, runnable examples that demonstrate common agentic design patterns and workflow styles in Python.

## What’s inside

- `hands_one_code_examples/` - chapter-based example scripts
- `tests/` - unit tests for the examples
- `pyproject.toml` - project metadata and dependencies
- `uv.lock` - locked dependency graph for reproducible installs
- `.env.example` - sample environment variables required by the examples

## Patterns covered

- Prompt chaining
- Routing
- Parallelization
- Reflection
- Tool calling
- Planning
- Deep research with OpenAI Responses API

## Requirements

- Python 3.10 or newer
- `uv`
- API access for the provider used by the example you run

## Setup

1. Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create your local environment file:

```bash
cp .env.example .env
```

3. Add the required API keys to `.env`.

4. Sync dependencies:

```bash
uv sync
```

## Environment variables

The examples currently use these variables:

- `OPENAI_API_KEY` for OpenAI and CrewAI examples
- `GOOGLE_API_KEY` for Gemini and Google ADK examples
- `GEMINI_API_KEY` is accepted by one of the routing examples

Deep research also reads these optional variables:

- `OPENAI_DEEP_RESEARCH_FAST_TEST_MODE=true` narrows search scope and reduces cost during validation runs
- `OPENAI_DEEP_RESEARCH_MAX_RETRIES` controls retry attempts for transient API failures
- `OPENAI_DEEP_RESEARCH_RETRY_BASE_DELAY_SECONDS` controls retry backoff timing

## Run examples

Run individual scripts with `uv`:

```bash
uv run python hands_one_code_examples/1_1_prompt_chaining.py
uv run python hands_one_code_examples/5_1_tools.py
uv run python hands_one_code_examples/6_1_plan.py
uv run python hands_one_code_examples/6_2_plan.py
```

Each example prints its output to the console.

## Deep Research

`hands_one_code_examples/6_2_plan.py` shows an OpenAI deep-research workflow built on the Responses API. It prints both the research reasoning and the final outcome, plus extracted citations and source references when they are available.

Use `OPENAI_DEEP_RESEARCH_FAST_TEST_MODE=true` when you want to verify the flow without spending as much time or tokens. That mode narrows the web-search context and asks the model for a shorter research pass, which is useful for local testing and CI.

## Run tests

```bash
uv run pytest -q
```

## Repository structure

```text
.
├── hands_one_code_examples/
├── tests/
├── .env.example
├── pyproject.toml
├── uv.lock
└── README.md
```

## Development notes

- Keep each example focused on a single pattern or workflow.
- Prefer small helper functions and simple classes over large scripts.
- Add a matching test file in `tests/` when introducing a new example or refactoring behavior.
- Do not commit `.env`; keep secrets local and use `.env.example` as the template.

## License

See [LICENSE](LICENSE).
