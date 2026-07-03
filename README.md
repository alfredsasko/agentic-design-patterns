# agentic-design-patterns
The practical coding exercises from Agentic Design Patterns: A Hands-On Guide to Building Intelligent Systems by Antonio Gullí

## Python environment (uv)

This project uses `uv` for dependency and virtual environment management.

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Sync dependencies

```bash
~/.local/bin/uv sync
```

### Run the prompt chaining example

```bash
~/.local/bin/uv run python hands_one_code_examples/1_prompt_chaning/1_1_prompt_chaining.py
```

### Run tests

```bash
~/.local/bin/uv run pytest -q
```
