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

| Chapter | Pattern | Framework | File Name | Run Example |
| --- | --- | --- | --- | --- |
| 1. Prompt Chaining | Two-step extraction pipeline | LangChain | `1_1_prompt_chaining.py` | `uv run python hands_one_code_examples/run_example.py hands_one_code_examples/1_1_prompt_chaining.py` |
| 2. Routing | LLM router with branch handlers | LangChain | `2_1_routing.py` | `uv run python hands_one_code_examples/run_example.py hands_one_code_examples/2_1_routing.py` |
| 2. Routing | Coordinator with delegated specialists | Google ADK | `2_2_routing.py` | `uv run python hands_one_code_examples/run_example.py hands_one_code_examples/2_2_routing.py` |
| 3. Parallelization | Parallel prompt fan-out with synthesis | LangChain | `3_1_parallelization.py` | `uv run python hands_one_code_examples/run_example.py hands_one_code_examples/3_1_parallelization.py` |
| 3. Parallelization | Parallel specialist fan-out | Google ADK | `3_2_parallelization.py` | `uv run python hands_one_code_examples/run_example.py hands_one_code_examples/3_2_parallelization.py` |
| 3. Parallelization | Workflow fan-out and join | Google ADK | `3_3_parallelization.py` | `uv run python hands_one_code_examples/run_example.py hands_one_code_examples/3_3_parallelization.py` |
| 4. Reflection | Generate-critique refinement loop | LangChain | `4_1_reflection.py` | `uv run python hands_one_code_examples/run_example.py hands_one_code_examples/4_1_reflection.py` |
| 4. Reflection | Generator-reviewer workflow | Google ADK | `4_2_reflection.py` | `uv run python hands_one_code_examples/run_example.py hands_one_code_examples/4_2_reflection.py` |
| 4. Reflection | Structured self-critique loop | LangChain | `4_3_reflection.py` | `uv run python hands_one_code_examples/run_example.py hands_one_code_examples/4_3_reflection.py` |
| 4. Reflection | Graph-based revision loop | LangGraph | `4_4_reflection.py` | `uv run python hands_one_code_examples/run_example.py hands_one_code_examples/4_4_reflection.py` |
| 5. Tool Use | Single-tool agent | LangChain | `5_1_tools.py` | `uv run python hands_one_code_examples/run_example.py hands_one_code_examples/5_1_tools.py` |
| 5. Tool Use | Custom tool-backed agent | CrewAI | `5_2_tools.py` | `uv run python hands_one_code_examples/run_example.py hands_one_code_examples/5_2_tools.py` |
| 5. Tool Use | Deterministic function tool | Google ADK | `5_3_tools.py` | `uv run python hands_one_code_examples/run_example.py hands_one_code_examples/5_3_tools.py` |
| 6. Planning | Plan-then-write crew | CrewAI | `6_1_plan.py` | `uv run python hands_one_code_examples/run_example.py hands_one_code_examples/6_1_plan.py` |
| 6. Planning | Web-research report workflow | OpenAI Responses API | `6_2_plan.py` | `uv run python hands_one_code_examples/run_example.py hands_one_code_examples/6_2_plan.py` |
| 7. Multi-Agent | Research-to-writer handoff | CrewAI | `7_1_multi_agent.py` | `uv run python hands_one_code_examples/run_example.py hands_one_code_examples/7_1_multi_agent.py` |
| 7. Multi-Agent | Hierarchical coordinator delegation | Google ADK | `7_2_multi_agent.py` | `uv run python hands_one_code_examples/run_example.py hands_one_code_examples/7_2_multi_agent.py` |
| 7. Multi-Agent | Writer-checker iteration loop | Google ADK | `7_3_multi_agent.py` | `uv run python hands_one_code_examples/run_example.py hands_one_code_examples/7_3_multi_agent.py` |
| 7. Multi-Agent | Specialist agent as tool | Google ADK | `7_4_multi_agent.py` | `uv run python hands_one_code_examples/run_example.py hands_one_code_examples/7_4_multi_agent.py` |
| 8. Memory | Short-term conversation memory | LangGraph | `8_1_memory.py` | `uv run python hands_one_code_examples/run_example.py hands_one_code_examples/8_1_memory.py` |
| 8. Memory | Long-term self-improving prompts | LangGraph | `8_2_memory.py` | `uv run python hands_one_code_examples/run_example.py hands_one_code_examples/8_2_memory.py` |
| 10. MCP | Filesystem agent with McpToolset tracing | Google ADK | `mcp_agent/agent.py` | `uv run adk web hands_one_code_examples/mcp_agent` |
| 11. Goals & Monitoring | Concept of self improving coding problem solver | LangChain | `11_1_goal_monitoring.py` | `uv run python hands_one_code_examples/run_example.py hands_one_code_examples/11_1_goal_monitoring.py` |
| 11. Goals & Monitoring | Product grade coding problem solver | CrewAI | `11_2_goal_monitoring.py` | `uv run python hands_one_code_examples/run_example.py hands_one_code_examples/11_2_goal_monitoring.py` |
| 12. Exception handling and recovery | Product grade location services gracefull tool degradation | Goolge ADK | `12_1_excpetion_recovery.py` | `uv run python hands_one_code_examples/run_example.py hands_one_code_examples/12_1_exception_recovery.py` |
| 13. Human in the loop | Supervisor-reviewed refund approval with execution | Google ADK | `13_1_human_in_the_loop.py` | `uv run adk web hands_one_code_examples/human_in_the_loop_agent` |
| 14. RAG | Google Search grounded research bot with source-aware answers | Google ADK | `14_1_rag.py` | `uv run adk web hands_one_code_examples/rag_google_search_agent` |

## Requirements

- Python 3.10 or newer
- `uv`
- Node.js with `npx` available on `PATH` for the MCP filesystem server example
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

`adk web` commands in the table start a local UI at `http://127.0.0.1:8000`. Use them for the interactive ADK examples, including the human-in-the-loop and Google Search grounded RAG agents.

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

Use the GitFlow branch structure for changes in this repository:

```bash
git checkout feature/my-change
git add hands_one_code_examples/my_example.py tests/test_my_example.py
git commit -m "feat: describe the change"
git checkout dev
git merge --no-ff feature/my-change -m "Merge feature/my-change into dev"
git push origin dev
git checkout main
git merge --no-ff dev -m "Merge dev into main"
git push origin main
git checkout feature/my-change
```

Keep each example focused on a single pattern or workflow. Prefer small helper functions and simple classes over large scripts. Add a matching test file in `tests/` when introducing a new example or refactoring behavior. Do not commit `.env`; keep secrets local and use `.env.example` as the template.

## License

See [LICENSE](LICENSE).
