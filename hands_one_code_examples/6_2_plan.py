from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]


DEFAULT_QUERY = (
    "Research the economic impact of semaglutide on global healthcare systems."
)
# Use a smaller default model because this example is meant to validate the
# deep-research workflow, not maximize answer quality at any cost.
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_SEARCH_CONTEXT_SIZE = "medium"
DEFAULT_BLOCKED_DOMAINS = (
    "reddit.com",
    "quora.com",
    "wikipedia.org",
)
DEFAULT_SYSTEM_PROMPT = (
    "You are a professional researcher preparing a structured, data-driven report. "
    "Use the web search tool, prioritize reliable sources, cite important claims in-line, "
    "and clearly separate verified facts from inference."
)
ENV_FAST_TEST_MODE = "OPENAI_DEEP_RESEARCH_FAST_TEST_MODE"
ENV_MAX_RETRIES = "OPENAI_DEEP_RESEARCH_MAX_RETRIES"
ENV_RETRY_BASE_DELAY_SECONDS = "OPENAI_DEEP_RESEARCH_RETRY_BASE_DELAY_SECONDS"


def load_environment_variables() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")


def require_openai_package() -> None:
    if OpenAI is None:
        raise ImportError(
            "openai is not installed. Install it before running this deep research example."
        )


def _model_requires_openai_api_key(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.startswith("gpt-") or normalized.startswith("o")


def validate_runtime_environment(model: str = DEFAULT_MODEL) -> None:
    if _model_requires_openai_api_key(model) and not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY not found. Set it before running this OpenAI deep research example."
        )


def create_openai_client(*, api_key: str | None = None) -> Any:
    require_openai_package()
    return OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))


def _read_bool_env(var_name: str, *, default: bool = False) -> bool:
    raw_value = os.getenv(var_name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"{var_name} must be one of: 1, 0, true, false, yes, no, on, off"
    )


def build_runtime_config_from_environment(
    *,
    query: str = DEFAULT_QUERY,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    search_context_size: str = DEFAULT_SEARCH_CONTEXT_SIZE,
    allowed_domains: tuple[str, ...] = (),
    blocked_domains: tuple[str, ...] = DEFAULT_BLOCKED_DOMAINS,
) -> "DeepResearchConfig":
    raw_max_retries = os.getenv(ENV_MAX_RETRIES)
    raw_retry_base_delay_seconds = os.getenv(ENV_RETRY_BASE_DELAY_SECONDS)

    max_retries = int(raw_max_retries) if raw_max_retries is not None else 3
    retry_base_delay_seconds = (
        float(raw_retry_base_delay_seconds)
        if raw_retry_base_delay_seconds is not None
        else 1.0
    )

    return DeepResearchConfig(
        query=query,
        system_prompt=system_prompt,
        model=model,
        reasoning_effort=reasoning_effort,
        search_context_size=search_context_size,
        allowed_domains=allowed_domains,
        blocked_domains=blocked_domains,
        fast_test_mode=_read_bool_env(ENV_FAST_TEST_MODE, default=False),
        max_retries=max_retries,
        retry_base_delay_seconds=retry_base_delay_seconds,
    )


@dataclass(frozen=True)
class Citation:
    cited_text: str
    title: str
    url: str
    start_index: int
    end_index: int


@dataclass(frozen=True)
class SourceReference:
    title: str
    url: str


@dataclass(frozen=True)
class DeepResearchResult:
    report: str
    reasoning_summary: tuple[str, ...] = ()
    search_queries: tuple[str, ...] = ()
    citations: tuple[Citation, ...] = ()
    sources: tuple[SourceReference, ...] = ()


@dataclass(frozen=True)
class DeepResearchConfig:
    query: str = DEFAULT_QUERY
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    model: str = DEFAULT_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    search_context_size: str = DEFAULT_SEARCH_CONTEXT_SIZE
    blocked_domains: tuple[str, ...] = DEFAULT_BLOCKED_DOMAINS
    allowed_domains: tuple[str, ...] = ()
    include: tuple[str, ...] = ("web_search_call.action.sources",)
    tool_choice: str = "auto"
    fast_test_mode: bool = False
    max_retries: int = 3
    retry_base_delay_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must not be empty")
        if not self.system_prompt.strip():
            raise ValueError("system_prompt must not be empty")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError("reasoning_effort must be one of: low, medium, high")
        if self.search_context_size not in {"low", "medium", "high"}:
            raise ValueError("search_context_size must be one of: low, medium, high")
        if self.max_retries < 0:
            raise ValueError("max_retries must be greater than or equal to 0")
        if self.retry_base_delay_seconds < 0:
            raise ValueError("retry_base_delay_seconds must be non-negative")


def _build_user_query(config: DeepResearchConfig) -> str:
    if not config.fast_test_mode:
        return config.query

    return (
        f"{config.query}\n\n"
        "Fast test mode: keep the search scope narrow, use only a few high-quality sources, "
        "and return a concise research summary."
    )


def build_research_input(
    *,
    config: DeepResearchConfig,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "developer",
            "content": [{"type": "input_text", "text": config.system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": _build_user_query(config)}],
        },
    ]


def build_web_search_tool(config: DeepResearchConfig) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    if config.allowed_domains:
        filters["allowed_domains"] = list(config.allowed_domains)
    if config.blocked_domains:
        filters["blocked_domains"] = list(config.blocked_domains)

    tool: dict[str, Any] = {
        "type": "web_search",
        "search_context_size": (
            "low" if config.fast_test_mode else config.search_context_size
        ),
    }
    if filters:
        tool["filters"] = filters
    return tool


def build_response_request(config: DeepResearchConfig) -> dict[str, Any]:
    return {
        "model": config.model,
        "input": build_research_input(config=config),
        "reasoning": {
            "effort": config.reasoning_effort,
            "summary": "auto",
        },
        "tools": [build_web_search_tool(config)],
        "tool_choice": config.tool_choice,
        "include": list(config.include),
    }


def run_deep_research(
    *,
    client: Any,
    config: DeepResearchConfig,
    sleep_func: Any = time.sleep,
) -> Any:
    request = build_response_request(config)

    for attempt in range(config.max_retries + 1):
        try:
            return client.responses.create(**request)
        except Exception as exc:
            if attempt >= config.max_retries or not _is_retryable_openai_error(exc):
                raise

            delay_seconds = _calculate_retry_delay_seconds(
                exc,
                attempt=attempt,
                base_delay_seconds=config.retry_base_delay_seconds,
            )
            sleep_func(delay_seconds)

    raise RuntimeError("run_deep_research exhausted retries unexpectedly")


def parse_deep_research_response(response: Any) -> DeepResearchResult:
    output_items = tuple(_as_sequence(getattr(response, "output", ())))
    report = _extract_report_text(response, output_items)
    citations = _extract_citations(output_items, report)
    reasoning_summary = _extract_reasoning_summary(output_items)
    search_queries = _extract_search_queries(output_items)
    sources = _extract_sources(output_items)
    return DeepResearchResult(
        report=report,
        reasoning_summary=reasoning_summary,
        search_queries=search_queries,
        citations=citations,
        sources=sources,
    )


def _extract_report_text(response: Any, output_items: tuple[Any, ...]) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    for item in reversed(output_items):
        item_type = _read_field(item, "type")
        if item_type != "message":
            continue

        for content_item in _as_sequence(_read_field(item, "content")):
            text = _read_field(content_item, "text")
            if isinstance(text, str) and text.strip():
                return text.strip()

    return ""


def _extract_reasoning_summary(output_items: tuple[Any, ...]) -> tuple[str, ...]:
    summaries: list[str] = []
    for item in output_items:
        if _read_field(item, "type") != "reasoning":
            continue

        for summary_item in _as_sequence(_read_field(item, "summary")):
            text = _read_field(summary_item, "text")
            if isinstance(text, str) and text.strip():
                summaries.append(text.strip())

    return tuple(summaries)


def _extract_search_queries(output_items: tuple[Any, ...]) -> tuple[str, ...]:
    queries: list[str] = []
    for item in output_items:
        if _read_field(item, "type") != "web_search_call":
            continue

        action = _read_field(item, "action")
        query = _read_field(action, "query")
        if isinstance(query, str) and query.strip():
            queries.append(query.strip())

    return tuple(queries)


def _extract_sources(output_items: tuple[Any, ...]) -> tuple[SourceReference, ...]:
    sources: list[SourceReference] = []
    seen: set[tuple[str, str]] = set()

    for item in output_items:
        if _read_field(item, "type") != "web_search_call":
            continue

        action = _read_field(item, "action")
        for source in _as_sequence(_read_field(action, "sources")):
            title = _string_or_default(_read_field(source, "title"), "Untitled source")
            url = _string_or_default(_read_field(source, "url"), "")
            if not url:
                continue

            key = (title, url)
            if key in seen:
                continue

            seen.add(key)
            sources.append(SourceReference(title=title, url=url))

    return tuple(sources)


def _extract_citations(
    output_items: tuple[Any, ...],
    report: str,
) -> tuple[Citation, ...]:
    citations: list[Citation] = []

    for item in output_items:
        if _read_field(item, "type") != "message":
            continue

        for content_item in _as_sequence(_read_field(item, "content")):
            for annotation in _as_sequence(_read_field(content_item, "annotations")):
                start_index = _coerce_int(_read_field(annotation, "start_index"))
                end_index = _coerce_int(_read_field(annotation, "end_index"))
                title = _string_or_default(_read_field(annotation, "title"), "Untitled citation")
                url = _string_or_default(_read_field(annotation, "url"), "")
                cited_text = ""
                if start_index is not None and end_index is not None and report:
                    cited_text = report[start_index:end_index]

                citations.append(
                    Citation(
                        cited_text=cited_text,
                        title=title,
                        url=url,
                        start_index=start_index or 0,
                        end_index=end_index or 0,
                    )
                )

    return tuple(citations)


def _read_field(value: Any, field_name: str) -> Any:
    if isinstance(value, dict):
        return value.get(field_name)
    return getattr(value, field_name, None)


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _string_or_default(value: Any, default: str) -> str:
    if isinstance(value, str):
        return value.strip()
    return default


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _is_retryable_openai_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429}:
        return True

    message = str(exc).lower()
    retry_markers = (
        "rate limit",
        "rate_limit",
        "temporarily unavailable",
        "try again",
        "timeout",
        "timed out",
    )
    return any(marker in message for marker in retry_markers)


def _calculate_retry_delay_seconds(
    exc: Exception,
    *,
    attempt: int,
    base_delay_seconds: float,
) -> float:
    retry_after_seconds = _extract_retry_after_seconds(exc)
    if retry_after_seconds is not None:
        return max(retry_after_seconds, 0.0)

    return base_delay_seconds * (2**attempt)


def _extract_retry_after_seconds(exc: Exception) -> float | None:
    retry_after = getattr(exc, "retry_after", None)
    if isinstance(retry_after, (int, float)):
        return float(retry_after)

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if isinstance(headers, dict):
        header_value = headers.get("retry-after") or headers.get("Retry-After")
        try:
            if header_value is not None:
                return float(header_value)
        except (TypeError, ValueError):
            pass

    match = re.search(r"try again in\s+([0-9]+(?:\.[0-9]+)?)s", str(exc), re.IGNORECASE)
    if match:
        return float(match.group(1))

    return None


@dataclass(frozen=True)
class OpenAIDeepResearchApp:
    client: Any
    config: DeepResearchConfig = field(default_factory=DeepResearchConfig)

    @classmethod
    def build(
        cls,
        *,
        client: Any | None = None,
        config: DeepResearchConfig | None = None,
        api_key: str | None = None,
    ) -> "OpenAIDeepResearchApp":
        active_config = config or DeepResearchConfig()
        active_client = client or create_openai_client(api_key=api_key)
        return cls(client=active_client, config=active_config)

    def run(self) -> DeepResearchResult:
        response = run_deep_research(
            client=self.client,
            config=self.config,
        )
        return parse_deep_research_response(response)

    def print_execution(self) -> DeepResearchResult:
        result = self.run()

        print("## Research reasoning")
        if result.reasoning_summary:
            for summary in result.reasoning_summary:
                print(f"- {summary}")
        else:
            print("No reasoning summary was returned.")

        if result.search_queries:
            print()
            print("Search queries:")
            for query in result.search_queries:
                print(f"- {query}")

        print()
        print("## Research outcome")
        print(result.report or "No research report was returned.")

        if result.sources:
            print()
            print("## Sources")
            for source in result.sources:
                print(f"- {source.title}: {source.url}")

        return result


def main() -> None:
    load_environment_variables()
    validate_runtime_environment()
    config = build_runtime_config_from_environment()
    app = OpenAIDeepResearchApp.build(config=config)
    app.print_execution()


if __name__ == "__main__":
    main()
