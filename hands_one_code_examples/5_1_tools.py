from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


DEFAULT_MODEL = "google_genai:gemini-2.5-flash"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the search_information tool for factual lookups "
    "when the answer is present in the demo knowledge base."
)
DEFAULT_KNOWLEDGE_BASE = {
    "weather in london": "The weather in London is currently cloudy with a temperature of 15 C.",
    "capital of france": "The capital of France is Paris.",
    "population of earth": "The estimated population of Earth is around 8 billion people.",
    "tallest mountain": "Mount Everest is the tallest mountain above sea level.",
}


def load_environment_variables() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")


class SearchInput(BaseModel):
    """Schema shown to the model for the demo search tool."""

    query: str = Field(
        min_length=1,
        description="The factual question or lookup phrase to search for.",
    )


@dataclass
class SearchInformationService:
    """Small OO wrapper that owns the curated fact store."""

    knowledge_base: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_KNOWLEDGE_BASE))

    def __post_init__(self) -> None:
        self.knowledge_base = {
            self.normalize_query(key): value for key, value in dict(self.knowledge_base).items()
        }

    @staticmethod
    def normalize_query(query: str) -> str:
        normalized = " ".join(query.strip().lower().split())
        return normalized.rstrip("?.!")

    def search(self, query: str) -> str:
        normalized_query = self.normalize_query(query)
        if not normalized_query:
            raise ValueError("query must not be empty")

        if normalized_query in self.knowledge_base:
            return self.knowledge_base[normalized_query]

        return (
            f"No curated fact is available for '{query.strip()}'. "
            "Ask a supported demo question instead."
        )


def lookup_information(
    query: str,
    *,
    service: SearchInformationService | None = None,
) -> str:
    """Procedural helper that delegates to the service layer."""

    active_service = service or SearchInformationService()
    return active_service.search(query)


def build_search_tool(service: SearchInformationService | None = None) -> BaseTool:
    """Create a LangChain tool with a stable schema and a descriptive name."""

    active_service = service or SearchInformationService()

    @tool("search_information", args_schema=SearchInput)
    def search_information(query: str) -> str:
        """Look up short factual answers from the demo knowledge base."""

        return active_service.search(query)

    return search_information


def create_chat_model(model: str = DEFAULT_MODEL, temperature: float = 0.0) -> Any:
    """Create a chat model instance using LangChain's provider-agnostic initializer."""

    if model.startswith("google_genai:") and not os.getenv("GOOGLE_API_KEY"):
        raise ValueError(
            "GOOGLE_API_KEY not found. Set it before running this tool-calling example."
        )

    return init_chat_model(model, temperature=temperature)


def create_tool_calling_agent_app(
    *,
    model: str | Any = DEFAULT_MODEL,
    tools: Sequence[BaseTool] | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    debug: bool = False,
) -> Any:
    """Build a modern LangChain agent app using create_agent."""

    active_tools = list(tools or [build_search_tool()])
    model_instance = create_chat_model(model) if isinstance(model, str) else model
    return create_agent(
        model=model_instance,
        tools=active_tools,
        system_prompt=system_prompt,
        debug=debug,
    )


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
                continue

            if isinstance(block, dict) and isinstance(block.get("text"), str):
                text_parts.append(block["text"])

        return "\n".join(part for part in text_parts if part).strip()

    return str(content)


def extract_final_text(agent_result: Mapping[str, Any]) -> str:
    """Normalize the final assistant response into plain text."""

    messages = list(agent_result.get("messages", []))
    if not messages:
        raise ValueError("Agent result did not contain any messages.")

    final_message = messages[-1]
    if isinstance(final_message, BaseMessage):
        return _content_to_text(final_message.content)

    if isinstance(final_message, Mapping):
        return _content_to_text(final_message.get("content"))

    content = getattr(final_message, "content", None)
    if content is not None:
        return _content_to_text(content)

    return str(final_message)


class AsyncAgent(Protocol):
    async def ainvoke(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class ToolCallingAssistant:
    """OO facade around a compiled LangChain agent."""

    agent: AsyncAgent
    tools: tuple[BaseTool, ...]
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @classmethod
    def build(
        cls,
        *,
        service: SearchInformationService | None = None,
        model: str | Any = DEFAULT_MODEL,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        debug: bool = False,
    ) -> "ToolCallingAssistant":
        tool_instance = build_search_tool(service)
        agent = create_tool_calling_agent_app(
            model=model,
            tools=[tool_instance],
            system_prompt=system_prompt,
            debug=debug,
        )
        return cls(agent=agent, tools=(tool_instance,), system_prompt=system_prompt)

    async def ask(self, query: str) -> str:
        if not query.strip():
            raise ValueError("query must not be empty")

        result = await self.agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]}
        )
        return extract_final_text(result)


async def run_demo_queries(
    assistant: ToolCallingAssistant,
    queries: Sequence[str],
) -> list[str]:
    """Run multiple user questions concurrently through the same assistant."""

    if not queries:
        return []

    return list(await asyncio.gather(*(assistant.ask(query) for query in queries)))


async def main() -> None:
    load_environment_variables()
    assistant = ToolCallingAssistant.build()

    queries = [
        "What is the capital of France?",
        "What's the weather like in London?",
        "Tell me something about dogs.",
    ]
    responses = await run_demo_queries(assistant, queries)

    print("--- Running LangChain Tool Calling Example ---")
    for query, response in zip(queries, responses):
        print(f"\nQuery: {query}")
        print(f"Response: {response}")


if __name__ == "__main__":
    asyncio.run(main())
