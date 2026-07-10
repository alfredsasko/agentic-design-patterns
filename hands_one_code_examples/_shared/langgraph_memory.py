from __future__ import annotations

import os
from typing import Any, Mapping

from langchain_core.messages import AIMessage, BaseMessage


def validate_openai_runtime_environment(model: str, *, example_name: str) -> None:
    if model.startswith("openai:") and not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            f"OPENAI_API_KEY not found. Set it before running this {example_name}."
        )


def build_thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    normalized_thread_id = thread_id.strip()
    if not normalized_thread_id:
        raise ValueError("thread_id must not be empty")

    return {"configurable": {"thread_id": normalized_thread_id}}


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
                continue

            if isinstance(block, Mapping) and isinstance(block.get("text"), str):
                text_parts.append(block["text"])

        return "\n".join(part for part in text_parts if part).strip()

    return str(content).strip()


def coerce_ai_message(response: Any) -> AIMessage:
    if isinstance(response, AIMessage):
        return response

    if isinstance(response, BaseMessage):
        return AIMessage(content=content_to_text(response.content))

    if isinstance(response, Mapping):
        return AIMessage(content=content_to_text(response.get("content", "")))

    return AIMessage(content=str(response))


def extract_final_text(result: Mapping[str, Any]) -> str:
    messages = list(result.get("messages", []))
    if not messages:
        raise ValueError("Conversation result did not contain any messages.")

    final_message = messages[-1]
    if isinstance(final_message, BaseMessage):
        return content_to_text(final_message.content)

    if isinstance(final_message, Mapping):
        return content_to_text(final_message.get("content", ""))

    content = getattr(final_message, "content", None)
    if content is not None:
        return content_to_text(content)

    return str(final_message).strip()
