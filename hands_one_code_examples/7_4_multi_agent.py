from __future__ import annotations

import asyncio
import os
from contextlib import aclosing
from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Sequence

try:
    from google.adk.agents import LlmAgent
    from google.adk.events import Event
    from google.adk.runners import InMemoryRunner
    from google.adk.tools import AgentTool
    from google.genai import types
except ImportError:
    LlmAgent = Event = InMemoryRunner = AgentTool = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]

from hands_one_code_examples._shared.adk_runtime import (
    build_user_message,
    content_to_text,
    derive_session_id,
    event_output_to_text,
    format_structured_value,
    load_environment_variables,
    require_google_adk,
    validate_runtime_environment,
)


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_APP_NAME = "agent_as_tool_demo"
DEFAULT_USER_ID = "demo-user"
DEFAULT_SESSION_ID = "agent-tool-session"
DEFAULT_REQUEST = (
    "Create a launch-poster concept for an AI engineering meetup. "
    "The visual should feel optimistic, modern, and suitable for LinkedIn promotion."
)


def _get_function_calls(event: Any) -> list[Any]:
    getter = getattr(event, "get_function_calls", None)
    if callable(getter):
        return list(getter())

    parts = getattr(getattr(event, "content", None), "parts", None) or []
    function_calls: list[Any] = []
    for part in parts:
        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            function_calls.append(function_call)
    return function_calls


def _get_function_responses(event: Any) -> list[Any]:
    getter = getattr(event, "get_function_responses", None)
    if callable(getter):
        return list(getter())

    parts = getattr(getattr(event, "content", None), "parts", None) or []
    function_responses: list[Any] = []
    for part in parts:
        function_response = getattr(part, "function_response", None)
        if function_response is not None:
            function_responses.append(function_response)
    return function_responses


@dataclass(frozen=True)
class WorkflowStep:
    actor: str
    kind: str
    text: str
    is_final: bool


@dataclass(frozen=True)
class AgentToolRunResult:
    request: str
    steps: list[WorkflowStep]
    final_response: str


@dataclass(frozen=True)
class MockImageStudio:
    aspect_ratio: str = "4:5"
    mime_type: str = "image/png"

    def render_asset(self, prompt: str) -> dict[str, str]:
        normalized_prompt = " ".join(prompt.strip().split())
        if not normalized_prompt:
            raise ValueError("prompt must not be empty")

        asset_hash = sha1(normalized_prompt.lower().encode("utf-8")).hexdigest()[:10]
        return {
            "status": "ready",
            "asset_id": f"mock-image-{asset_hash}",
            "prompt": normalized_prompt,
            "aspect_ratio": self.aspect_ratio,
            "mime_type": self.mime_type,
            "preview_url": f"https://example.invalid/assets/{asset_hash}.png",
        }


def generate_mock_image_asset(
    prompt: str,
    *,
    studio: MockImageStudio | None = None,
) -> dict[str, str]:
    """Create a deterministic mock image asset manifest for an illustration prompt."""

    active_studio = studio or MockImageStudio()
    return active_studio.render_asset(prompt)


def build_image_generation_tool(studio: MockImageStudio | None = None) -> Any:
    active_studio = studio or MockImageStudio()

    def generate_mock_image_asset_tool(prompt: str) -> dict[str, str]:
        """
        Create a deterministic mock image asset manifest for an illustration prompt.

        Args:
            prompt (str): A single polished prompt describing the final visual to render.
        """

        return active_studio.render_asset(prompt)

    return generate_mock_image_asset_tool


def build_illustration_specialist(
    *,
    model: str = DEFAULT_MODEL,
    tool: Any | None = None,
) -> Any:
    require_google_adk()
    active_tool = tool or build_image_generation_tool()
    return LlmAgent(
        name="IllustrationSpecialist",
        model=model,
        description=(
            "Turns a creative brief into a polished illustration prompt and produces "
            "a deterministic mock image asset manifest."
        ),
        instruction=(
            "You are an illustration specialist.\n"
            "Use the `generate_mock_image_asset_tool` tool exactly once for every request.\n"
            "First, convert the brief into a single production-ready illustration prompt.\n"
            "Then call the tool with that prompt.\n"
            "If the tool succeeds, return a concise summary containing the asset_id, "
            "the aspect ratio, and the final prompt.\n"
            "If the tool fails, explain the failure and ask the user for a clearer brief.\n"
            "Do not invent tool results."
        ),
        tools=[active_tool],
    )


def build_illustration_tool(agent: Any | None = None, *, model: str = DEFAULT_MODEL) -> Any:
    require_google_adk()
    active_agent = agent or build_illustration_specialist(model=model)
    return AgentTool(agent=active_agent)


def build_creative_director(
    *,
    model: str = DEFAULT_MODEL,
    illustration_tool: Any | None = None,
) -> Any:
    require_google_adk()
    active_illustration_tool = illustration_tool or build_illustration_tool(model=model)
    return LlmAgent(
        name="CreativeDirector",
        model=model,
        description=(
            "Decides when to delegate visual-design requests to a specialized agent tool."
        ),
        instruction=(
            "You are a creative director.\n"
            "When the user asks for a poster, illustration, image, cover, or visual concept, "
            "call the `IllustrationSpecialist` tool exactly once.\n"
            "After the tool returns, provide a short final response with two paragraphs:\n"
            "1. Explain that you delegated the image-design work to the specialist.\n"
            "2. Summarize the returned asset manifest and recommend how the user should use it.\n"
            "Do not claim you rendered the asset yourself."
        ),
        tools=[active_illustration_tool],
    )


def build_runner(agent: Any | None = None, *, app_name: str = DEFAULT_APP_NAME) -> Any:
    require_google_adk()
    active_agent = agent or build_creative_director()
    return InMemoryRunner(agent=active_agent, app_name=app_name)


async def ensure_session(
    runner: Any,
    *,
    user_id: str = DEFAULT_USER_ID,
    session_id: str = DEFAULT_SESSION_ID,
) -> Any:
    session = await runner.session_service.get_session(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id,
    )
    if session is not None:
        return session

    return await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id,
    )


def event_to_workflow_steps(event: Any) -> list[WorkflowStep]:
    actor = getattr(event, "author", None) or "unknown"
    if actor == "user":
        return []

    steps: list[WorkflowStep] = []

    for function_call in _get_function_calls(event):
        tool_name = getattr(function_call, "name", None) or "unknown_tool"
        arguments = getattr(function_call, "args", None)
        steps.append(
            WorkflowStep(
                actor=actor,
                kind="tool-call",
                text=f"Called `{tool_name}` with {format_structured_value(arguments)}.",
                is_final=False,
            )
        )

    for function_response in _get_function_responses(event):
        tool_name = getattr(function_response, "name", None) or "unknown_tool"
        response = getattr(function_response, "response", None)
        steps.append(
            WorkflowStep(
                actor=actor,
                kind="tool-result",
                text=f"`{tool_name}` returned {format_structured_value(response)}.",
                is_final=False,
            )
        )

    text = content_to_text(getattr(event, "content", None))
    if not text:
        text = event_output_to_text(event)
    if text:
        is_final_response = getattr(event, "is_final_response", None)
        is_final = bool(callable(is_final_response) and is_final_response())
        steps.append(
            WorkflowStep(
                actor=actor,
                kind="final" if is_final else "message",
                text=text,
                is_final=is_final,
            )
        )

    return steps


def extract_workflow_steps(events: Sequence[Any]) -> list[WorkflowStep]:
    steps: list[WorkflowStep] = []
    for event in events:
        steps.extend(event_to_workflow_steps(event))
    return steps


def extract_final_response_text(events: Sequence[Any]) -> str:
    steps = extract_workflow_steps(events)
    for step in reversed(steps):
        if step.is_final:
            return step.text
    for step in reversed(steps):
        if step.kind in {"message", "final"}:
            return step.text
    raise ValueError("No textual workflow steps were captured from the agent run.")


def print_workflow_result(result: AgentToolRunResult) -> None:
    print("## Agent-as-Tool Scenario")
    print(f"Request: {result.request}")
    print("Workflow Steps:")
    for index, step in enumerate(result.steps, start=1):
        final_marker = " [final]" if step.is_final and step.kind != "final" else ""
        print(f"{index}. [{step.actor}] [{step.kind}]{final_marker} {step.text}")
    print("Final Response:")
    print(result.final_response)


@dataclass
class AgentAsToolDemo:
    creative_director: Any
    illustration_specialist: Any
    runner: Any
    user_id: str = DEFAULT_USER_ID
    session_id: str = DEFAULT_SESSION_ID
    app_name: str = DEFAULT_APP_NAME

    @classmethod
    def build(
        cls,
        *,
        model: str = DEFAULT_MODEL,
        app_name: str = DEFAULT_APP_NAME,
        user_id: str = DEFAULT_USER_ID,
        session_id: str = DEFAULT_SESSION_ID,
        studio: MockImageStudio | None = None,
    ) -> AgentAsToolDemo:
        active_studio = studio or MockImageStudio()
        image_tool = build_image_generation_tool(active_studio)
        illustration_specialist = build_illustration_specialist(
            model=model,
            tool=image_tool,
        )
        illustration_agent_tool = build_illustration_tool(illustration_specialist)
        creative_director = build_creative_director(
            model=model,
            illustration_tool=illustration_agent_tool,
        )
        runner = build_runner(creative_director, app_name=app_name)
        return cls(
            creative_director=creative_director,
            illustration_specialist=illustration_specialist,
            runner=runner,
            user_id=user_id,
            session_id=session_id,
            app_name=app_name,
        )

    async def run_request(
        self,
        request: str,
        *,
        session_id: str | None = None,
    ) -> AgentToolRunResult:
        active_session_id = session_id or self.session_id
        await ensure_session(
            self.runner,
            user_id=self.user_id,
            session_id=active_session_id,
        )
        message = build_user_message(request)

        async with aclosing(
            self.runner.run_async(
                user_id=self.user_id,
                session_id=active_session_id,
                new_message=message,
                yield_user_message=True,
            )
        ) as event_stream:
            events = [event async for event in event_stream]

        return AgentToolRunResult(
            request=request,
            steps=extract_workflow_steps(events),
            final_response=extract_final_response_text(events),
        )

    async def close(self) -> None:
        close = getattr(self.runner, "close", None)
        if close is None:
            return
        result = close()
        if asyncio.iscoroutine(result):
            await result


async def run_demo_request(
    demo: AgentAsToolDemo | None = None,
    request: str = DEFAULT_REQUEST,
) -> AgentToolRunResult:
    active_demo = demo or AgentAsToolDemo.build()
    return await active_demo.run_request(request)


async def main() -> None:
    load_environment_variables()
    validate_runtime_environment()

    demo = AgentAsToolDemo.build()
    try:
        result = await run_demo_request(demo)
        print_workflow_result(result)
    finally:
        await demo.close()


if __name__ == "__main__":
    asyncio.run(main())
