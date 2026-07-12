from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, Sequence

try:
    from google.adk.agents import Agent
    from google.adk.apps import App
    from google.adk.events import Event
    from google.adk.models import LlmResponse
    from google.adk.runners import InMemoryRunner
    from google.adk.tools import google_search
    from google.genai import types
except ImportError:
    Agent = App = Event = LlmResponse = InMemoryRunner = google_search = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]

from hands_one_code_examples._shared.adk_runtime import (
    build_user_message,
    content_to_text,
    derive_session_id,
    event_output_to_text,
    load_environment_variables,
    require_google_adk,
    validate_runtime_environment,
)


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_APP_NAME = "google_search_rag_demo"
DEFAULT_USER_ID = "demo-user"
DEFAULT_SESSION_ID = "google-search-rag"
DEFAULT_GREETING_REQUEST = "Hi"
DEFAULT_SEARCH_REQUEST = (
    "Search Google for the latest Google Agent Development Kit updates and "
    "summarize the top takeaways with sources."
)
DEFAULT_WEB_GREETING = "Hello! I am Google Search Bot."
STATE_INTRO_SHOWN = "google_search_bot_intro_shown"


@dataclass(frozen=True)
class SearchInteractionStep:
    author: str
    kind: str
    text: str
    is_final: bool


@dataclass(frozen=True)
class SearchRunResult:
    request: str
    steps: list[SearchInteractionStep]
    final_response: str
    session_id: str


def build_google_search_instruction() -> str:
    return (
        "You are Google Search Bot.\n"
        "Introduce yourself only on the first user-facing turn in a session.\n"
        "If the user's first message already includes a concrete search request, "
        "introduce yourself and answer that request in the same response.\n"
        "If the user has not asked for anything specific yet, ask: "
        "'What can I help you search for on Google today?'\n"
        "For later turns in the same session, skip the introduction and answer "
        "the search request directly.\n"
        "When the user asks a factual, current, or web-research question, use "
        "the Google Search tool when it would improve the answer.\n"
        "Summarize findings clearly and include source URLs or source domains when "
        "available.\n"
        "If the request is too vague to search well, ask one short clarifying "
        "question instead of guessing.\n"
        "Do not fabricate search results or citations."
    )


def build_google_search_description() -> str:
    return (
        "Google Search Bot introduces itself once, then answers search requests "
        "with grounded results and sources."
    )


def _mark_intro_shown(state: Any) -> None:
    try:
        state[STATE_INTRO_SHOWN] = True
    except Exception:
        pass


def _intro_already_shown(state: Any) -> bool:
    try:
        return bool(state.get(STATE_INTRO_SHOWN))
    except Exception:
        return False


def _strip_intro_prefix(text: str) -> str:
    if not text.startswith(DEFAULT_WEB_GREETING):
        return text

    stripped = text[len(DEFAULT_WEB_GREETING) :].lstrip()
    if stripped.startswith("\n"):
        stripped = stripped.lstrip("\n").lstrip()
    return stripped


def normalize_search_response(
    callback_context: Any,
    llm_response: Any,
) -> Any | None:
    state = getattr(callback_context, "state", None)
    if state is None:
        return None

    response_content = getattr(llm_response, "content", llm_response)
    text = content_to_text(response_content)
    if not text:
        return None

    if _intro_already_shown(state):
        stripped_text = _strip_intro_prefix(text)
        if stripped_text == text:
            return None
        if types is None or LlmResponse is None:
            return None
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=stripped_text)],
            )
        )

    _mark_intro_shown(state)
    if text.startswith(DEFAULT_WEB_GREETING):
        return None

    if types is None or LlmResponse is None:
        return None

    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(text=f"{DEFAULT_WEB_GREETING}\n\n{text}")],
        )
    )


def build_google_search_agent(
    *,
    model: str = DEFAULT_MODEL,
    tool: Any = None,
) -> Any:
    require_google_adk(Agent, LlmResponse, google_search)
    active_tool = tool if tool is not None else google_search
    return Agent(
        name="google_search_bot",
        model=model,
        description=build_google_search_description(),
        instruction=build_google_search_instruction(),
        tools=[active_tool],
        after_model_callback=normalize_search_response,
    )


def build_adk_web_app(
    agent: Any | None = None,
    *,
    app_name: str = DEFAULT_APP_NAME,
) -> Any:
    require_google_adk(App)
    active_agent = agent or build_google_search_agent()
    return App(name=app_name, root_agent=active_agent)


def build_runner(agent: Any | None = None, *, app_name: str = DEFAULT_APP_NAME) -> Any:
    require_google_adk(InMemoryRunner)
    active_agent = agent or build_google_search_agent()
    return InMemoryRunner(agent=active_agent, app_name=app_name)


async def create_run_session(
    runner: Any,
    *,
    user_id: str,
    base_session_id: str,
    session_index: int,
) -> Any:
    session_id = derive_session_id(base_session_id, session_index)
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


def _get_function_calls(event: Any) -> list[Any]:
    getter = getattr(event, "get_function_calls", None)
    if callable(getter):
        return list(getter())

    parts = getattr(getattr(event, "content", None), "parts", None) or []
    return [
        function_call
        for function_call in (getattr(part, "function_call", None) for part in parts)
        if function_call is not None
    ]


def _get_function_responses(event: Any) -> list[Any]:
    getter = getattr(event, "get_function_responses", None)
    if callable(getter):
        return list(getter())

    parts = getattr(getattr(event, "content", None), "parts", None) or []
    return [
        function_response
        for function_response in (
            getattr(part, "function_response", None) for part in parts
        )
        if function_response is not None
    ]


def event_to_interaction_steps(event: Any) -> list[SearchInteractionStep]:
    author = getattr(event, "author", None) or "unknown"
    if author == "user":
        return []

    steps: list[SearchInteractionStep] = []

    for function_call in _get_function_calls(event):
        steps.append(
            SearchInteractionStep(
                author=author,
                kind="tool-call",
                text=f"Called `{getattr(function_call, 'name', 'unknown_tool')}`.",
                is_final=False,
            )
        )

    for function_response in _get_function_responses(event):
        steps.append(
            SearchInteractionStep(
                author=author,
                kind="tool-result",
                text=f"`{getattr(function_response, 'name', 'unknown_tool')}` returned a result.",
                is_final=False,
            )
        )

    text = content_to_text(getattr(event, "content", None))
    if not text:
        text = event_output_to_text(event)
    if not text:
        return steps

    is_final_response = getattr(event, "is_final_response", None)
    is_final = bool(callable(is_final_response) and is_final_response())
    steps.append(
        SearchInteractionStep(
            author=author,
            kind="final" if is_final else "message",
            text=text,
            is_final=is_final,
        )
    )
    return steps


def extract_interaction_steps(events: Sequence[Any]) -> list[SearchInteractionStep]:
    steps: list[SearchInteractionStep] = []
    for event in events:
        steps.extend(event_to_interaction_steps(event))
    return steps


def extract_final_response_text(events: Sequence[Any]) -> str:
    steps = extract_interaction_steps(events)
    for step in reversed(steps):
        if step.is_final:
            return step.text
    for step in reversed(steps):
        if step.kind in {"final", "message"}:
            return step.text
    return ""


@dataclass(frozen=True)
class GoogleSearchBotApp:
    agent: Any
    runner: Any
    app_name: str
    user_id: str
    session_id: str

    @classmethod
    def build(
        cls,
        *,
        model: str = DEFAULT_MODEL,
        app_name: str = DEFAULT_APP_NAME,
        user_id: str = DEFAULT_USER_ID,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> "GoogleSearchBotApp":
        agent = build_google_search_agent(model=model)
        runner = build_runner(agent, app_name=app_name)
        return cls(
            agent=agent,
            runner=runner,
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )

    async def run_request(
        self,
        request: str,
        *,
        session_index: int = 0,
    ) -> SearchRunResult:
        session = await create_run_session(
            self.runner,
            user_id=self.user_id,
            base_session_id=self.session_id,
            session_index=session_index,
        )
        async with aclosing(
            self.runner.run_async(
                user_id=self.user_id,
                session_id=session.id,
                new_message=build_user_message(request),
                yield_user_message=True,
            )
        ) as event_stream:
            events = [event async for event in event_stream]

        return SearchRunResult(
            request=request,
            steps=extract_interaction_steps(events),
            final_response=extract_final_response_text(events),
            session_id=session.id,
        )

    async def close(self) -> None:
        await self.runner.close()


def print_run_result(result: SearchRunResult) -> None:
    print("# Google Search Bot Demo")
    print(f"Session: {result.session_id}")
    print("User:")
    print(f"  {result.request}")
    print("Agent Workflow:")
    for index, step in enumerate(result.steps, start=1):
        print(f"  {index}. [{step.author}] [{step.kind}] {step.text}")
    print("Final Response:")
    print(f"  {result.final_response}")
    print()


def print_run_instructions() -> None:
    print("# ADK Web")
    print(
        "Run `uv run adk web hands_one_code_examples/rag_google_search_agent` "
        "from the repository root, then open http://127.0.0.1:8000."
    )
    print("Set `GOOGLE_API_KEY` before using the interactive ADK Web workflow.")
    print(
        "Start with `Hi` to trigger the introduction, then ask a live search question "
        "such as: `Search Google for the latest Gemini model updates.`"
    )
    print()


async def _run_demo() -> None:
    load_environment_variables()
    validate_runtime_environment(model=DEFAULT_MODEL)
    app = GoogleSearchBotApp.build()
    try:
        result = await app.run_request(DEFAULT_SEARCH_REQUEST, session_index=0)
    finally:
        await app.close()
    print_run_result(result)
    print_run_instructions()


def main() -> None:
    asyncio.run(_run_demo())


root_agent = (
    build_google_search_agent()
    if all(
        dependency is not None
        for dependency in (Agent, LlmResponse, InMemoryRunner, google_search, types)
    )
    else None
)

app = (
    build_adk_web_app(root_agent, app_name=DEFAULT_APP_NAME)
    if all(dependency is not None for dependency in (App, root_agent))
    else None
)


if __name__ == "__main__":
    main()
