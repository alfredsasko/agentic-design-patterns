from __future__ import annotations

import asyncio
import inspect
import os
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Sequence

from dotenv import load_dotenv

try:
    from google.adk.agents import BaseAgent, LlmAgent
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.events import Event
    from google.adk.runners import InMemoryRunner
    from google.genai import types
except ImportError:
    BaseAgent = LlmAgent = InvocationContext = Event = InMemoryRunner = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_APP_NAME = "hierarchical_agent_demo"
DEFAULT_USER_ID = "demo-user"
DEFAULT_SESSION_ID = "hierarchical-session"
DEFAULT_GREETING_REQUEST = "Please greet the new teammate, Priya."
DEFAULT_TASK_REQUEST = "Please perform the deployment checklist task."


def load_environment_variables() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")


def require_google_adk() -> None:
    if (
        BaseAgent is None
        or LlmAgent is None
        or InvocationContext is None
        or Event is None
        or InMemoryRunner is None
        or types is None
    ):
        raise ImportError(
            "google-adk is not installed. Install it with `uv add google-adk` "
            "before running this hierarchical ADK example."
        )


def validate_runtime_environment(model: str = DEFAULT_MODEL) -> None:
    if model.startswith("gemini") and not os.getenv("GOOGLE_API_KEY"):
        raise ValueError(
            "GOOGLE_API_KEY not found. Set it before running this hierarchical ADK example."
        )


def derive_session_id(base_session_id: str, scenario_index: int) -> str:
    if scenario_index < 0:
        raise ValueError("scenario_index must not be negative")
    return f"{base_session_id}-{scenario_index + 1}"


def build_user_message(request: str) -> Any:
    require_google_adk()
    if not request.strip():
        raise ValueError("request must not be empty")

    user_content_type = getattr(types, "UserContent", types.Content)
    return user_content_type(parts=[types.Part(text=request)])


def _content_to_text(content: Any) -> str:
    parts = getattr(content, "parts", None) or []
    text_parts: list[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if isinstance(text, str) and text.strip():
            text_parts.append(text.strip())
    return "\n".join(text_parts).strip()


def _event_output_to_text(event: Any) -> str:
    output = getattr(event, "output", None)
    if isinstance(output, str) and output.strip():
        return output.strip()
    return ""


def _event_transfer_target(event: Any) -> str:
    actions = getattr(event, "actions", None)
    transfer_target = getattr(actions, "transfer_to_agent", None)
    if isinstance(transfer_target, str) and transfer_target.strip():
        return transfer_target.strip()
    return ""


@dataclass(frozen=True)
class AgentInteractionStep:
    author: str
    text: str
    is_final: bool


@dataclass(frozen=True)
class ScenarioResult:
    scenario_name: str
    request: str
    steps: list[AgentInteractionStep]
    final_response: str


TaskExecutorBase = BaseAgent if BaseAgent is not None else object


class TaskExecutor(TaskExecutorBase):
    """Deterministic specialist used to illustrate a non-LLM worker agent."""

    name: str = "TaskExecutor"
    description: str = (
        "Executes deterministic operational tasks delegated by the coordinator."
    )

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        del ctx
        require_google_adk()
        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=(
                            "Task finished successfully. Deployment checklist simulated: "
                            "validate configuration, run smoke tests, and confirm rollout."
                        )
                    )
                ],
            ),
        )


def build_greeter_agent(model: str = DEFAULT_MODEL) -> Any:
    require_google_adk()
    return LlmAgent(
        name="Greeter",
        model=model,
        description="Handles greeting and welcome-style requests.",
        instruction=(
            "You are a friendly greeter. When delegated a greeting task, respond with a "
            "short welcome message and mention the person's name if it is provided."
        ),
    )


def build_task_executor() -> Any:
    require_google_adk()
    return TaskExecutor()


def build_coordinator_agent(
    *,
    greeter: Any | None = None,
    task_executor: Any | None = None,
    model: str = DEFAULT_MODEL,
) -> Any:
    require_google_adk()
    active_greeter = greeter or build_greeter_agent(model)
    active_task_executor = task_executor or build_task_executor()
    return LlmAgent(
        name="Coordinator",
        model=model,
        description=(
            "Coordinator for a simple ADK hierarchy that delegates greeting requests "
            "to the Greeter and operational requests to the TaskExecutor."
        ),
        instruction=(
            "You are the coordinator of a two-agent hierarchy.\n"
            "- Delegate greetings and welcome messages to the Greeter.\n"
            "- Delegate operational or execution-style requests to the TaskExecutor.\n"
            "- Do not solve specialist tasks yourself when a matching sub-agent exists."
        ),
        sub_agents=[active_greeter, active_task_executor],
    )


def build_runner(agent: Any | None = None, *, app_name: str = DEFAULT_APP_NAME) -> Any:
    require_google_adk()
    active_agent = agent or build_coordinator_agent()
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


def event_to_interaction_step(event: Any) -> AgentInteractionStep | None:
    author = getattr(event, "author", None) or "unknown"
    if author == "user":
        return None

    text = _content_to_text(getattr(event, "content", None))
    if not text:
        text = _event_output_to_text(event)
    if not text:
        transfer_target = _event_transfer_target(event)
        if transfer_target:
            return AgentInteractionStep(
                author=author,
                text=f"Delegating to {transfer_target}.",
                is_final=False,
            )
    if not text:
        return None

    is_final_response = getattr(event, "is_final_response", None)
    is_final = bool(callable(is_final_response) and is_final_response())
    return AgentInteractionStep(author=author, text=text, is_final=is_final)


def extract_interaction_steps(events: Sequence[Any]) -> list[AgentInteractionStep]:
    steps: list[AgentInteractionStep] = []
    for event in events:
        step = event_to_interaction_step(event)
        if step is not None:
            steps.append(step)
    return steps


def extract_final_response_text(events: Sequence[Any]) -> str:
    steps = extract_interaction_steps(events)
    for step in reversed(steps):
        if step.is_final:
            return step.text
    for step in reversed(steps):
        return step.text
    raise ValueError("No textual interaction steps were captured from the agent run.")


def build_demo_requests() -> list[tuple[str, str]]:
    return [
        ("Greeting Scenario", DEFAULT_GREETING_REQUEST),
        ("Task Scenario", DEFAULT_TASK_REQUEST),
    ]


@dataclass(frozen=True)
class HierarchicalAgentDemo:
    """
    Explanatory ADK example for a coordinator/sub-agent hierarchy.

    ADK's current documentation distinguishes this routing/delegation pattern from
    fixed-order workflow agents such as SequentialAgent. Here, the coordinator uses
    `sub_agents` so an LLM can decide which specialist should handle each request.
    """

    coordinator: Any
    greeter: Any
    task_executor: Any
    runner: Any
    app_name: str = DEFAULT_APP_NAME
    user_id: str = DEFAULT_USER_ID
    session_id: str = DEFAULT_SESSION_ID

    @classmethod
    def build(
        cls,
        *,
        model: str = DEFAULT_MODEL,
        app_name: str = DEFAULT_APP_NAME,
        user_id: str = DEFAULT_USER_ID,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> "HierarchicalAgentDemo":
        greeter = build_greeter_agent(model)
        task_executor = build_task_executor()
        coordinator = build_coordinator_agent(
            greeter=greeter,
            task_executor=task_executor,
            model=model,
        )
        runner = build_runner(coordinator, app_name=app_name)
        return cls(
            coordinator=coordinator,
            greeter=greeter,
            task_executor=task_executor,
            runner=runner,
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )

    async def run_scenario(
        self,
        *,
        scenario_name: str,
        request: str,
        session_id: str,
    ) -> ScenarioResult:
        await ensure_session(
            self.runner,
            user_id=self.user_id,
            session_id=session_id,
        )
        user_message = build_user_message(request)
        events: list[Any] = []
        async with aclosing(
            self.runner.run_async(
                user_id=self.user_id,
                session_id=session_id,
                new_message=user_message,
                yield_user_message=True,
            )
        ) as event_stream:
            async for event in event_stream:
                events.append(event)

        return ScenarioResult(
            scenario_name=scenario_name,
            request=request,
            steps=extract_interaction_steps(events),
            final_response=extract_final_response_text(events),
        )

    async def close(self) -> None:
        await _close_agent_tree_async_clients(self.coordinator)
        close = getattr(self.runner, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result


async def _close_agent_tree_async_clients(agent: Any) -> None:
    model = getattr(agent, "canonical_model", None)
    if model is not None:
        api_client = getattr(model, "api_client", None)
        if api_client is not None:
            base_api_client = getattr(api_client, "_api_client", None)
            if base_api_client is not None:
                aclose = getattr(base_api_client, "aclose", None)
                if callable(aclose):
                    await aclose()
            close = getattr(api_client, "close", None)
            if callable(close):
                close()

    for sub_agent in getattr(agent, "sub_agents", []) or []:
        await _close_agent_tree_async_clients(sub_agent)


async def run_demo_requests(
    demo: HierarchicalAgentDemo,
    requests: Sequence[tuple[str, str]] | None = None,
) -> list[ScenarioResult]:
    active_requests = list(requests or build_demo_requests())
    results: list[ScenarioResult] = []
    for index, (scenario_name, request) in enumerate(active_requests):
        results.append(
            await demo.run_scenario(
                scenario_name=scenario_name,
                request=request,
                session_id=derive_session_id(demo.session_id, index),
            )
        )
    return results


def print_scenario_result(result: ScenarioResult) -> None:
    print(f"## {result.scenario_name}")
    print(f"Request: {result.request}")
    print("Interaction Steps:")
    for index, step in enumerate(result.steps, start=1):
        final_suffix = " [final]" if step.is_final else ""
        print(f"{index}. [{step.author}]{final_suffix} {step.text}")
    print("Final Response:")
    print(result.final_response)
    print()


async def main() -> None:
    load_environment_variables()
    validate_runtime_environment()
    demo = HierarchicalAgentDemo.build()
    try:
        results = await run_demo_requests(demo)
        for result in results:
            print_scenario_result(result)
    finally:
        await demo.close()


def _suppress_known_asyncio_shutdown_noise(
    loop: asyncio.AbstractEventLoop,
    context: dict[str, Any],
) -> None:
    message = str(context.get("message", ""))
    exception = context.get("exception")
    if "Fatal error on SSL transport" in message:
        return
    if isinstance(exception, RuntimeError) and "Event loop is closed" in str(exception):
        return
    loop.default_exception_handler(context)


def run_main() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_exception_handler(_suppress_known_asyncio_shutdown_noise)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(asyncio.sleep(0))
        loop.close()


if __name__ == "__main__":
    run_main()
