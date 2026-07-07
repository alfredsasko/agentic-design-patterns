from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Sequence

try:
    from google.adk.agents import BaseAgent, LlmAgent
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.events import Event
    from google.adk.runners import InMemoryRunner
    from google.genai import types
except ImportError:
    BaseAgent = LlmAgent = InvocationContext = Event = InMemoryRunner = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]

from hands_one_code_examples._shared.adk_runtime import (
    build_user_message as runtime_build_user_message,
    close_agent_tree_async_clients,
    close_runner,
    content_to_text,
    derive_session_id as runtime_derive_session_id,
    ensure_session as runtime_ensure_session,
    event_output_to_text,
    event_transfer_target,
    load_project_environment,
    require_google_adk_dependencies,
    suppress_known_asyncio_shutdown_noise,
    validate_runtime_environment as runtime_validate_runtime_environment,
)


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_APP_NAME = "hierarchical_agent_demo"
DEFAULT_USER_ID = "demo-user"
DEFAULT_SESSION_ID = "hierarchical-session"
DEFAULT_GREETING_REQUEST = "Please greet the new teammate, Priya."
DEFAULT_TASK_REQUEST = "Please perform the deployment checklist task."


def load_environment_variables() -> None:
    load_project_environment(__file__)


def require_google_adk() -> None:
    require_google_adk_dependencies(
        {
            "BaseAgent": BaseAgent,
            "LlmAgent": LlmAgent,
            "InvocationContext": InvocationContext,
            "Event": Event,
            "InMemoryRunner": InMemoryRunner,
            "types": types,
        },
        example_name="hierarchical ADK example",
    )


def validate_runtime_environment(model: str = DEFAULT_MODEL) -> None:
    runtime_validate_runtime_environment(
        model,
        example_name="hierarchical ADK example",
    )


def derive_session_id(base_session_id: str, scenario_index: int) -> str:
    return runtime_derive_session_id(
        base_session_id,
        scenario_index,
        index_name="scenario_index",
    )


def build_user_message(request: str) -> Any:
    require_google_adk()
    return runtime_build_user_message(types, request)


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
    return await runtime_ensure_session(
        runner,
        user_id=user_id,
        session_id=session_id,
    )


def event_to_interaction_step(event: Any) -> AgentInteractionStep | None:
    author = getattr(event, "author", None) or "unknown"
    if author == "user":
        return None

    text = content_to_text(getattr(event, "content", None))
    if not text:
        text = event_output_to_text(event)
    if not text:
        transfer_target = event_transfer_target(event)
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
        await close_agent_tree_async_clients(self.coordinator)
        await close_runner(self.runner)


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


def run_main() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_exception_handler(suppress_known_asyncio_shutdown_noise)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(asyncio.sleep(0))
        loop.close()


if __name__ == "__main__":
    run_main()
