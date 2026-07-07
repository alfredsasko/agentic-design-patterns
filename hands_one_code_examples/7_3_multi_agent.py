from __future__ import annotations

import asyncio
import re
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, Sequence

try:
    from google.adk.agents import BaseAgent, LlmAgent
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.events import Event, EventActions
    from google.adk.runners import InMemoryRunner
    from google.adk.workflow import START, Workflow, node
    from google.genai import types
except ImportError:
    BaseAgent = LlmAgent = InvocationContext = Event = EventActions = None  # type: ignore[assignment]
    InMemoryRunner = None  # type: ignore[assignment]
    START = Workflow = node = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]

from hands_one_code_examples._shared.adk_runtime import (
    build_user_message as runtime_build_user_message,
    close_agent_tree_async_clients,
    close_runner,
    content_to_text,
    derive_session_id as runtime_derive_session_id,
    ensure_session as runtime_ensure_session,
    event_node_name,
    finalize_event_loop,
    load_project_environment,
    require_google_adk_dependencies,
    suppress_known_asyncio_shutdown_noise,
    validate_runtime_environment as runtime_validate_runtime_environment,
)


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_APP_NAME = "loop_agent_incident_update_demo"
DEFAULT_USER_ID = "demo-user"
DEFAULT_SESSION_ID = "incident-update-loop"
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_REQUEST = (
    "Draft an executive incident update about a payment outage affecting EU customers. "
    "Mention customer impact, the current mitigation, and the immediate next actions."
)
DEFAULT_INITIAL_FEEDBACK = (
    "Create a lean first draft with only '## Summary' and '## Impact'. "
    "Do not add '## Next Actions' yet."
)
REQUIRED_HEADINGS = ("## Summary", "## Impact", "## Next Actions")
MAX_WORD_COUNT = 120


def load_environment_variables() -> None:
    load_project_environment(__file__)


def require_google_adk() -> None:
    require_google_adk_dependencies(
        {
            "BaseAgent": BaseAgent,
            "LlmAgent": LlmAgent,
            "InvocationContext": InvocationContext,
            "Event": Event,
            "EventActions": EventActions,
            "InMemoryRunner": InMemoryRunner,
            "Workflow": Workflow,
            "node": node,
            "START": START,
            "types": types,
        },
        example_name="loop-pattern example",
    )


def validate_runtime_environment(model: str = DEFAULT_MODEL) -> None:
    runtime_validate_runtime_environment(
        model,
        example_name="ADK loop example",
    )


def build_user_message(request: str) -> Any:
    require_google_adk()
    return runtime_build_user_message(types, request)


def derive_session_id(base_session_id: str, run_index: int) -> str:
    return runtime_derive_session_id(
        base_session_id,
        run_index,
        index_name="run_index",
    )


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def _normalize_heading_label(heading: str) -> str:
    return heading.replace("## ", "")


@dataclass(frozen=True)
class CheckResult:
    iteration_number: int
    is_complete: bool
    missing_headings: tuple[str, ...]
    word_count: int
    feedback: str


@dataclass(frozen=True)
class LoopIterationRecord:
    iteration_number: int
    draft: str
    checker_feedback: str
    completed: bool


@dataclass(frozen=True)
class LoopRunResult:
    request: str
    iterations: list[LoopIterationRecord]
    final_update: str
    completed: bool
    max_iterations: int


def evaluate_incident_update(
    draft: str,
    *,
    iteration_number: int,
    required_headings: Sequence[str] = REQUIRED_HEADINGS,
    max_word_count: int = MAX_WORD_COUNT,
) -> CheckResult:
    stripped_draft = draft.strip()
    missing_headings = tuple(
        heading for heading in required_headings if heading not in stripped_draft
    )
    word_count = _word_count(stripped_draft)
    issues: list[str] = []

    if missing_headings:
        issues.append(
            "Missing headings: "
            + ", ".join(_normalize_heading_label(heading) for heading in missing_headings)
            + "."
        )
    if word_count > max_word_count:
        issues.append(
            f"Word count is {word_count}; keep the update at or below {max_word_count} words."
        )

    if issues:
        feedback = (
            f"Iteration {iteration_number} review: continue refining.\n"
            + "\n".join(f"- {issue}" for issue in issues)
        )
        return CheckResult(
            iteration_number=iteration_number,
            is_complete=False,
            missing_headings=missing_headings,
            word_count=word_count,
            feedback=feedback,
        )

    feedback = (
        f"Iteration {iteration_number} review: all requirements satisfied. Stop the loop.\n"
        f"- Word count: {word_count}/{max_word_count}"
    )
    return CheckResult(
        iteration_number=iteration_number,
        is_complete=True,
        missing_headings=(),
        word_count=word_count,
        feedback=feedback,
    )


@node(name="ConditionChecker")
def condition_checker(ctx: Any) -> Event:
    """Deterministic reviewer that decides whether the workflow should stop."""
    state = getattr(ctx, "state", None)
    if state is None and hasattr(ctx, "session"):
        state = getattr(ctx.session, "state", None)
    if state is None:
        raise AttributeError("Workflow context does not expose state.")

    iteration_number = int(state.get("iteration_number", 0)) + 1
    state["iteration_number"] = iteration_number

    draft = str(state.get("current_update", "")).strip()
    result = evaluate_incident_update(
        draft,
        iteration_number=iteration_number,
    )

    state["checker_feedback"] = result.feedback
    state["loop_completed"] = result.is_complete
    state["current_word_count"] = result.word_count

    return Event(
        author="ConditionChecker",
        content=types.Content(
            role="model",
            parts=[types.Part(text=result.feedback)],
        ),
        actions=EventActions(
            escalate=result.is_complete,
            route=not result.is_complete,
        ),
    )


@node(name="WorkflowTerminator")
def workflow_terminator(ctx: Any) -> None:
    """Terminal workflow node used to end the routed loop cleanly."""
    del ctx
    return None


def build_writer_agent(model: str = DEFAULT_MODEL) -> Any:
    require_google_adk()
    return LlmAgent(
        name="IncidentUpdateWriter",
        model=model,
        description=(
            "Drafts and refines a concise executive incident update from deterministic feedback."
        ),
        instruction=f"""
        You are refining an executive incident update.

        Original request:
        {{incident_request}}

        Current checker feedback:
        {{checker_feedback}}

        Current draft:
        {{current_update}}

        Rules:
        - If the current draft is empty, follow the feedback literally and create a short first draft.
        - On later iterations, address every issue listed in the checker feedback.
        - The final target must contain exactly these markdown headings:
          {REQUIRED_HEADINGS[0]}
          {REQUIRED_HEADINGS[1]}
          {REQUIRED_HEADINGS[2]}
        - Keep the full update at or below {MAX_WORD_COUNT} words.
        - Return only the markdown update.
        """.strip(),
        output_key="current_update",
        include_contents="none",
    )


def build_condition_checker() -> Any:
    require_google_adk()
    return condition_checker


def build_loop_agent(
    *,
    writer: Any | None = None,
    checker: Any | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> Any:
    require_google_adk()
    if max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")

    active_writer = writer or build_writer_agent()
    active_checker = checker or build_condition_checker()
    terminator = workflow_terminator
    return Workflow(
        name="IncidentUpdateRefinementWorkflow",
        description=(
            "Runs the writer and deterministic checker through a routed graph "
            "until the update meets all requirements or the iteration limit is reached."
        ),
        edges=[
            (START, active_writer),
            (active_writer, active_checker),
            (active_checker, {True: active_writer, False: terminator}),
        ],
    )


def build_runner(agent: Any | None = None, *, app_name: str = DEFAULT_APP_NAME) -> Any:
    require_google_adk()
    active_agent = agent or build_loop_agent()
    return InMemoryRunner(agent=active_agent, app_name=app_name)


async def ensure_session(
    runner: Any,
    *,
    user_id: str,
    session_id: str,
    request: str,
) -> Any:
    return await runtime_ensure_session(
        runner,
        user_id=user_id,
        session_id=session_id,
        initial_state={
            "incident_request": request,
            "current_update": "",
            "checker_feedback": DEFAULT_INITIAL_FEEDBACK,
            "iteration_number": 0,
            "loop_completed": False,
        },
    )


def extract_iteration_records(events: Sequence[Any]) -> list[LoopIterationRecord]:
    records: list[LoopIterationRecord] = []
    current_draft = ""

    for event in events:
        author = getattr(event, "author", None) or ""
        node_name = event_node_name(event)
        effective_name = node_name or author
        text = content_to_text(getattr(event, "content", None))
        if not text:
            continue

        if effective_name == "IncidentUpdateWriter":
            current_draft = text
            continue

        if effective_name == "ConditionChecker":
            match = re.search(r"Iteration\s+(\d+)\s+review", text)
            iteration_number = int(match.group(1)) if match else len(records) + 1
            completed = "Stop the loop." in text
            records.append(
                LoopIterationRecord(
                    iteration_number=iteration_number,
                    draft=current_draft,
                    checker_feedback=text,
                    completed=completed,
                )
            )

    return records


def build_final_update(records: Sequence[LoopIterationRecord]) -> str:
    if not records:
        return ""
    return records[-1].draft


@dataclass(frozen=True)
class IncidentUpdateLoopApp:
    """OO facade around an ADK loop workflow with deterministic stopping logic."""

    loop_agent: Any
    writer: Any
    checker: Any
    runner: Any
    app_name: str = DEFAULT_APP_NAME
    user_id: str = DEFAULT_USER_ID
    session_id: str = DEFAULT_SESSION_ID
    max_iterations: int = DEFAULT_MAX_ITERATIONS

    @classmethod
    def build(
        cls,
        *,
        model: str = DEFAULT_MODEL,
        app_name: str = DEFAULT_APP_NAME,
        user_id: str = DEFAULT_USER_ID,
        session_id: str = DEFAULT_SESSION_ID,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> "IncidentUpdateLoopApp":
        writer = build_writer_agent(model)
        checker = build_condition_checker()
        loop_agent = build_loop_agent(
            writer=writer,
            checker=checker,
            max_iterations=max_iterations,
        )
        runner = build_runner(loop_agent, app_name=app_name)
        return cls(
            loop_agent=loop_agent,
            writer=writer,
            checker=checker,
            runner=runner,
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            max_iterations=max_iterations,
        )

    async def run(self, request: str = DEFAULT_REQUEST) -> LoopRunResult:
        session = await ensure_session(
            self.runner,
            user_id=self.user_id,
            session_id=self.session_id,
            request=request,
        )
        if hasattr(session, "state") and isinstance(session.state, dict):
            session.state["incident_request"] = request

        events: list[Any] = []
        async with aclosing(
            self.runner.run_async(
                user_id=self.user_id,
                session_id=self.session_id,
                new_message=build_user_message(request),
                yield_user_message=False,
            )
        ) as event_stream:
            async for event in event_stream:
                events.append(event)

        iterations = extract_iteration_records(events)
        final_update = build_final_update(iterations)
        completed = bool(iterations and iterations[-1].completed)
        return LoopRunResult(
            request=request,
            iterations=iterations,
            final_update=final_update,
            completed=completed,
            max_iterations=self.max_iterations,
        )

    async def close(self) -> None:
        await close_agent_tree_async_clients(self.loop_agent)
        await close_runner(self.runner)


def print_run_result(result: LoopRunResult) -> None:
    print("## ADK Workflow Pattern: Incident Update Refinement")
    print(f"Request: {result.request}")
    print()

    for record in result.iterations:
        print(f"### Iteration {record.iteration_number}")
        print("Draft:")
        print(record.draft or "No draft produced.")
        print()
        print("Checker:")
        print(record.checker_feedback)
        print()

    print("## Final Update")
    print(result.final_update or "No final update produced.")
    print()
    print(
        f"Completed: {'yes' if result.completed else 'no'} "
        f"(max iterations: {result.max_iterations})"
    )


async def main() -> None:
    load_environment_variables()
    validate_runtime_environment()
    running_loop = asyncio.get_running_loop()
    running_loop.set_exception_handler(suppress_known_asyncio_shutdown_noise)
    app = IncidentUpdateLoopApp.build(
        session_id=derive_session_id(DEFAULT_SESSION_ID, 0),
    )
    try:
        result = await app.run(DEFAULT_REQUEST)
        print_run_result(result)
    finally:
        await app.close()
        await asyncio.sleep(0)
        await asyncio.sleep(0)


def run_main() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_exception_handler(suppress_known_asyncio_shutdown_noise)
    try:
        loop.run_until_complete(main())
    finally:
        finalize_event_loop(loop)


if __name__ == "__main__":
    run_main()
