from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, Sequence

try:
    from google.adk.agents import LlmAgent
    from google.adk.events import Event, EventActions
    from google.adk.runners import InMemoryRunner
    from google.adk.workflow import START, Workflow, node
    from google.genai import types
except ImportError:
    LlmAgent = Event = EventActions = InMemoryRunner = START = Workflow = node = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]

try:
    from google.adk.models.llm_response import LlmResponse
except ImportError:
    LlmResponse = None  # type: ignore[assignment]

try:
    from google.adk.tools import ToolContext
except ImportError:
    ToolContext = Any  # type: ignore[misc,assignment]

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
DEFAULT_APP_NAME = "location_exception_recovery_demo"
DEFAULT_USER_ID = "demo-user"
DEFAULT_SESSION_ID = "location-recovery"
DEFAULT_SUCCESS_REQUEST = (
    "Find precise location details for 221b baker street, london."
)
DEFAULT_RECOVERY_REQUEST = (
    "Find precise location details for 404 Missing Street, London."
)

STATE_REQUEST = "location_request"
STATE_LOOKUP_STATUS = "temp:lookup_status"
STATE_LOOKUP_ERROR = "temp:lookup_error"
STATE_PRECISE_RESULT = "temp:precise_result"
STATE_GENERAL_RESULT = "temp:general_result"
STATE_SELECTED_RESULT = "temp:selected_result"
STATE_FALLBACK_CITY = "temp:fallback_city"
STATE_RECOVERY_TRIGGERED = "temp:recovery_triggered"
STATE_MODEL_ERROR = "temp:model_error"
STATE_WORKFLOW_PATH = "temp:workflow_path"

WORKFLOW_PENDING = "pending"
WORKFLOW_PRECISE = "precise"
WORKFLOW_GENERAL = "general"
WORKFLOW_FALLBACK_REQUIRED = "fallback-required"


PRECISE_LOCATION_DIRECTORY: dict[str, dict[str, Any]] = {
    "221b baker street, london": {
        "resolution": WORKFLOW_PRECISE,
        "name": "Sherlock Holmes Museum",
        "address": "221B Baker Street, London",
        "city": "London",
        "country": "United Kingdom",
        "coordinates": {"lat": 51.523767, "lng": -0.1585557},
        "summary": (
            "A precise landmark record in Marylebone, close to Baker Street station."
        ),
    },
    "1600 amphitheatre parkway, mountain view": {
        "resolution": WORKFLOW_PRECISE,
        "name": "Googleplex",
        "address": "1600 Amphitheatre Parkway, Mountain View",
        "city": "Mountain View",
        "country": "United States",
        "coordinates": {"lat": 37.422, "lng": -122.0841},
        "summary": "A precise campus location in Mountain View, California.",
    },
}

GENERAL_AREA_DIRECTORY: dict[str, dict[str, Any]] = {
    "london": {
        "resolution": WORKFLOW_GENERAL,
        "city": "London",
        "region": "Greater London",
        "country": "United Kingdom",
        "summary": (
            "A broad area result for central London with dense transit coverage and "
            "many commercial and residential districts."
        ),
    },
    "mountain view": {
        "resolution": WORKFLOW_GENERAL,
        "city": "Mountain View",
        "region": "California",
        "country": "United States",
        "summary": (
            "A broad area result for Silicon Valley centered on Mountain View, "
            "useful when the street-level record is missing."
        ),
    },
}


class PreciseLocationLookupError(RuntimeError):
    """Raised when street-level lookup fails and recovery should continue."""

    def __init__(self, message: str, *, fallback_city: str | None = None) -> None:
        super().__init__(message)
        self.fallback_city = fallback_city


@dataclass(frozen=True)
class WorkflowStep:
    actor: str
    kind: str
    text: str
    is_final: bool


@dataclass(frozen=True)
class ScenarioRunResult:
    scenario_name: str
    request: str
    steps: list[WorkflowStep]
    final_response: str
    recovery_triggered: bool
    session_id: str


def normalize_location_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def normalize_address(value: str) -> str:
    normalized = normalize_location_text(value)
    return normalized.replace(" ,", ",")


def title_case_location(value: str) -> str:
    return " ".join(part.capitalize() for part in normalize_location_text(value).split())


def extract_city_from_text(value: str) -> str | None:
    normalized = normalize_address(value)
    if not normalized:
        return None

    if "," in normalized:
        city = normalized.rsplit(",", 1)[-1].strip()
        return title_case_location(city) if city else None

    words = normalized.split()
    if len(words) >= 2:
        for candidate in ("mountain view", "new york", "san francisco"):
            if candidate in normalized:
                return title_case_location(candidate)
    if words:
        return title_case_location(words[-1])
    return None


def build_initial_state(request: str) -> dict[str, Any]:
    if not request.strip():
        raise ValueError("request must not be empty")
    return {
        STATE_REQUEST: request.strip(),
        STATE_LOOKUP_STATUS: WORKFLOW_PENDING,
        STATE_RECOVERY_TRIGGERED: False,
        STATE_WORKFLOW_PATH: ["request-received"],
    }


def _state_from_context(context: Any) -> dict[str, Any]:
    state = getattr(context, "state", None)
    if isinstance(state, dict):
        return state

    session = getattr(context, "session", None)
    session_state = getattr(session, "state", None)
    if isinstance(session_state, dict):
        return session_state

    raise AttributeError("The provided context does not expose a mutable state dict.")


def append_workflow_marker(state: dict[str, Any], marker: str) -> None:
    path = list(state.get(STATE_WORKFLOW_PATH, []))
    if not path or path[-1] != marker:
        path.append(marker)
    state[STATE_WORKFLOW_PATH] = path


def build_generic_area_profile(city: str) -> dict[str, Any]:
    canonical_city = title_case_location(city)
    return {
        "resolution": WORKFLOW_GENERAL,
        "city": canonical_city,
        "region": "Unknown region",
        "country": "Unknown country",
        "summary": (
            f"No curated area profile exists for {canonical_city}. The agent recovered "
            "with a generic city-level result and recommends manual verification."
        ),
    }


def get_precise_location_info(address: str) -> dict[str, Any]:
    normalized_address = normalize_address(address)
    if not normalized_address:
        raise ValueError("address must not be empty")

    precise_result = PRECISE_LOCATION_DIRECTORY.get(normalized_address)
    if precise_result is not None:
        return dict(precise_result)

    fallback_city = extract_city_from_text(address)
    if fallback_city:
        raise PreciseLocationLookupError(
            (
                "No street-level record was found for the requested address. "
                f"Recover with a city-level lookup for {fallback_city}."
            ),
            fallback_city=fallback_city,
        )

    raise PreciseLocationLookupError(
        "No street-level record was found and no fallback city could be derived."
    )


def get_general_area_info(city: str) -> dict[str, Any]:
    normalized_city = normalize_location_text(city)
    if not normalized_city:
        raise ValueError("city must not be empty")

    area_result = GENERAL_AREA_DIRECTORY.get(normalized_city)
    if area_result is not None:
        return dict(area_result)
    return build_generic_area_profile(normalized_city)


def record_precise_lookup_success(
    tool: Any,
    args: dict[str, Any],
    tool_context: Any,
    tool_response: dict[str, Any],
) -> None:
    del tool, args
    state = _state_from_context(tool_context)
    state[STATE_PRECISE_RESULT] = tool_response
    state[STATE_SELECTED_RESULT] = tool_response
    state[STATE_LOOKUP_STATUS] = WORKFLOW_PRECISE
    state[STATE_RECOVERY_TRIGGERED] = False
    state.pop(STATE_LOOKUP_ERROR, None)
    state.pop(STATE_FALLBACK_CITY, None)
    append_workflow_marker(state, "precise-lookup-succeeded")


def record_general_lookup_success(
    tool: Any,
    args: dict[str, Any],
    tool_context: Any,
    tool_response: dict[str, Any],
) -> None:
    del tool, args
    state = _state_from_context(tool_context)
    state[STATE_GENERAL_RESULT] = tool_response
    state[STATE_SELECTED_RESULT] = tool_response
    state[STATE_LOOKUP_STATUS] = WORKFLOW_GENERAL
    state[STATE_RECOVERY_TRIGGERED] = True
    append_workflow_marker(state, "general-lookup-succeeded")


def recover_from_precise_lookup_error(
    tool: Any,
    args: dict[str, Any],
    tool_context: Any,
    error: Exception,
) -> dict[str, Any]:
    del tool
    state = _state_from_context(tool_context)
    request_text = str(args.get("address") or state.get(STATE_REQUEST, "")).strip()
    fallback_city = getattr(error, "fallback_city", None) or extract_city_from_text(
        request_text
    )
    error_message = str(error).strip() or "Precise lookup failed."

    state[STATE_LOOKUP_STATUS] = WORKFLOW_FALLBACK_REQUIRED
    state[STATE_LOOKUP_ERROR] = error_message
    state[STATE_RECOVERY_TRIGGERED] = True
    if fallback_city is not None:
        state[STATE_FALLBACK_CITY] = fallback_city
    append_workflow_marker(state, "precise-lookup-failed")

    return {
        "status": WORKFLOW_FALLBACK_REQUIRED,
        "reason": error_message,
        "fallback_city": fallback_city,
    }


def recover_from_model_error(
    callback_context: Any,
    llm_request: Any,
    error: Exception,
) -> Any:
    del llm_request
    state = _state_from_context(callback_context)
    error_message = f"{type(error).__name__}: {error}"
    state[STATE_MODEL_ERROR] = error_message
    append_workflow_marker(state, "model-error")

    if LlmResponse is None or types is None:
        return None

    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    text=(
                        "The agent encountered an internal model error while preparing "
                        "the response. Please retry the request."
                    )
                )
            ],
        ),
        error_code=type(error).__name__,
        error_message=error_message,
    )


@node(name="RecoveryRouter")
def recovery_router(ctx: Any) -> Event:
    state = _state_from_context(ctx)
    recovery_required = bool(state.get(STATE_RECOVERY_TRIGGERED, False))
    if recovery_required:
        fallback_city = state.get(STATE_FALLBACK_CITY, "the broader area")
        route_message = (
            f"Primary lookup failed. Route to the fallback agent for {fallback_city}."
        )
        append_workflow_marker(state, "route-to-fallback-agent")
    else:
        route_message = "Primary lookup succeeded. Route directly to the response agent."
        append_workflow_marker(state, "route-directly-to-response-agent")

    return Event(
        author="RecoveryRouter",
        content=types.Content(role="model", parts=[types.Part(text=route_message)]),
        actions=EventActions(route=recovery_required),
    )


@node(name="WorkflowTerminator")
def workflow_terminator(ctx: Any) -> None:
    del ctx
    return None


def build_primary_agent(model: str = DEFAULT_MODEL) -> Any:
    require_google_adk(LlmAgent)
    return LlmAgent(
        name="PreciseLocationAgent",
        model=model,
        description=(
            "Attempts a street-level location lookup and relies on a structured "
            "error callback to trigger recovery instead of crashing the workflow."
        ),
        instruction=(
            "You are the primary location resolver.\n"
            "Use the `get_precise_location_info` tool exactly once.\n"
            "Pass the full user request as the address input.\n"
            "If the tool succeeds, briefly confirm that precise data was found.\n"
            "If the tool returns a fallback-required result, acknowledge the degraded "
            "mode and let the workflow continue.\n"
            "Do not invent location data."
        ),
        tools=[get_precise_location_info],
        include_contents="none",
        after_tool_callback=record_precise_lookup_success,
        on_tool_error_callback=recover_from_precise_lookup_error,
        on_model_error_callback=recover_from_model_error,
    )


def build_fallback_agent(model: str = DEFAULT_MODEL) -> Any:
    require_google_adk(LlmAgent)
    return LlmAgent(
        name="FallbackLocationAgent",
        model=model,
        description=(
            "Recovers from a failed street-level lookup by producing a city-level result."
        ),
        instruction=(
            "You are the fallback location resolver.\n"
            "Read the fallback city from state key `temp:fallback_city`.\n"
            "Call `get_general_area_info` exactly once using that city.\n"
            "Explain that this is a degraded-but-useful recovery path.\n"
            "Do not claim the city-level result is street-accurate."
        ),
        tools=[get_general_area_info],
        include_contents="none",
        after_tool_callback=record_general_lookup_success,
        on_model_error_callback=recover_from_model_error,
    )


def build_response_agent(model: str = DEFAULT_MODEL) -> Any:
    require_google_adk(LlmAgent)
    return LlmAgent(
        name="LocationResponseAgent",
        model=model,
        description="Formats the final user-facing response from workflow state.",
        instruction=(
            "You are the final response agent.\n"
            "Read the structured workflow state.\n"
            "Use `temp:selected_result` as the authoritative result.\n"
            "If `temp:recovery_triggered` is false, explain that the exact address was "
            "resolved.\n"
            "If `temp:recovery_triggered` is true, explain that the exact lookup failed, "
            "state the recovery reason from `temp:lookup_error`, and present the "
            "city-level fallback result.\n"
            "Answer in two short paragraphs.\n"
            "Do not mention internal callback names or hidden state keys."
        ),
        include_contents="none",
        on_model_error_callback=recover_from_model_error,
    )


def build_recovery_workflow(
    *,
    primary_agent: Any | None = None,
    fallback_agent: Any | None = None,
    response_agent: Any | None = None,
) -> Any:
    require_google_adk(Workflow, START, node)
    active_primary_agent = primary_agent or build_primary_agent()
    active_fallback_agent = fallback_agent or build_fallback_agent()
    active_response_agent = response_agent or build_response_agent()
    return Workflow(
        name="LocationExceptionRecoveryWorkflow",
        description=(
            "Runs a precise lookup first, routes through structured recovery when "
            "the primary tool fails, and then produces a final response."
        ),
        edges=[
            (START, active_primary_agent),
            (active_primary_agent, recovery_router),
            (recovery_router, {False: active_response_agent, True: active_fallback_agent}),
            (active_fallback_agent, active_response_agent),
            (active_response_agent, workflow_terminator),
        ],
    )


def build_runner(agent: Any | None = None, *, app_name: str = DEFAULT_APP_NAME) -> Any:
    require_google_adk(InMemoryRunner)
    active_agent = agent or build_recovery_workflow()
    return InMemoryRunner(agent=active_agent, app_name=app_name)


async def create_run_session(
    runner: Any,
    *,
    user_id: str,
    base_session_id: str,
    session_index: int,
    state: dict[str, Any],
) -> Any:
    session_id = derive_session_id(base_session_id, session_index)
    return await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id,
        state=state,
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


def event_to_workflow_steps(event: Any) -> list[WorkflowStep]:
    actor = getattr(event, "author", None) or "unknown"
    if actor == "user":
        return []

    steps: list[WorkflowStep] = []
    for function_call in _get_function_calls(event):
        steps.append(
            WorkflowStep(
                actor=actor,
                kind="tool-call",
                text=(
                    f"Called `{getattr(function_call, 'name', 'unknown_tool')}` with "
                    f"{format_structured_value(getattr(function_call, 'args', None))}."
                ),
                is_final=False,
            )
        )

    for function_response in _get_function_responses(event):
        steps.append(
            WorkflowStep(
                actor=actor,
                kind="tool-result",
                text=(
                    f"`{getattr(function_response, 'name', 'unknown_tool')}` returned "
                    f"{format_structured_value(getattr(function_response, 'response', None))}."
                ),
                is_final=False,
            )
        )

    text = content_to_text(getattr(event, "content", None))
    if not text:
        text = event_output_to_text(event)

    if text:
        is_final_response = getattr(event, "is_final_response", None)
        is_final = bool(callable(is_final_response) and is_final_response())
        step_kind = "route" if actor == "RecoveryRouter" else ("final" if is_final else "message")
        steps.append(
            WorkflowStep(
                actor=actor,
                kind=step_kind,
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
    for event in reversed(events):
        is_final_response = getattr(event, "is_final_response", None)
        if callable(is_final_response) and is_final_response():
            text = content_to_text(getattr(event, "content", None))
            if text:
                return text
            text = event_output_to_text(event)
            if text:
                return text
    return ""


def detect_recovery_triggered(events: Sequence[Any]) -> bool:
    for event in events:
        author = getattr(event, "author", None) or ""
        if author == "FallbackLocationAgent":
            return True
        actions = getattr(event, "actions", None)
        if getattr(actions, "route", None) is True:
            return True
        for function_response in _get_function_responses(event):
            response_payload = getattr(function_response, "response", None)
            if isinstance(response_payload, dict) and response_payload.get(
                "status"
            ) == WORKFLOW_FALLBACK_REQUIRED:
                return True
    return False


@dataclass(frozen=True)
class LocationRecoveryDemo:
    primary_agent: Any
    fallback_agent: Any
    response_agent: Any
    workflow: Any
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
    ) -> "LocationRecoveryDemo":
        primary_agent = build_primary_agent(model=model)
        fallback_agent = build_fallback_agent(model=model)
        response_agent = build_response_agent(model=model)
        workflow = build_recovery_workflow(
            primary_agent=primary_agent,
            fallback_agent=fallback_agent,
            response_agent=response_agent,
        )
        runner = build_runner(workflow, app_name=app_name)
        return cls(
            primary_agent=primary_agent,
            fallback_agent=fallback_agent,
            response_agent=response_agent,
            workflow=workflow,
            runner=runner,
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )

    async def run_request(
        self,
        request: str,
        *,
        scenario_name: str,
        session_index: int,
    ) -> ScenarioRunResult:
        state = build_initial_state(request)
        session = await create_run_session(
            self.runner,
            user_id=self.user_id,
            base_session_id=self.session_id,
            session_index=session_index,
            state=state,
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

        steps = extract_workflow_steps(events)
        final_response = extract_final_response_text(events)
        recovery_triggered = detect_recovery_triggered(events)

        return ScenarioRunResult(
            scenario_name=scenario_name,
            request=request,
            steps=steps,
            final_response=final_response,
            recovery_triggered=recovery_triggered,
            session_id=session.id,
        )

    async def run_demo(self) -> list[ScenarioRunResult]:
        return [
            await self.run_request(
                DEFAULT_SUCCESS_REQUEST,
                scenario_name="Successful Precise Lookup",
                session_index=0,
            ),
            await self.run_request(
                DEFAULT_RECOVERY_REQUEST,
                scenario_name="Recovered Fallback Lookup",
                session_index=1,
            ),
        ]

    async def close(self) -> None:
        await self.runner.close()


def print_run_result(result: ScenarioRunResult) -> None:
    print(f"## {result.scenario_name}")
    print(f"Session: {result.session_id}")
    print(f"Recovery triggered: {'yes' if result.recovery_triggered else 'no'}")
    print("User:")
    print(f"  {result.request}")
    print("Agent Workflow:")
    for index, step in enumerate(result.steps, start=1):
        print(f"  {index}. [{step.actor}] [{step.kind}] {step.text}")
    print("Final Response:")
    print(f"  {result.final_response}")
    print()


def print_demo_results(results: Sequence[ScenarioRunResult]) -> None:
    print("# Exception Handling And Recovery Demo")
    print()
    for result in results:
        print_run_result(result)


async def _run_demo() -> None:
    load_environment_variables()
    validate_runtime_environment(model=DEFAULT_MODEL)
    demo = LocationRecoveryDemo.build()
    try:
        results = await demo.run_demo()
    finally:
        await demo.close()
    print_demo_results(results)


def main() -> None:
    asyncio.run(_run_demo())


if __name__ == "__main__":
    main()
