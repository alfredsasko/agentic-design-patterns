from __future__ import annotations

import pathlib
import sys
import types as builtin_types
import uuid
from contextlib import contextmanager

import pytest

from tests._support.fake_adk import load_module_with_fake_adk, make_event, run


MODULE_PATH = pathlib.Path("hands_one_code_examples/12_1_exception_recovery.py")
FAKE_MODULE_NAMES = (
    "google",
    "google.adk",
    "google.adk.agents",
    "google.adk.agents.invocation_context",
    "google.adk.events",
    "google.adk.runners",
    "google.adk.tools",
    "google.adk.workflow",
    "google.genai",
)


@contextmanager
def load_example():
    module_name = f"exception_recovery_example_{uuid.uuid4().hex}"
    saved_modules = {name: sys.modules.get(name) for name in FAKE_MODULE_NAMES}
    saved_module = sys.modules.get(module_name)
    try:
        module = load_module_with_fake_adk(
            module_path=MODULE_PATH,
            module_name=module_name,
            include_workflow=True,
        )
        yield module
    finally:
        sys.modules.pop(module_name, None)
        if saved_module is not None:
            sys.modules[module_name] = saved_module
        for name, original in saved_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_precise_lookup_and_city_extraction_are_deterministic():
    with load_example() as exception_recovery_example:
        result = exception_recovery_example.get_precise_location_info(
            "221B Baker Street, London"
        )

        assert result["resolution"] == exception_recovery_example.WORKFLOW_PRECISE
        assert result["city"] == "London"
        assert (
            exception_recovery_example.extract_city_from_text(
                "404 Missing Street, London"
            )
            == "London"
        )


def test_recover_from_precise_lookup_error_updates_state_and_returns_fallback_payload():
    with load_example() as exception_recovery_example:
        state = exception_recovery_example.build_initial_state(
            "Find precise location details for 404 Missing Street, London."
        )
        tool_context = builtin_types.SimpleNamespace(state=state)
        error = exception_recovery_example.PreciseLocationLookupError(
            "No street-level record was found.",
            fallback_city="London",
        )

        payload = exception_recovery_example.recover_from_precise_lookup_error(
            tool=builtin_types.SimpleNamespace(name="get_precise_location_info"),
            args={"address": "404 Missing Street, London"},
            tool_context=tool_context,
            error=error,
        )

        assert payload["status"] == exception_recovery_example.WORKFLOW_FALLBACK_REQUIRED
        assert payload["fallback_city"] == "London"
        assert state[exception_recovery_example.STATE_RECOVERY_TRIGGERED] is True
        assert state[exception_recovery_example.STATE_FALLBACK_CITY] == "London"
        assert state[exception_recovery_example.STATE_LOOKUP_STATUS] == (
            exception_recovery_example.WORKFLOW_FALLBACK_REQUIRED
        )


def test_recovery_router_uses_shared_state_to_route():
    with load_example() as exception_recovery_example:
        success_context = builtin_types.SimpleNamespace(
            state=exception_recovery_example.build_initial_state(
                "Find precise location details for 221B Baker Street, London."
            )
        )
        success_event = exception_recovery_example.recovery_router(success_context)

        failure_state = exception_recovery_example.build_initial_state(
            "Find precise location details for 404 Missing Street, London."
        )
        failure_state[exception_recovery_example.STATE_RECOVERY_TRIGGERED] = True
        failure_state[exception_recovery_example.STATE_FALLBACK_CITY] = "London"
        failure_context = builtin_types.SimpleNamespace(state=failure_state)
        failure_event = exception_recovery_example.recovery_router(failure_context)

        assert success_event.actions.route is False
        assert (
            "Route directly to the response agent."
            in success_event.content.parts[0].text
        )
        assert failure_event.actions.route is True
        assert (
            "Route to the fallback agent for London."
            in failure_event.content.parts[0].text
        )


def test_build_workflow_wires_primary_router_fallback_and_response_agents():
    with load_example() as exception_recovery_example:
        demo = exception_recovery_example.LocationRecoveryDemo.build(
            app_name="location-recovery-app",
            user_id="alice",
            session_id="location-session",
        )

        assert demo.primary_agent.name == "PreciseLocationAgent"
        assert demo.fallback_agent.name == "FallbackLocationAgent"
        assert demo.response_agent.name == "LocationResponseAgent"
        assert demo.workflow.edges[0][0].name == "START"
        assert demo.workflow.edges[0][1] is demo.primary_agent
        assert demo.workflow.edges[1][1] is exception_recovery_example.recovery_router
        assert demo.workflow.edges[2][1][True] is demo.fallback_agent
        assert demo.workflow.edges[2][1][False] is demo.response_agent


def test_run_request_success_path_collects_direct_response_trace():
    with load_example() as module:
        demo = module.LocationRecoveryDemo.build(
            app_name="location-recovery-app",
            user_id="alice",
            session_id="location-session",
        )
        demo.runner.queue_events(
            [
                make_event(module, author="user", text=module.DEFAULT_SUCCESS_REQUEST),
                make_event(
                    module,
                    author="PreciseLocationAgent",
                    function_call=module.types.FunctionCall(
                        name="get_precise_location_info",
                        args={"address": "221B Baker Street, London"},
                    ),
                ),
                make_event(
                    module,
                    author="PreciseLocationAgent",
                    function_response=module.types.FunctionResponse(
                        name="get_precise_location_info",
                        response={
                            "resolution": module.WORKFLOW_PRECISE,
                            "address": "221B Baker Street, London",
                        },
                    ),
                ),
                make_event(
                    module,
                    author="RecoveryRouter",
                    text="Primary lookup succeeded. Route directly to the response agent.",
                    route=False,
                ),
                make_event(
                    module,
                    author="LocationResponseAgent",
                    text=(
                        "The exact address was resolved successfully.\n"
                        "221B Baker Street, London is available as a precise result."
                    ),
                    is_final=True,
                ),
            ]
        )

        result = run(
            demo.run_request(
                module.DEFAULT_SUCCESS_REQUEST,
                scenario_name="Successful Precise Lookup",
                session_index=0,
            )
        )

        assert result.recovery_triggered is False
        assert result.session_id == "location-session-1"
        assert [step.kind for step in result.steps] == [
            "tool-call",
            "tool-result",
            "route",
            "final",
        ]
        assert result.final_response.startswith("The exact address was resolved")
        assert demo.runner.session_service.create_calls == [
            {
                "app_name": "location-recovery-app",
                "user_id": "alice",
                "session_id": "location-session-1",
                "state": module.build_initial_state(module.DEFAULT_SUCCESS_REQUEST),
            }
        ]


def test_run_request_recovery_path_collects_fallback_trace():
    with load_example() as module:
        demo = module.LocationRecoveryDemo.build(
            app_name="location-recovery-app",
            user_id="alice",
            session_id="location-session",
        )
        demo.runner.queue_events(
            [
                make_event(module, author="user", text=module.DEFAULT_RECOVERY_REQUEST),
                make_event(
                    module,
                    author="PreciseLocationAgent",
                    function_call=module.types.FunctionCall(
                        name="get_precise_location_info",
                        args={"address": "404 Missing Street, London"},
                    ),
                ),
                make_event(
                    module,
                    author="PreciseLocationAgent",
                    function_response=module.types.FunctionResponse(
                        name="get_precise_location_info",
                        response={
                            "status": module.WORKFLOW_FALLBACK_REQUIRED,
                            "reason": "No street-level record was found.",
                            "fallback_city": "London",
                        },
                    ),
                ),
                make_event(
                    module,
                    author="RecoveryRouter",
                    text="Primary lookup failed. Route to the fallback agent for London.",
                    route=True,
                ),
                make_event(
                    module,
                    author="FallbackLocationAgent",
                    function_call=module.types.FunctionCall(
                        name="get_general_area_info",
                        args={"city": "London"},
                    ),
                ),
                make_event(
                    module,
                    author="FallbackLocationAgent",
                    function_response=module.types.FunctionResponse(
                        name="get_general_area_info",
                        response={
                            "resolution": module.WORKFLOW_GENERAL,
                            "city": "London",
                        },
                    ),
                ),
                make_event(
                    module,
                    author="LocationResponseAgent",
                    text=(
                        "The exact lookup failed, so the agent recovered with a broader "
                        "city-level result.\n"
                        "London is returned as a fallback area and should be verified "
                        "before operational use."
                    ),
                    is_final=True,
                ),
            ]
        )

        result = run(
            demo.run_request(
                module.DEFAULT_RECOVERY_REQUEST,
                scenario_name="Recovered Fallback Lookup",
                session_index=1,
            )
        )

        assert result.recovery_triggered is True
        assert result.session_id == "location-session-2"
        assert [step.kind for step in result.steps] == [
            "tool-call",
            "tool-result",
            "route",
            "tool-call",
            "tool-result",
            "final",
        ]
        assert "recovered with a broader city-level result" in result.final_response


def test_print_demo_results_displays_both_workflows(capsys):
    with load_example() as module:
        results = [
            module.ScenarioRunResult(
                scenario_name="Successful Precise Lookup",
                request=module.DEFAULT_SUCCESS_REQUEST,
                steps=[
                    module.WorkflowStep(
                        actor="PreciseLocationAgent",
                        kind="tool-call",
                        text='Called `get_precise_location_info` with {"address": "221B Baker Street, London"}.',
                        is_final=False,
                    ),
                    module.WorkflowStep(
                        actor="LocationResponseAgent",
                        kind="final",
                        text="The exact address was resolved successfully.",
                        is_final=True,
                    ),
                ],
                final_response="The exact address was resolved successfully.",
                recovery_triggered=False,
                session_id="location-session-1",
            ),
            module.ScenarioRunResult(
                scenario_name="Recovered Fallback Lookup",
                request=module.DEFAULT_RECOVERY_REQUEST,
                steps=[
                    module.WorkflowStep(
                        actor="RecoveryRouter",
                        kind="route",
                        text="Primary lookup failed. Route to the fallback agent for London.",
                        is_final=False,
                    ),
                    module.WorkflowStep(
                        actor="LocationResponseAgent",
                        kind="final",
                        text="The agent recovered with a city-level result.",
                        is_final=True,
                    ),
                ],
                final_response="The agent recovered with a city-level result.",
                recovery_triggered=True,
                session_id="location-session-2",
            ),
        ]

        module.print_demo_results(results)
        output = capsys.readouterr().out

        assert "# Exception Handling And Recovery Demo" in output
        assert "## Successful Precise Lookup" in output
        assert "## Recovered Fallback Lookup" in output
        assert "Agent Workflow:" in output
        assert "Recovery triggered: yes" in output


def test_validate_runtime_environment_requires_google_api_key(monkeypatch):
    with load_example() as exception_recovery_example:
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        with pytest.raises(ValueError, match="GOOGLE_API_KEY not found"):
            exception_recovery_example.validate_runtime_environment()
