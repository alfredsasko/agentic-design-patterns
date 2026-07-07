from __future__ import annotations

import asyncio
import pathlib

import pytest

from tests._support.fake_adk import load_module_with_fake_adk


MODULE_PATH = pathlib.Path("hands_one_code_examples/7_4_multi_agent.py")

multi_agent_example = load_module_with_fake_adk(MODULE_PATH, "multi_agent_tool_example")


def _run(coro):
    return asyncio.run(coro)


def _make_event(module, *, author, text=None, function_call=None, function_response=None, is_final=False):
    parts = []
    if text is not None:
        parts.append(module.types.Part(text=text))
    if function_call is not None:
        parts.append(module.types.Part(function_call=function_call))
    if function_response is not None:
        parts.append(module.types.Part(function_response=function_response))
    return module.Event(
        author=author,
        content=module.types.Content(parts=parts),
        is_final=is_final,
    )


def test_mock_image_studio_returns_deterministic_asset_manifest():
    studio = multi_agent_example.MockImageStudio()

    asset = studio.render_asset("  Cinematic skyline with warm sunrise lighting  ")

    assert asset["status"] == "ready"
    assert asset["asset_id"].startswith("mock-image-")
    assert asset["prompt"] == "Cinematic skyline with warm sunrise lighting"
    assert asset["aspect_ratio"] == "4:5"
    assert asset["preview_url"].startswith("https://example.invalid/assets/")


def test_build_agents_and_agent_tool_wire_dependencies_correctly():
    demo = multi_agent_example.AgentAsToolDemo.build(
        app_name="agent-tool-app",
        user_id="alice",
        session_id="agent-tool-session",
    )

    assert demo.creative_director.name == "CreativeDirector"
    assert demo.illustration_specialist.name == "IllustrationSpecialist"
    assert demo.creative_director.tools[0].name == "IllustrationSpecialist"
    assert (
        demo.illustration_specialist.tools[0].__name__
        == "generate_mock_image_asset_tool"
    )


def test_extract_workflow_steps_captures_tool_calls_results_and_final_response():
    module = multi_agent_example
    function_call = module.types.FunctionCall(
        name="IllustrationSpecialist",
        args={"request": "Design a poster concept."},
    )
    nested_tool_response = module.types.FunctionResponse(
        name="generate_mock_image_asset_tool",
        response={"asset_id": "mock-image-123", "status": "ready"},
    )
    events = [
        _make_event(
            module,
            author="CreativeDirector",
            function_call=function_call,
        ),
        _make_event(
            module,
            author="IllustrationSpecialist",
            function_response=nested_tool_response,
        ),
        _make_event(
            module,
            author="CreativeDirector",
            text=(
                "I delegated the poster concept to the illustration specialist.\n"
                "The asset is ready for review."
            ),
            is_final=True,
        ),
    ]

    steps = module.extract_workflow_steps(events)

    assert [step.kind for step in steps] == ["tool-call", "tool-result", "final"]
    assert "Called `IllustrationSpecialist`" in steps[0].text
    assert "`generate_mock_image_asset_tool` returned" in steps[1].text
    assert steps[-1].is_final is True
    assert module.extract_final_response_text(events).startswith("I delegated")


def test_run_request_collects_steps_and_final_response_from_runner():
    module = multi_agent_example
    demo = module.AgentAsToolDemo.build(
        app_name="agent-tool-app",
        user_id="alice",
        session_id="agent-tool-session",
    )
    demo.runner.queue_events(
        [
            _make_event(
                module,
                author="user",
                text="Create a meetup poster.",
            ),
            _make_event(
                module,
                author="CreativeDirector",
                function_call=module.types.FunctionCall(
                    name="IllustrationSpecialist",
                    args={"request": "Create a meetup poster."},
                ),
            ),
            _make_event(
                module,
                author="IllustrationSpecialist",
                function_response=module.types.FunctionResponse(
                    name="generate_mock_image_asset_tool",
                    response={
                        "asset_id": "mock-image-abc",
                        "status": "ready",
                        "prompt": "A bold meetup poster.",
                    },
                ),
            ),
            _make_event(
                module,
                author="CreativeDirector",
                text=(
                    "I delegated the visual concept to the illustration specialist.\n"
                    "Use asset `mock-image-abc` as the hero image for the campaign."
                ),
                is_final=True,
            ),
        ]
    )

    result = _run(demo.run_request("Create a meetup poster."))

    assert result.request == "Create a meetup poster."
    assert result.final_response.startswith("I delegated the visual concept")
    assert [step.kind for step in result.steps] == ["tool-call", "tool-result", "final"]
    assert demo.runner.run_calls[0]["yield_user_message"] is True
    assert demo.runner.session_service.create_calls == [
        {
            "app_name": "agent-tool-app",
            "user_id": "alice",
            "session_id": "agent-tool-session",
            "state": None,
        }
    ]


def test_print_workflow_result_displays_structured_trace(capsys):
    result = multi_agent_example.AgentToolRunResult(
        request="Create a meetup poster.",
        steps=[
            multi_agent_example.WorkflowStep(
                actor="CreativeDirector",
                kind="tool-call",
                text='Called `IllustrationSpecialist` with {"request": "Create a meetup poster."}.',
                is_final=False,
            ),
            multi_agent_example.WorkflowStep(
                actor="IllustrationSpecialist",
                kind="tool-result",
                text='`generate_mock_image_asset_tool` returned {"asset_id": "mock-image-abc"}.',
                is_final=False,
            ),
            multi_agent_example.WorkflowStep(
                actor="CreativeDirector",
                kind="final",
                text="Use the generated asset for the campaign poster.",
                is_final=True,
            ),
        ],
        final_response="Use the generated asset for the campaign poster.",
    )

    multi_agent_example.print_workflow_result(result)
    output = capsys.readouterr().out

    assert "## Agent-as-Tool Scenario" in output
    assert "1. [CreativeDirector] [tool-call]" in output
    assert "2. [IllustrationSpecialist] [tool-result]" in output
    assert "3. [CreativeDirector] [final] Use the generated asset for the campaign poster." in output
    assert "Final Response:" in output


def test_validate_runtime_environment_requires_google_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GOOGLE_API_KEY not found"):
        multi_agent_example.validate_runtime_environment(model="gemini-2.5-flash")


def test_close_awaits_runner_close():
    demo = multi_agent_example.AgentAsToolDemo.build(app_name="agent-tool-app")

    _run(demo.close())

    assert demo.runner.close_calls == 1
