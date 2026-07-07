from __future__ import annotations

import pathlib

import pytest

from tests._support.fake_adk import (
    collect_async_events,
    load_module_with_fake_adk,
    make_event,
    run,
)


MODULE_PATH = pathlib.Path("hands_one_code_examples/7_2_multi_agent.py")

multi_agent_example = load_module_with_fake_adk(
    module_name="multi_agent_hierarchy_example",
    module_path=MODULE_PATH,
)


def test_build_hierarchical_demo_assigns_parent_child_relationships():
    demo = multi_agent_example.HierarchicalAgentDemo.build(
        model="gemini-2.5-flash",
        app_name="hierarchy-app",
    )

    assert demo.coordinator.sub_agents == [demo.greeter, demo.task_executor]
    assert demo.greeter.parent_agent is demo.coordinator
    assert demo.task_executor.parent_agent is demo.coordinator


def test_task_executor_emits_a_deterministic_completion_event():
    executor = multi_agent_example.build_task_executor()
    context = multi_agent_example.InvocationContext()

    events = run(collect_async_events(executor._run_async_impl(context)))

    assert len(events) == 1
    assert events[0].author == "TaskExecutor"
    assert "Task finished successfully." in events[0].content.parts[0].text


def test_build_user_message_uses_user_role_and_rejects_blank_requests():
    message = multi_agent_example.build_user_message("Say hello to Alex.")

    assert message.role == "user"
    assert message.parts[0].text == "Say hello to Alex."

    with pytest.raises(ValueError, match="request must not be empty"):
        multi_agent_example.build_user_message("   ")


def test_extract_interaction_steps_and_final_response_preserve_order():
    events = [
        make_event(
            multi_agent_example,
            author="Coordinator",
            transfer_to_agent="Greeter",
        ),
        make_event(
            multi_agent_example,
            author="Greeter",
            text="Welcome, Priya.",
            is_final=True,
        ),
    ]

    steps = multi_agent_example.extract_interaction_steps(events)

    assert [step.author for step in steps] == ["Coordinator", "Greeter"]
    assert steps[0].text == "Delegating to Greeter."
    assert steps[-1].is_final is True
    assert multi_agent_example.extract_final_response_text(events) == "Welcome, Priya."


def test_run_scenario_collects_steps_and_final_response_from_runner():
    demo = multi_agent_example.HierarchicalAgentDemo.build(
        app_name="hierarchy-app",
        user_id="alice",
        session_id="hierarchy-session",
    )
    demo.runner.queue_events(
        [
            make_event(
                multi_agent_example,
                author="Coordinator",
                transfer_to_agent="Greeter",
            ),
            make_event(
                multi_agent_example,
                author="Greeter",
                text="Welcome, Priya.",
                is_final=True,
            ),
        ]
    )

    result = run(
        demo.run_scenario(
            scenario_name="Greeting Scenario",
            request="Please greet Priya.",
            session_id="hierarchy-session-1",
        )
    )

    assert result.scenario_name == "Greeting Scenario"
    assert result.final_response == "Welcome, Priya."
    assert [step.author for step in result.steps] == ["Coordinator", "Greeter"]
    assert result.steps[0].text == "Delegating to Greeter."
    assert demo.runner.session_service.create_calls == [
        {
            "app_name": "hierarchy-app",
            "user_id": "alice",
            "session_id": "hierarchy-session-1",
            "state": None,
        }
    ]
    assert demo.runner.run_calls[0]["yield_user_message"] is True


def test_run_demo_requests_executes_both_default_scenarios():
    demo = multi_agent_example.HierarchicalAgentDemo.build(
        app_name="hierarchy-app",
        user_id="alice",
        session_id="hierarchy-session",
    )
    demo.runner.queue_events(
        [
            make_event(
                multi_agent_example,
                author="Coordinator",
                transfer_to_agent="Greeter",
            ),
            make_event(
                multi_agent_example,
                author="Greeter",
                text="Greeting completed.",
                is_final=True,
            ),
        ]
    )
    demo.runner.queue_events(
        [
            make_event(
                multi_agent_example,
                author="Coordinator",
                transfer_to_agent="TaskExecutor",
            ),
            make_event(
                multi_agent_example,
                author="TaskExecutor",
                text="Task execution completed.",
                is_final=True,
            ),
        ]
    )

    results = run(multi_agent_example.run_demo_requests(demo))

    assert [result.scenario_name for result in results] == [
        "Greeting Scenario",
        "Task Scenario",
    ]
    assert [result.final_response for result in results] == [
        "Greeting completed.",
        "Task execution completed.",
    ]
    assert [step.text for step in results[0].steps] == [
        "Delegating to Greeter.",
        "Greeting completed.",
    ]
    assert [step.text for step in results[1].steps] == [
        "Delegating to TaskExecutor.",
        "Task execution completed.",
    ]


def test_print_scenario_result_displays_each_step(capsys):
    result = multi_agent_example.ScenarioResult(
        scenario_name="Task Scenario",
        request="Please perform the deployment checklist task.",
        steps=[
            multi_agent_example.AgentInteractionStep(
                author="Coordinator",
                text="Delegating to TaskExecutor.",
                is_final=False,
            ),
            multi_agent_example.AgentInteractionStep(
                author="TaskExecutor",
                text="Task execution completed.",
                is_final=True,
            ),
        ],
        final_response="Task execution completed.",
    )

    multi_agent_example.print_scenario_result(result)
    output = capsys.readouterr().out

    assert "## Task Scenario" in output
    assert "1. [Coordinator] Delegating to TaskExecutor." in output
    assert "2. [TaskExecutor] [final] Task execution completed." in output
    assert "Final Response:" in output


def test_validate_runtime_environment_requires_google_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GOOGLE_API_KEY not found"):
        multi_agent_example.validate_runtime_environment()


def test_validate_runtime_environment_allows_non_gemini_models(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    multi_agent_example.validate_runtime_environment(model="gpt-4o-mini")


def test_close_awaits_runner_close():
    demo = multi_agent_example.HierarchicalAgentDemo.build(app_name="hierarchy-app")

    run(demo.close())

    assert demo.runner.close_calls == 1
