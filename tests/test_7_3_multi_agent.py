from __future__ import annotations

import pathlib
import types as builtin_types

import pytest

from tests._support.fake_adk import load_module_with_fake_adk, make_event, run


MODULE_PATH = pathlib.Path("hands_one_code_examples/7_3_multi_agent.py")

multi_agent_example = load_module_with_fake_adk(
    module_name="multi_agent_loop_example",
    module_path=MODULE_PATH,
    include_workflow=True,
)


def test_evaluate_incident_update_requires_headings_and_word_limit():
    incomplete = multi_agent_example.evaluate_incident_update(
        "## Summary\nA short update.\n\n## Impact\nSome impact.",
        iteration_number=1,
    )

    assert incomplete.is_complete is False
    assert incomplete.missing_headings == ("## Next Actions",)
    assert "Missing headings" in incomplete.feedback

    complete = multi_agent_example.evaluate_incident_update(
        "## Summary\nPayments are degraded.\n\n## Impact\nEU customers cannot complete checkout.\n\n## Next Actions\nEngineering is rolling out a fix and monitoring recovery.",
        iteration_number=2,
    )

    assert complete.is_complete is True
    assert complete.missing_headings == ()
    assert "Stop the loop." in complete.feedback


def test_condition_checker_escalates_when_draft_is_complete():
    checker = multi_agent_example.build_condition_checker()
    session = builtin_types.SimpleNamespace(
        state={
            "iteration_number": 1,
            "current_update": (
                "## Summary\nPayments are degraded.\n\n"
                "## Impact\nEU customers cannot complete checkout.\n\n"
                "## Next Actions\nEngineering is rolling out a fix and monitoring recovery."
            ),
        }
    )
    context = multi_agent_example.InvocationContext(session=session)

    event = checker(context)

    assert event.author == "ConditionChecker"
    assert event.actions.escalate is True
    assert event.actions.route is False
    assert session.state["loop_completed"] is True
    assert "Stop the loop." in event.content.parts[0].text


def test_build_writer_agent_sets_output_key_and_model():
    writer = multi_agent_example.build_writer_agent(model="gemini-2.5-flash")

    assert writer.name == "IncidentUpdateWriter"
    assert writer.model == "gemini-2.5-flash"
    assert writer.output_key == "current_update"
    assert "## Next Actions" in writer.instruction


def test_build_loop_agent_uses_writer_checker_and_max_iterations():
    writer = multi_agent_example.build_writer_agent()
    checker = multi_agent_example.build_condition_checker()

    loop_agent = multi_agent_example.build_loop_agent(
        writer=writer,
        checker=checker,
        max_iterations=10,
    )

    assert loop_agent.name == "IncidentUpdateRefinementWorkflow"
    assert loop_agent.edges[0][0].name == "START"
    assert loop_agent.edges[0][1] is writer
    assert loop_agent.edges[2][0] is checker
    assert loop_agent.edges[2][1][False] is multi_agent_example.workflow_terminator


def test_extract_iteration_records_groups_writer_and_checker_events():
    events = [
        make_event(
            multi_agent_example,
            author="IncidentUpdateWriter",
            text="## Summary\nDraft one.\n\n## Impact\nImpact one.",
        ),
        make_event(
            multi_agent_example,
            author="ConditionChecker",
            text="Iteration 1 review: continue refining.\n- Missing headings: Next Actions.",
            escalate=False,
        ),
        make_event(
            multi_agent_example,
            author="IncidentUpdateWriter",
            text="## Summary\nDraft two.\n\n## Impact\nImpact two.\n\n## Next Actions\nNext actions.",
        ),
        make_event(
            multi_agent_example,
            author="ConditionChecker",
            text="Iteration 2 review: all requirements satisfied. Stop the loop.\n- Word count: 20/120",
            escalate=True,
        ),
    ]

    records = multi_agent_example.extract_iteration_records(events)

    assert len(records) == 2
    assert records[0].iteration_number == 1
    assert records[0].completed is False
    assert records[1].iteration_number == 2
    assert records[1].completed is True
    assert records[1].draft.startswith("## Summary")


def test_incident_update_loop_app_builds_consistent_objects():
    app = multi_agent_example.IncidentUpdateLoopApp.build(
        app_name="loop-app",
        user_id="alice",
        session_id="loop-session",
        max_iterations=10,
    )

    assert app.loop_agent.edges[0][1] is app.writer
    assert app.loop_agent.edges[1][0] is app.writer
    assert app.loop_agent.edges[1][1] is app.checker
    assert app.runner.agent is app.loop_agent
    assert app.max_iterations == 10


def test_run_returns_structured_iteration_history():
    app = multi_agent_example.IncidentUpdateLoopApp.build(
        app_name="loop-app",
        user_id="alice",
        session_id="loop-session",
        max_iterations=10,
    )
    app.runner.queue_events(
        [
            make_event(
                multi_agent_example,
                author="IncidentUpdateWriter",
                text="## Summary\nDraft one.\n\n## Impact\nImpact one.",
            ),
            make_event(
                multi_agent_example,
                author="ConditionChecker",
                text="Iteration 1 review: continue refining.\n- Missing headings: Next Actions.",
                escalate=False,
            ),
            make_event(
                multi_agent_example,
                author="IncidentUpdateWriter",
                text="## Summary\nDraft two.\n\n## Impact\nImpact two.\n\n## Next Actions\nNext actions.",
            ),
            make_event(
                multi_agent_example,
                author="ConditionChecker",
                text="Iteration 2 review: all requirements satisfied. Stop the loop.\n- Word count: 20/120",
                escalate=True,
            ),
        ]
    )

    result = run(app.run("Create an incident update."))

    assert result.request == "Create an incident update."
    assert result.completed is True
    assert len(result.iterations) == 2
    assert result.final_update.endswith("## Next Actions\nNext actions.")
    assert (
        app.runner.session_service.create_calls[0]["state"]["checker_feedback"]
        == multi_agent_example.DEFAULT_INITIAL_FEEDBACK
    )


def test_print_run_result_displays_each_iteration(capsys):
    result = multi_agent_example.LoopRunResult(
        request="Create an incident update.",
        iterations=[
            multi_agent_example.LoopIterationRecord(
                iteration_number=1,
                draft="## Summary\nDraft one.",
                checker_feedback="Iteration 1 review: continue refining.",
                completed=False,
            ),
            multi_agent_example.LoopIterationRecord(
                iteration_number=2,
                draft="## Summary\nDraft two.",
                checker_feedback="Iteration 2 review: all requirements satisfied. Stop the loop.",
                completed=True,
            ),
        ],
        final_update="## Summary\nDraft two.",
        completed=True,
        max_iterations=10,
    )

    multi_agent_example.print_run_result(result)
    output = capsys.readouterr().out

    assert "### Iteration 1" in output
    assert "Checker:" in output
    assert "## Final Update" in output
    assert "Completed: yes (max iterations: 10)" in output


def test_validate_runtime_environment_requires_google_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GOOGLE_API_KEY not found"):
        multi_agent_example.validate_runtime_environment()


def test_validate_runtime_environment_allows_non_gemini_models(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    multi_agent_example.validate_runtime_environment(model="gpt-4o-mini")
