import asyncio
import importlib.util
import pathlib
import sys
import types as builtin_types

import pytest


MODULE_PATH = pathlib.Path("hands_one_code_examples/7_3_multi_agent.py")


def _install_fake_google_adk_modules():
    google_module = builtin_types.ModuleType("google")
    adk_module = builtin_types.ModuleType("google.adk")
    agents_module = builtin_types.ModuleType("google.adk.agents")
    invocation_context_module = builtin_types.ModuleType(
        "google.adk.agents.invocation_context"
    )
    events_module = builtin_types.ModuleType("google.adk.events")
    runners_module = builtin_types.ModuleType("google.adk.runners")
    workflow_module = builtin_types.ModuleType("google.adk.workflow")
    genai_module = builtin_types.ModuleType("google.genai")

    class FakePart:
        def __init__(self, text=None):
            self.text = text

    class FakeContent:
        def __init__(self, role=None, parts=None):
            self.role = role
            self.parts = parts or []

    class FakeUserContent(FakeContent):
        def __init__(self, parts=None):
            super().__init__(role="user", parts=parts)

    class FakeEventActions:
        def __init__(self, *, escalate=None, route=None):
            self.escalate = escalate
            self.route = route

    class FakeEvent:
        def __init__(self, *, author="", content=None, actions=None, output=None):
            self.author = author
            self.content = content
            self.actions = actions or FakeEventActions(escalate=False)
            self.output = output

        def is_final_response(self):
            return False

    class FakeBaseAgent:
        def __init__(self, **kwargs):
            self.name = kwargs.pop("name", getattr(self, "name", ""))
            self.description = kwargs.pop(
                "description",
                getattr(self, "description", ""),
            )
            self.parent_agent = kwargs.pop("parent_agent", None)
            self.sub_agents = list(kwargs.pop("sub_agents", []))
            for key, value in kwargs.items():
                setattr(self, key, value)
            for sub_agent in self.sub_agents:
                sub_agent.parent_agent = self

    class FakeLoopAgent(FakeBaseAgent):
        pass

    class FakeLlmAgent(FakeBaseAgent):
        pass

    class FakeWorkflow(FakeBaseAgent):
        def __init__(self, **kwargs):
            self.edges = kwargs.pop("edges", [])
            super().__init__(**kwargs)

    class FakeSession:
        def __init__(self, session_id, state=None):
            self.id = session_id
            self.state = dict(state or {})

    class FakeInvocationContext:
        def __init__(self, session=None):
            self.session = session or FakeSession("session-1", {})

    class FakeSessionService:
        def __init__(self):
            self.sessions = {}
            self.get_calls = []
            self.create_calls = []

        async def get_session(self, *, app_name, user_id, session_id):
            self.get_calls.append(
                {
                    "app_name": app_name,
                    "user_id": user_id,
                    "session_id": session_id,
                }
            )
            return self.sessions.get((app_name, user_id, session_id))

        async def create_session(self, *, app_name, user_id, session_id, state=None):
            self.create_calls.append(
                {
                    "app_name": app_name,
                    "user_id": user_id,
                    "session_id": session_id,
                    "state": state,
                }
            )
            session = FakeSession(session_id, state=state)
            self.sessions[(app_name, user_id, session_id)] = session
            return session

    class FakeInMemoryRunner:
        def __init__(self, agent=None, app_name=None):
            self.agent = agent
            self.app_name = app_name or "InMemoryRunner"
            self.session_service = FakeSessionService()
            self.run_calls = []
            self.close_calls = 0
            self._event_batches = []

        def queue_events(self, events):
            self._event_batches.append(list(events))

        async def run_async(self, **kwargs):
            self.run_calls.append(kwargs)
            events = self._event_batches.pop(0) if self._event_batches else []
            for event in events:
                yield event

        async def close(self):
            self.close_calls += 1

    agents_module.BaseAgent = FakeBaseAgent
    agents_module.LlmAgent = FakeLlmAgent
    agents_module.LoopAgent = FakeLoopAgent
    invocation_context_module.InvocationContext = FakeInvocationContext
    events_module.Event = FakeEvent
    events_module.EventActions = FakeEventActions
    runners_module.InMemoryRunner = FakeInMemoryRunner
    def fake_node(*args, **kwargs):
        def decorator(func):
            func.name = kwargs.get("name", getattr(func, "name", func.__name__))
            return func

        if args and callable(args[0]):
            return decorator(args[0])
        return decorator

    workflow_module.START = builtin_types.SimpleNamespace(name="START")
    workflow_module.Workflow = FakeWorkflow
    workflow_module.node = fake_node
    genai_module.types = builtin_types.SimpleNamespace(
        Content=FakeContent,
        Part=FakePart,
        UserContent=FakeUserContent,
    )

    google_module.adk = adk_module
    google_module.genai = genai_module
    adk_module.agents = agents_module
    adk_module.events = events_module
    adk_module.runners = runners_module

    sys.modules["google"] = google_module
    sys.modules["google.adk"] = adk_module
    sys.modules["google.adk.agents"] = agents_module
    sys.modules["google.adk.agents.invocation_context"] = invocation_context_module
    sys.modules["google.adk.events"] = events_module
    sys.modules["google.adk.runners"] = runners_module
    sys.modules["google.adk.workflow"] = workflow_module
    sys.modules["google.genai"] = genai_module


def load_module_with_fake_adk():
    _install_fake_google_adk_modules()
    spec = importlib.util.spec_from_file_location("multi_agent_loop_example", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


multi_agent_example = load_module_with_fake_adk()


def _run(coro):
    return asyncio.run(coro)


async def _collect_async_events(async_iterable):
    results = []
    async for item in async_iterable:
        results.append(item)
    return results


def _make_text_event(module, *, author, text, escalate=False):
    return module.Event(
        author=author,
        content=module.types.Content(parts=[module.types.Part(text=text)]),
        actions=module.EventActions(escalate=escalate),
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
    assert loop_agent.edges[2][1][True] is multi_agent_example.workflow_terminator


def test_extract_iteration_records_groups_writer_and_checker_events():
    events = [
        _make_text_event(
            multi_agent_example,
            author="IncidentUpdateWriter",
            text="## Summary\nDraft one.\n\n## Impact\nImpact one.",
        ),
        _make_text_event(
            multi_agent_example,
            author="ConditionChecker",
            text="Iteration 1 review: continue refining.\n- Missing headings: Next Actions.",
            escalate=False,
        ),
        _make_text_event(
            multi_agent_example,
            author="IncidentUpdateWriter",
            text="## Summary\nDraft two.\n\n## Impact\nImpact two.\n\n## Next Actions\nNext actions.",
        ),
        _make_text_event(
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
            _make_text_event(
                multi_agent_example,
                author="IncidentUpdateWriter",
                text="## Summary\nDraft one.\n\n## Impact\nImpact one.",
            ),
            _make_text_event(
                multi_agent_example,
                author="ConditionChecker",
                text="Iteration 1 review: continue refining.\n- Missing headings: Next Actions.",
                escalate=False,
            ),
            _make_text_event(
                multi_agent_example,
                author="IncidentUpdateWriter",
                text="## Summary\nDraft two.\n\n## Impact\nImpact two.\n\n## Next Actions\nNext actions.",
            ),
            _make_text_event(
                multi_agent_example,
                author="ConditionChecker",
                text="Iteration 2 review: all requirements satisfied. Stop the loop.\n- Word count: 20/120",
                escalate=True,
            ),
        ]
    )

    result = _run(app.run("Create an incident update."))

    assert result.request == "Create an incident update."
    assert result.completed is True
    assert len(result.iterations) == 2
    assert result.final_update.endswith("## Next Actions\nNext actions.")
    assert app.runner.session_service.create_calls[0]["state"]["checker_feedback"] == multi_agent_example.DEFAULT_INITIAL_FEEDBACK


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
