import asyncio
import importlib.util
import pathlib
import sys
import types as builtin_types

import pytest


MODULE_PATH = pathlib.Path("hands_one_code_examples/7_2_multi_agent.py")


def _install_fake_google_adk_modules():
    google_module = builtin_types.ModuleType("google")
    adk_module = builtin_types.ModuleType("google.adk")
    agents_module = builtin_types.ModuleType("google.adk.agents")
    invocation_context_module = builtin_types.ModuleType(
        "google.adk.agents.invocation_context"
    )
    events_module = builtin_types.ModuleType("google.adk.events")
    runners_module = builtin_types.ModuleType("google.adk.runners")
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

    class FakeEvent:
        def __init__(
            self,
            *,
            author="",
            content=None,
            output=None,
            is_final=False,
            actions=None,
        ):
            self.author = author
            self.content = content
            self.output = output
            self._is_final = is_final
            self.actions = actions or builtin_types.SimpleNamespace(
                transfer_to_agent=None
            )

        def is_final_response(self):
            return self._is_final

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

    class FakeLlmAgent(FakeBaseAgent):
        pass

    class FakeInvocationContext:
        pass

    class FakeSession:
        def __init__(self, session_id):
            self.id = session_id

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

        async def create_session(self, *, app_name, user_id, session_id):
            self.create_calls.append(
                {
                    "app_name": app_name,
                    "user_id": user_id,
                    "session_id": session_id,
                }
            )
            session = FakeSession(session_id)
            self.sessions[(app_name, user_id, session_id)] = session
            return session

    class FakeInMemoryRunner:
        def __init__(self, agent=None, app_name=None):
            self.agent = agent
            self.app_name = app_name or "InMemoryRunner"
            self.session_service = FakeSessionService()
            self.run_calls = []
            self._event_batches = []
            self.close_calls = 0

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
    invocation_context_module.InvocationContext = FakeInvocationContext
    events_module.Event = FakeEvent
    runners_module.InMemoryRunner = FakeInMemoryRunner
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
    sys.modules["google.genai"] = genai_module


def load_module_with_fake_adk():
    _install_fake_google_adk_modules()
    spec = importlib.util.spec_from_file_location("multi_agent_hierarchy_example", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


multi_agent_example = load_module_with_fake_adk()


def _run(coro):
    return asyncio.run(coro)


def _make_event(module, *, author, text="", is_final=False, output=None, transfer_to_agent=None):
    content = None
    if text:
        content = module.types.Content(parts=[module.types.Part(text=text)])
    actions = builtin_types.SimpleNamespace(transfer_to_agent=transfer_to_agent)
    return module.Event(
        author=author,
        content=content,
        output=output,
        is_final=is_final,
        actions=actions,
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

    events = _run(_collect_async_events(executor._run_async_impl(context)))

    assert len(events) == 1
    assert events[0].author == "TaskExecutor"
    assert "Task finished successfully." in events[0].content.parts[0].text


async def _collect_async_events(async_iterable):
    results = []
    async for item in async_iterable:
        results.append(item)
    return results


def test_build_user_message_uses_user_role_and_rejects_blank_requests():
    message = multi_agent_example.build_user_message("Say hello to Alex.")

    assert message.role == "user"
    assert message.parts[0].text == "Say hello to Alex."

    with pytest.raises(ValueError, match="request must not be empty"):
        multi_agent_example.build_user_message("   ")


def test_extract_interaction_steps_and_final_response_preserve_order():
    events = [
        _make_event(
            multi_agent_example,
            author="Coordinator",
            transfer_to_agent="Greeter",
        ),
        _make_event(
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
            _make_event(
                multi_agent_example,
                author="Coordinator",
                transfer_to_agent="Greeter",
            ),
            _make_event(multi_agent_example, author="Greeter", text="Welcome, Priya.", is_final=True),
        ]
    )

    result = _run(
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
            _make_event(
                multi_agent_example,
                author="Coordinator",
                transfer_to_agent="Greeter",
            ),
            _make_event(
                multi_agent_example,
                author="Greeter",
                text="Greeting completed.",
                is_final=True,
            ),
        ]
    )
    demo.runner.queue_events(
        [
            _make_event(
                multi_agent_example,
                author="Coordinator",
                transfer_to_agent="TaskExecutor",
            ),
            _make_event(
                multi_agent_example,
                author="TaskExecutor",
                text="Task execution completed.",
                is_final=True,
            ),
        ]
    )

    results = _run(multi_agent_example.run_demo_requests(demo))

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

    _run(demo.close())

    assert demo.runner.close_calls == 1
