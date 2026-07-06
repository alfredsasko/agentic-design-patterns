import asyncio
import importlib.util
import pathlib
import sys
import types as builtin_types

import pytest


MODULE_PATH = pathlib.Path("hands_one_code_examples/5_3_tools.py")


def _install_fake_google_adk_modules():
    google_module = builtin_types.ModuleType("google")
    adk_module = builtin_types.ModuleType("google.adk")
    agents_module = builtin_types.ModuleType("google.adk.agents")
    runners_module = builtin_types.ModuleType("google.adk.runners")
    tools_module = builtin_types.ModuleType("google.adk.tools")
    genai_module = builtin_types.ModuleType("google.genai")

    class FakeFunctionTool:
        def __init__(self, func):
            self.func = func
            self.name = func.__name__
            self.description = (func.__doc__ or "").strip()

    class FakeLlmAgent:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeSession:
        def __init__(self, session_id):
            self.id = session_id

    class FakeSessionService:
        def __init__(self):
            self.sessions = {}
            self.create_calls = []
            self.get_calls = []

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
        def __init__(self, agent, app_name=None):
            self.agent = agent
            self.app_name = app_name or "InMemoryRunner"
            self.session_service = FakeSessionService()
            self.run_calls = []
            self._event_batches = []

        def queue_events(self, events):
            self._event_batches.append(list(events))

        async def run_async(self, **kwargs):
            self.run_calls.append(kwargs)
            events = self._event_batches.pop(0) if self._event_batches else []
            for event in events:
                yield event

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

    agents_module.LlmAgent = FakeLlmAgent
    runners_module.InMemoryRunner = FakeInMemoryRunner
    tools_module.FunctionTool = FakeFunctionTool
    genai_module.types = builtin_types.SimpleNamespace(
        Content=FakeContent,
        Part=FakePart,
        UserContent=FakeUserContent,
    )

    google_module.adk = adk_module
    google_module.genai = genai_module
    adk_module.agents = agents_module
    adk_module.runners = runners_module
    adk_module.tools = tools_module

    sys.modules["google"] = google_module
    sys.modules["google.adk"] = adk_module
    sys.modules["google.adk.agents"] = agents_module
    sys.modules["google.adk.runners"] = runners_module
    sys.modules["google.adk.tools"] = tools_module
    sys.modules["google.genai"] = genai_module


_install_fake_google_adk_modules()

spec = importlib.util.spec_from_file_location("tools_adk_example", MODULE_PATH)
tools_adk_example = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = tools_adk_example
spec.loader.exec_module(tools_adk_example)


class DummyEvent:
    def __init__(self, *, text="", is_final=True):
        part = builtin_types.SimpleNamespace(text=text)
        self.content = builtin_types.SimpleNamespace(parts=[part])
        self._is_final = is_final

    def is_final_response(self):
        return self._is_final


def _run(coro):
    return asyncio.run(coro)


def test_math_expression_service_evaluates_arithmetic_and_functions():
    service = tools_adk_example.MathExpressionService()

    assert service.format_result("(5 + 7) * 3") == "36"
    assert service.format_result("factorial(5)") == "120"
    assert service.format_result("sqrt(81) / 3") == "3"


def test_math_expression_service_rejects_blank_and_unsafe_inputs():
    service = tools_adk_example.MathExpressionService()

    with pytest.raises(ValueError, match="must not be empty"):
        service.evaluate("   ")

    with pytest.raises(ValueError, match="Unsupported expression element|Unsupported function"):
        service.evaluate("__import__('os').system('pwd')")


def test_calculate_expression_delegates_to_service():
    service = tools_adk_example.MathExpressionService()

    result = tools_adk_example.calculate_expression("2 ** 5", service=service)

    assert result == "32"


def test_build_calculation_tool_exposes_stable_metadata():
    tool = tools_adk_example.build_calculation_tool()

    assert tool.name == "calculate_expression_tool"
    assert "numeric result" in tool.description
    assert tool.func("round(10 / 3, 2)") == "3.33"


def test_build_calculator_agent_uses_single_tool_and_instruction():
    tool = tools_adk_example.build_calculation_tool()

    agent = tools_adk_example.build_calculator_agent(
        model="gemini-2.5-flash",
        tool=tool,
        instruction="Use the tool.",
    )

    assert agent.name == "calculator_agent"
    assert agent.model == "gemini-2.5-flash"
    assert agent.tools == [tool]
    assert agent.instruction == "Use the tool."


def test_build_runner_uses_in_memory_runner_with_app_name():
    agent = tools_adk_example.build_calculator_agent()

    runner = tools_adk_example.build_runner(agent, app_name="math-app")

    assert runner.agent is agent
    assert runner.app_name == "math-app"


def test_ensure_session_creates_once_and_then_reuses():
    runner = tools_adk_example.build_runner()

    session_1 = _run(
        tools_adk_example.ensure_session(
            runner,
            user_id="alice",
            session_id="session-1",
        )
    )
    session_2 = _run(
        tools_adk_example.ensure_session(
            runner,
            user_id="alice",
            session_id="session-1",
        )
    )

    assert session_1 is session_2
    assert len(runner.session_service.create_calls) == 1


def test_derive_session_id_is_stable_and_indexed():
    assert tools_adk_example.derive_session_id("demo", 0) == "demo-1"
    assert tools_adk_example.derive_session_id("demo", 1) == "demo-2"


def test_build_user_message_uses_user_role_and_query_text():
    message = tools_adk_example.build_user_message("What is 2 + 2?")

    assert message.role == "user"
    assert message.parts[0].text == "What is 2 + 2?"


def test_extract_final_response_text_returns_last_final_event_text():
    events = [
        DummyEvent(text="intermediate", is_final=False),
        DummyEvent(text="36", is_final=True),
    ]

    assert tools_adk_example.extract_final_response_text(events) == "36"


def test_calculator_tool_assistant_runs_agent_and_returns_text():
    assistant = tools_adk_example.CalculatorToolAssistant.build(
        app_name="math-app",
        user_id="alice",
        session_id="session-2",
    )
    assistant.runner.queue_events([DummyEvent(text="120", is_final=True)])

    result = _run(assistant.ask("What is 10 factorial?"))

    assert result == "120"
    assert len(assistant.runner.session_service.create_calls) == 1
    assert assistant.runner.run_calls == [
        {
            "user_id": "alice",
            "session_id": "session-2",
            "new_message": assistant.runner.run_calls[0]["new_message"],
        }
    ]
    assert assistant.runner.run_calls[0]["new_message"].parts[0].text == "What is 10 factorial?"


def test_run_demo_queries_preserves_input_order():
    assistant = tools_adk_example.CalculatorToolAssistant.build()
    assistant.runner.queue_events([DummyEvent(text="36", is_final=True)])
    assistant.runner.queue_events([DummyEvent(text="120", is_final=True)])

    result = _run(
        tools_adk_example.run_demo_queries(
            assistant,
            ["(5 + 7) * 3", "factorial(5)"],
        )
    )

    assert result == ["36", "120"]
    assert [call["session_id"] for call in assistant.runner.run_calls] == [
        "calculator-session-1",
        "calculator-session-2",
    ]


def test_validate_runtime_environment_requires_google_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GOOGLE_API_KEY not found"):
        tools_adk_example.validate_runtime_environment()
