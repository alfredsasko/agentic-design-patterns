from __future__ import annotations

import importlib.util
import pathlib
import sys
import types as builtin_types
from typing import Any


def install_fake_google_adk_modules() -> None:
    google_module = builtin_types.ModuleType("google")
    adk_module = builtin_types.ModuleType("google.adk")
    agents_module = builtin_types.ModuleType("google.adk.agents")
    runners_module = builtin_types.ModuleType("google.adk.runners")
    tools_module = builtin_types.ModuleType("google.adk.tools")
    genai_module = builtin_types.ModuleType("google.genai")
    events_module = builtin_types.ModuleType("google.adk.events")
    invocation_context_module = builtin_types.ModuleType(
        "google.adk.agents.invocation_context"
    )
    workflow_module = builtin_types.ModuleType("google.adk.workflow")

    class FakeFunctionCall:
        def __init__(self, *, name, args=None):
            self.name = name
            self.args = args or {}

    class FakeFunctionResponse:
        def __init__(self, *, name, response=None):
            self.name = name
            self.response = response

    class FakePart:
        def __init__(self, text=None, function_call=None, function_response=None):
            self.text = text
            self.function_call = function_call
            self.function_response = function_response

    class FakeContent:
        def __init__(self, role=None, parts=None):
            self.role = role
            self.parts = parts or []

    class FakeUserContent(FakeContent):
        def __init__(self, parts=None):
            super().__init__(role="user", parts=parts)

    class FakeEventActions:
        def __init__(self, *, escalate=None, route=None, transfer_to_agent=None):
            self.escalate = escalate
            self.route = route
            self.transfer_to_agent = transfer_to_agent

    class FakeEvent:
        def __init__(
            self,
            *,
            author="",
            content=None,
            output=None,
            is_final=False,
            actions=None,
            node_path="",
        ):
            self.author = author
            self.content = content
            self.output = output
            self._is_final = is_final
            self.actions = actions or FakeEventActions()
            self.node_info = builtin_types.SimpleNamespace(path=node_path)

        def is_final_response(self):
            return self._is_final

        def get_function_calls(self):
            calls = []
            for part in getattr(self.content, "parts", []) or []:
                if getattr(part, "function_call", None) is not None:
                    calls.append(part.function_call)
            return calls

        def get_function_responses(self):
            responses = []
            for part in getattr(self.content, "parts", []) or []:
                if getattr(part, "function_response", None) is not None:
                    responses.append(part.function_response)
            return responses

    class FakeBaseAgent:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.sub_agents = list(kwargs.get("sub_agents", []))
            for sub_agent in self.sub_agents:
                sub_agent.parent_agent = self

    class FakeLlmAgent(FakeBaseAgent):
        pass

    class FakeLoopAgent(FakeBaseAgent):
        pass

    class FakeWorkflow(FakeBaseAgent):
        def __init__(self, **kwargs):
            self.edges = kwargs.pop("edges", [])
            super().__init__(**kwargs)

    class FakeAgentTool:
        def __init__(self, agent):
            self.agent = agent
            self.name = agent.name
            self.description = getattr(agent, "description", "")

    class FakeNodeDecorator:
        def __init__(self, name=None):
            self.name = name

        def __call__(self, func):
            func.name = self.name or func.__name__
            return func

    def fake_node(*args, **kwargs):
        if args and callable(args[0]):
            return FakeNodeDecorator(kwargs.get("name"))(args[0])
        return FakeNodeDecorator(kwargs.get("name"))

    class FakeTaskExecutorBase(FakeBaseAgent):
        pass

    class FakeSession:
        def __init__(self, session_id, state=None):
            self.id = session_id
            self.state = dict(state or {})

    class FakeInvocationContext:
        def __init__(self, session=None):
            self.session = session or FakeSession("session-1")
            self.state = self.session.state

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
    tools_module.AgentTool = FakeAgentTool
    tools_module.FunctionTool = builtin_types.SimpleNamespace
    genai_module.types = builtin_types.SimpleNamespace(
        Content=FakeContent,
        Part=FakePart,
        UserContent=FakeUserContent,
        FunctionCall=FakeFunctionCall,
        FunctionResponse=FakeFunctionResponse,
    )
    google_module.adk = adk_module
    google_module.genai = genai_module
    adk_module.agents = agents_module
    adk_module.runners = runners_module
    adk_module.tools = tools_module
    adk_module.events = events_module
    adk_module.workflow = workflow_module
    workflow_module.START = builtin_types.SimpleNamespace(name="START")
    workflow_module.Workflow = FakeWorkflow
    workflow_module.node = fake_node

    sys.modules["google"] = google_module
    sys.modules["google.adk"] = adk_module
    sys.modules["google.adk.agents"] = agents_module
    sys.modules["google.adk.agents.invocation_context"] = invocation_context_module
    sys.modules["google.adk.runners"] = runners_module
    sys.modules["google.adk.tools"] = tools_module
    sys.modules["google.adk.events"] = events_module
    sys.modules["google.adk.workflow"] = workflow_module
    sys.modules["google.genai"] = genai_module


def load_module_with_fake_adk(module_path: pathlib.Path, module_name: str) -> Any:
    install_fake_google_adk_modules()
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
