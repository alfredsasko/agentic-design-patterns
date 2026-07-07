from __future__ import annotations

import asyncio
import importlib.util
import sys
import types as builtin_types
from pathlib import Path
from typing import Any


def install_fake_google_adk(*, include_workflow: bool = False) -> None:
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
        def __init__(self, text: str | None = None):
            self.text = text

    class FakeContent:
        def __init__(self, role: str | None = None, parts: list[Any] | None = None):
            self.role = role
            self.parts = parts or []

    class FakeUserContent(FakeContent):
        def __init__(self, parts: list[Any] | None = None):
            super().__init__(role="user", parts=parts)

    class FakeEventActions:
        def __init__(
            self,
            *,
            transfer_to_agent: str | None = None,
            escalate: bool | None = None,
            route: Any = None,
        ):
            self.transfer_to_agent = transfer_to_agent
            self.escalate = escalate
            self.route = route

    class FakeEvent:
        def __init__(
            self,
            *,
            author: str = "",
            content: Any = None,
            output: Any = None,
            is_final: bool = False,
            actions: Any = None,
        ):
            self.author = author
            self.content = content
            self.output = output
            self._is_final = is_final
            self.actions = actions or FakeEventActions()

        def is_final_response(self) -> bool:
            return self._is_final

    class FakeBaseAgent:
        def __init__(self, **kwargs: Any):
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
        def __init__(self, **kwargs: Any):
            self.edges = kwargs.pop("edges", [])
            super().__init__(**kwargs)

    class FakeSession:
        def __init__(self, session_id: str, state: dict[str, Any] | None = None):
            self.id = session_id
            self.state = dict(state or {})

    class FakeInvocationContext:
        def __init__(self, session: Any | None = None):
            self.session = session or FakeSession("session-1", {})

    class FakeSessionService:
        def __init__(self):
            self.sessions: dict[tuple[str, str, str], Any] = {}
            self.get_calls: list[dict[str, Any]] = []
            self.create_calls: list[dict[str, Any]] = []

        async def get_session(self, *, app_name: str, user_id: str, session_id: str):
            self.get_calls.append(
                {
                    "app_name": app_name,
                    "user_id": user_id,
                    "session_id": session_id,
                }
            )
            return self.sessions.get((app_name, user_id, session_id))

        async def create_session(
            self,
            *,
            app_name: str,
            user_id: str,
            session_id: str,
            state: dict[str, Any] | None = None,
        ):
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
        def __init__(self, agent: Any = None, app_name: str | None = None):
            self.agent = agent
            self.app_name = app_name or "InMemoryRunner"
            self.session_service = FakeSessionService()
            self.run_calls: list[dict[str, Any]] = []
            self.close_calls = 0
            self._event_batches: list[list[Any]] = []

        def queue_events(self, events: list[Any]) -> None:
            self._event_batches.append(list(events))

        async def run_async(self, **kwargs: Any):
            self.run_calls.append(kwargs)
            events = self._event_batches.pop(0) if self._event_batches else []
            for event in events:
                yield event

        async def close(self) -> None:
            self.close_calls += 1

    def fake_node(*args: Any, **kwargs: Any):
        def decorator(func: Any) -> Any:
            func.name = kwargs.get("name", getattr(func, "name", func.__name__))
            return func

        if args and callable(args[0]):
            return decorator(args[0])
        return decorator

    agents_module.BaseAgent = FakeBaseAgent
    agents_module.LlmAgent = FakeLlmAgent
    agents_module.LoopAgent = FakeLoopAgent
    invocation_context_module.InvocationContext = FakeInvocationContext
    events_module.Event = FakeEvent
    events_module.EventActions = FakeEventActions
    runners_module.InMemoryRunner = FakeInMemoryRunner
    genai_module.types = builtin_types.SimpleNamespace(
        Content=FakeContent,
        Part=FakePart,
        UserContent=FakeUserContent,
    )

    if include_workflow:
        workflow_module.START = builtin_types.SimpleNamespace(name="START")
        workflow_module.Workflow = FakeWorkflow
        workflow_module.node = fake_node

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
    if include_workflow:
        sys.modules["google.adk.workflow"] = workflow_module
    sys.modules["google.genai"] = genai_module


def load_module_with_fake_adk(
    *,
    module_name: str,
    module_path: str | Path,
    include_workflow: bool = False,
):
    install_fake_google_adk(include_workflow=include_workflow)
    spec = importlib.util.spec_from_file_location(module_name, Path(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(coro):
    return asyncio.run(coro)


async def collect_async_events(async_iterable):
    results = []
    async for item in async_iterable:
        results.append(item)
    return results


def make_event(
    module: Any,
    *,
    author: str,
    text: str = "",
    is_final: bool = False,
    output: Any = None,
    transfer_to_agent: str | None = None,
    escalate: bool | None = None,
    route: Any = None,
):
    content = None
    if text:
        content = module.types.Content(parts=[module.types.Part(text=text)])
    event_actions_type = getattr(module, "EventActions", None)
    if event_actions_type is None:
        actions = builtin_types.SimpleNamespace(
            transfer_to_agent=transfer_to_agent,
            escalate=escalate,
            route=route,
        )
    else:
        actions = event_actions_type(
            transfer_to_agent=transfer_to_agent,
            escalate=escalate,
            route=route,
        )
    return module.Event(
        author=author,
        content=content,
        output=output,
        is_final=is_final,
        actions=actions,
    )
