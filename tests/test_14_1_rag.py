from __future__ import annotations

import pathlib
import sys
import uuid
from contextlib import contextmanager

from tests._support.fake_adk import load_module_with_fake_adk, make_event, run


MODULE_PATH = pathlib.Path("hands_one_code_examples/14_1_rag.py")
FAKE_MODULE_NAMES = (
    "google",
    "google.adk",
    "google.adk.agents",
    "google.adk.apps",
    "google.adk.apps.app",
    "google.adk.agents.invocation_context",
    "google.adk.events",
    "google.adk.runners",
    "google.adk.tools",
    "google.adk.workflow",
    "google.genai",
)


@contextmanager
def load_example():
    module_name = f"rag_example_{uuid.uuid4().hex}"
    saved_modules = {name: sys.modules.get(name) for name in FAKE_MODULE_NAMES}
    saved_module = sys.modules.get(module_name)
    try:
        module = load_module_with_fake_adk(
            module_path=MODULE_PATH,
            module_name=module_name,
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


def test_build_google_search_instruction_mentions_intro_search_and_sources():
    with load_example() as module:
        instruction = module.build_google_search_instruction()

        assert "You are Google Search Bot." in instruction
        assert "Introduce yourself only on the first user-facing turn" in instruction
        assert "Google Search tool" in instruction
        assert "Do not fabricate search results or citations." in instruction


def test_build_google_search_agent_uses_google_search_tool():
    with load_example() as module:
        agent = module.build_google_search_agent(model="gemini-2.5-flash")

        assert agent.name == "google_search_bot"
        assert agent.model == "gemini-2.5-flash"
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "google_search"


def test_normalize_search_response_prepends_intro():
    with load_example() as module:
        ctx = type("Ctx", (), {"state": {}})()
        response = type(
            "Response",
            (),
            {
                "content": module.types.Content(
                    role="model",
                    parts=[module.types.Part(text="I found useful sources.")],
                )
            },
        )()

        normalized = module.normalize_search_response(ctx, response)

        assert normalized is not None
        assert module.content_to_text(normalized.content).startswith(
            "Hello! I am Google Search Bot."
        )
        assert ctx.state[module.STATE_INTRO_SHOWN] is True


def test_normalize_search_response_skips_intro_after_first_turn():
    with load_example() as module:
        ctx = type("Ctx", (), {"state": {module.STATE_INTRO_SHOWN: True}})()
        response = type(
            "Response",
            (),
            {
                "content": module.types.Content(
                    role="model",
                    parts=[
                        module.types.Part(
                            text="Hello! I am Google Search Bot. I found useful sources."
                        )
                    ],
                )
            },
        )()

        normalized = module.normalize_search_response(ctx, response)

        assert normalized is not None
        assert module.content_to_text(normalized.content) == "I found useful sources."


def test_google_search_bot_app_builds_consistent_objects():
    with load_example() as module:
        app = module.GoogleSearchBotApp.build(
            app_name="search-app",
            user_id="alice",
            session_id="search-session",
        )

        assert app.agent.name == "google_search_bot"
        assert app.runner.agent is app.agent
        assert app.runner.app_name == "search-app"


def test_run_request_collects_search_tool_trace_and_final_response():
    with load_example() as module:
        app = module.GoogleSearchBotApp.build(
            app_name="search-app",
            user_id="alice",
            session_id="search-session",
        )
        app.runner.queue_events(
            [
                make_event(module, author="user", text=module.DEFAULT_SEARCH_REQUEST),
                make_event(
                    module,
                    author="google_search_bot",
                    function_call=module.types.FunctionCall(
                        name="google_search",
                        args={"query": "latest Google Agent Development Kit updates"},
                    ),
                ),
                make_event(
                    module,
                    author="google_search_bot",
                    function_response=module.types.FunctionResponse(
                        name="google_search",
                        response={
                            "results": [
                                {
                                    "title": "ADK 2.0",
                                    "url": "https://adk.dev/",
                                }
                            ]
                        },
                    ),
                ),
                make_event(
                    module,
                    author="google_search_bot",
                    text=(
                        "Hello! I am Google Search Bot. I searched Google and found the "
                        "latest ADK updates at adk.dev."
                    ),
                    is_final=True,
                ),
            ]
        )

        result = run(app.run_request(module.DEFAULT_SEARCH_REQUEST, session_index=0))

        assert result.session_id == "search-session-1"
        assert [step.kind for step in result.steps] == ["tool-call", "tool-result", "final"]
        assert result.final_response.startswith("Hello! I am Google Search Bot.")
        assert app.runner.session_service.create_calls == [
            {
                "app_name": "search-app",
                "user_id": "alice",
                "session_id": "search-session-1",
                "state": None,
            }
        ]
