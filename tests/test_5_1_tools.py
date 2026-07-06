import asyncio
import importlib.util
import pathlib
import sys

import pytest
from langchain_core.messages import AIMessage


MODULE_PATH = pathlib.Path("hands_one_code_examples/5_1_tools.py")

spec = importlib.util.spec_from_file_location("tools_langchain_example", MODULE_PATH)
tools_langchain_example = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = tools_langchain_example
spec.loader.exec_module(tools_langchain_example)


class FakeAgent:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def ainvoke(self, payload):
        self.calls.append(payload)
        return self.response


def _run(coro):
    return asyncio.run(coro)


def test_lookup_information_normalizes_queries():
    service = tools_langchain_example.SearchInformationService()

    result = tools_langchain_example.lookup_information(
        "  What is not used here?  ",
        service=service,
    )
    missing = tools_langchain_example.lookup_information(
        "  What is not used here?  ",
        service=service,
    )
    known = tools_langchain_example.lookup_information(
        " Capital of France? ",
        service=service,
    )

    assert result == missing
    assert known == "The capital of France is Paris."


def test_search_service_rejects_blank_queries():
    service = tools_langchain_example.SearchInformationService()

    with pytest.raises(ValueError, match="must not be empty"):
        service.search("   ")


def test_build_search_tool_exposes_stable_langchain_metadata():
    tool = tools_langchain_example.build_search_tool()
    schema = tool.args_schema.model_json_schema()

    assert tool.name == "search_information"
    assert "demo knowledge base" in tool.description
    assert schema["properties"]["query"]["description"] == (
        "The factual question or lookup phrase to search for."
    )
    assert tool.invoke({"query": "population of earth"}) == (
        "The estimated population of Earth is around 8 billion people."
    )


def test_create_chat_model_requires_google_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GOOGLE_API_KEY not found"):
        tools_langchain_example.create_chat_model()


def test_create_tool_calling_agent_app_uses_current_langchain_entrypoint(monkeypatch):
    captured = {}
    fake_tool = tools_langchain_example.build_search_tool()

    def fake_create_chat_model(model, temperature=0.0):
        captured["model_input"] = model
        captured["temperature"] = temperature
        return "fake-model-instance"

    def fake_create_agent(*, model, tools, system_prompt, debug):
        captured["model"] = model
        captured["tools"] = tools
        captured["system_prompt"] = system_prompt
        captured["debug"] = debug
        return "fake-agent-app"

    monkeypatch.setattr(tools_langchain_example, "create_chat_model", fake_create_chat_model)
    monkeypatch.setattr(tools_langchain_example, "create_agent", fake_create_agent)

    result = tools_langchain_example.create_tool_calling_agent_app(
        model="google_genai:gemini-2.0-flash",
        tools=[fake_tool],
        system_prompt="Use tools carefully.",
        debug=True,
    )

    assert result == "fake-agent-app"
    assert captured["model_input"] == "google_genai:gemini-2.0-flash"
    assert captured["model"] == "fake-model-instance"
    assert captured["tools"] == [fake_tool]
    assert captured["system_prompt"] == "Use tools carefully."
    assert captured["debug"] is True


def test_extract_final_text_supports_langchain_messages_and_content_blocks():
    message_result = {"messages": [AIMessage(content="Paris")]}
    block_result = {"messages": [{"content": [{"type": "text", "text": "Cloudy in London"}]}]}

    assert tools_langchain_example.extract_final_text(message_result) == "Paris"
    assert tools_langchain_example.extract_final_text(block_result) == "Cloudy in London"


def test_tool_calling_assistant_sends_message_payload_and_returns_text():
    agent = FakeAgent({"messages": [AIMessage(content="Paris")]})
    assistant = tools_langchain_example.ToolCallingAssistant(
        agent=agent,
        tools=(tools_langchain_example.build_search_tool(),),
    )

    result = _run(assistant.ask("What is the capital of France?"))

    assert result == "Paris"
    assert agent.calls == [
        {"messages": [{"role": "user", "content": "What is the capital of France?"}]}
    ]


def test_run_demo_queries_collects_responses_in_input_order():
    class SequentialAssistant:
        def __init__(self):
            self.calls = []

        async def ask(self, query):
            self.calls.append(query)
            await asyncio.sleep(0)
            return query.upper()

    assistant = SequentialAssistant()

    responses = _run(
        tools_langchain_example.run_demo_queries(
            assistant,
            ["capital of france", "weather in london"],
        )
    )

    assert responses == ["CAPITAL OF FRANCE", "WEATHER IN LONDON"]
    assert assistant.calls == ["capital of france", "weather in london"]
