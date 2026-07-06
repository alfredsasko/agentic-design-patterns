import importlib.util
import pathlib
import sys
import types

import pytest


MODULE_PATH = pathlib.Path("hands_one_code_examples/5_2_tools.py")


class FakeBaseTool:
    def __init__(self, **kwargs):
        self.cache_function = None
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeAgent:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeTask:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeCrew:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.kickoff_calls = 0

    def kickoff(self):
        self.kickoff_calls += 1
        return types.SimpleNamespace(raw="The simulated stock price for AAPL is $178.15.")


class FakeProcess:
    sequential = "sequential"


def load_module_with_fake_crewai():
    fake_crewai = types.ModuleType("crewai")
    fake_crewai.Agent = FakeAgent
    fake_crewai.Task = FakeTask
    fake_crewai.Crew = FakeCrew
    fake_crewai.Process = FakeProcess

    fake_tools = types.ModuleType("crewai.tools")
    fake_tools.BaseTool = FakeBaseTool

    sys.modules["crewai"] = fake_crewai
    sys.modules["crewai.tools"] = fake_tools

    spec = importlib.util.spec_from_file_location("tools_crewai_example", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tools_crewai_example = load_module_with_fake_crewai()


def test_lookup_stock_price_returns_raw_float():
    service = tools_crewai_example.StockPriceService()

    result = tools_crewai_example.lookup_stock_price(" aapl ", service=service)

    assert result == 178.15


def test_stock_price_service_rejects_blank_and_unknown_tickers():
    service = tools_crewai_example.StockPriceService()

    with pytest.raises(ValueError, match="must not be empty"):
        service.get_price("   ")

    with pytest.raises(ValueError, match="was not found"):
        service.get_price("NVDA")


def test_stock_price_service_formats_answer_with_normalized_ticker():
    service = tools_crewai_example.StockPriceService()

    result = service.format_price_answer(" msft ")

    assert result == "The simulated stock price for MSFT is $425.50."


def test_build_stock_price_tool_exposes_crewai_metadata_and_cache_strategy():
    tool = tools_crewai_example.build_stock_price_tool()

    assert tool.name == "stock_price_lookup"
    assert "simulated current stock price" in tool.description
    assert tool.args_schema is tools_crewai_example.StockPriceInput
    assert tool.result_as_answer is True
    assert tool.cache_function({"ticker": "AAPL"}, "The simulated stock price for AAPL is $178.15.")
    assert tool._run("googl") == "The simulated stock price for GOOGL is $1750.30."


def test_build_financial_analyst_agent_uses_tool_and_safe_defaults():
    tool = tools_crewai_example.build_stock_price_tool()

    agent = tools_crewai_example.build_financial_analyst_agent(
        tools=[tool],
        llm="fake-llm",
        function_calling_llm="fake-tool-llm",
        verbose=True,
        max_iter=3,
    )

    assert agent.role == "Senior Financial Analyst"
    assert agent.tools == [tool]
    assert agent.llm == "fake-llm"
    assert agent.function_calling_llm == "fake-tool-llm"
    assert agent.allow_delegation is False
    assert agent.max_iter == 3
    assert agent.verbose is True


def test_build_stock_price_task_limits_tooling_and_normalizes_ticker():
    tool = tools_crewai_example.build_stock_price_tool()
    agent = tools_crewai_example.build_financial_analyst_agent(tools=[tool])

    task = tools_crewai_example.build_stock_price_task(
        ticker=" aapl ",
        company_name="Apple",
        agent=agent,
        tools=[tool],
    )

    assert "ticker: AAPL" in task.description
    assert "single sentence" in task.expected_output
    assert task.agent is agent
    assert task.tools == [tool]


def test_build_financial_crew_uses_sequential_process_and_cache():
    tool = tools_crewai_example.build_stock_price_tool()
    agent = tools_crewai_example.build_financial_analyst_agent(tools=[tool])
    task = tools_crewai_example.build_stock_price_task(agent=agent, tools=[tool])

    crew = tools_crewai_example.build_financial_crew(
        agent=agent,
        task=task,
        tools=[tool],
        verbose=True,
    )

    assert crew.agents == [agent]
    assert crew.tasks == [task]
    assert crew.process == FakeProcess.sequential
    assert crew.cache is True
    assert crew.verbose is True


def test_extract_crew_result_text_supports_strings_and_raw_objects():
    raw_result = types.SimpleNamespace(raw="raw answer")

    assert tools_crewai_example.extract_crew_result_text("plain text") == "plain text"
    assert tools_crewai_example.extract_crew_result_text(raw_result) == "raw answer"


def test_stock_price_crew_app_builds_consistent_objects_and_runs():
    app = tools_crewai_example.StockPriceCrewApp.build(
        ticker="msft",
        company_name="Microsoft",
        verbose=True,
    )

    result = app.run()

    assert app.tool.result_as_answer is True
    assert app.agent.tools == [app.tool]
    assert "ticker: MSFT" in app.task.description
    assert app.crew.tasks == [app.task]
    assert result == "The simulated stock price for AAPL is $178.15."
    assert app.crew.kickoff_calls == 1


def test_require_crewai_raises_helpful_error_when_dependency_is_missing(monkeypatch):
    monkeypatch.setattr(tools_crewai_example, "Agent", None)
    monkeypatch.setattr(tools_crewai_example, "Crew", None)
    monkeypatch.setattr(tools_crewai_example, "Task", None)
    monkeypatch.setattr(tools_crewai_example, "Process", None)

    with pytest.raises(ImportError, match="crewai is not installed"):
        tools_crewai_example.require_crewai()
