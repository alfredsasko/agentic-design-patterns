from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv
from pydantic import BaseModel, Field, PrivateAttr

try:
    from crewai import Agent, Crew, Process, Task
    from crewai.tools import BaseTool
except ImportError:
    Agent = Crew = Process = Task = None  # type: ignore[assignment]
    BaseTool = object  # type: ignore[assignment,misc]


logger = logging.getLogger(__name__)

DEFAULT_PRICES = {
    "AAPL": 178.15,
    "GOOGL": 1750.30,
    "MSFT": 425.50,
}


def load_environment_variables() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")


def require_crewai() -> None:
    if Agent is None or Crew is None or Process is None or Task is None:
        raise ImportError(
            "crewai is not installed. Install it with `uv add crewai` or "
            "`pip install crewai 'crewai[tools]'` before running this example."
        )


class StockPriceInput(BaseModel):
    """Schema exposed to CrewAI for tool calling."""

    ticker: str = Field(
        min_length=1,
        description="The stock ticker symbol to look up, for example AAPL.",
    )


@dataclass
class StockPriceService:
    """Owns the simulated price store and domain lookup rules."""

    prices: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_PRICES))

    def __post_init__(self) -> None:
        self.prices = {
            self.normalize_ticker(ticker): price for ticker, price in dict(self.prices).items()
        }

    @staticmethod
    def normalize_ticker(ticker: str) -> str:
        return ticker.strip().upper()

    def get_price(self, ticker: str) -> float:
        normalized_ticker = self.normalize_ticker(ticker)
        if not normalized_ticker:
            raise ValueError("ticker must not be empty")

        try:
            return self.prices[normalized_ticker]
        except KeyError as exc:
            raise ValueError(
                f"Simulated price for ticker '{normalized_ticker}' was not found."
            ) from exc

    def format_price_answer(self, ticker: str) -> str:
        normalized_ticker = self.normalize_ticker(ticker)
        price = self.get_price(normalized_ticker)
        return f"The simulated stock price for {normalized_ticker} is ${price:.2f}."


def lookup_stock_price(
    ticker: str,
    *,
    service: StockPriceService | None = None,
) -> float:
    """Procedural helper that returns raw stock data."""

    active_service = service or StockPriceService()
    return active_service.get_price(ticker)


def should_cache_tool_result(arguments: dict[str, Any], result: str) -> bool:
    return bool(arguments.get("ticker")) and bool(result)


class StockPriceLookupTool(BaseTool):
    """
    CrewAI tool implemented as a BaseTool subclass.

    This follows the CrewAI custom tool pattern when you need a stable schema,
    private state, and explicit control over tool behavior.
    """

    name: str = "stock_price_lookup"
    description: str = (
        "Return the simulated current stock price for a given ticker symbol. "
        "Use this when the task asks for the latest demo stock price."
    )
    args_schema: type[BaseModel] = StockPriceInput

    _service: StockPriceService = PrivateAttr()

    def __init__(
        self,
        service: StockPriceService | None = None,
        *,
        result_as_answer: bool = True,
        **kwargs: Any,
    ) -> None:
        require_crewai()
        super().__init__(result_as_answer=result_as_answer, **kwargs)
        self._service = service or StockPriceService()
        self.cache_function = should_cache_tool_result

    def _run(self, ticker: str) -> str:
        logger.info("Tool call: stock_price_lookup ticker=%s", ticker)
        return self._service.format_price_answer(ticker)


def build_stock_price_tool(
    service: StockPriceService | None = None,
    *,
    result_as_answer: bool = True,
) -> BaseTool:
    """Build the stock price tool with direct result handoff enabled by default."""

    require_crewai()
    return StockPriceLookupTool(
        service=service,
        result_as_answer=result_as_answer,
    )


def build_financial_analyst_agent(
    *,
    tools: Sequence[BaseTool] | None = None,
    llm: Any = None,
    function_calling_llm: Any = None,
    verbose: bool = False,
    max_iter: int = 5,
) -> Any:
    """Create a focused CrewAI agent with constrained tool access."""

    require_crewai()
    active_tools = list(tools or [build_stock_price_tool()])
    return Agent(
        role="Senior Financial Analyst",
        goal="Use the available tools to return accurate simulated stock prices.",
        backstory=(
            "You are a precise financial analyst who answers with the tool output and "
            "does not invent prices."
        ),
        llm=llm,
        function_calling_llm=function_calling_llm,
        verbose=verbose,
        allow_delegation=False,
        max_iter=max_iter,
        tools=active_tools,
    )


def build_stock_price_task(
    *,
    ticker: str = "AAPL",
    company_name: str = "Apple",
    agent: Any | None = None,
    tools: Sequence[BaseTool] | None = None,
) -> Any:
    """Create a task with a single clear tool-calling objective."""

    require_crewai()
    active_tools = list(tools or [build_stock_price_tool()])
    active_agent = agent or build_financial_analyst_agent(tools=active_tools)
    normalized_ticker = StockPriceService.normalize_ticker(ticker)
    return Task(
        description=(
            f"Return the current simulated stock price for {company_name} "
            f"(ticker: {normalized_ticker}). Use only the stock_price_lookup tool. "
            "If the lookup fails, clearly state that the simulated price is unavailable."
        ),
        expected_output=(
            "A single sentence with the simulated stock price, or a single sentence "
            "stating that the simulated price is unavailable."
        ),
        agent=active_agent,
        tools=active_tools,
    )


def build_financial_crew(
    *,
    agent: Any | None = None,
    task: Any | None = None,
    tools: Sequence[BaseTool] | None = None,
    verbose: bool = False,
) -> Any:
    """Create a sequential crew for the single-tool stock price workflow."""

    require_crewai()
    active_tools = list(tools or [build_stock_price_tool()])
    active_agent = agent or build_financial_analyst_agent(
        tools=active_tools,
        verbose=verbose,
    )
    active_task = task or build_stock_price_task(
        agent=active_agent,
        tools=active_tools,
    )
    return Crew(
        agents=[active_agent],
        tasks=[active_task],
        process=Process.sequential,
        cache=True,
        verbose=verbose,
    )


def extract_crew_result_text(result: Any) -> str:
    if isinstance(result, str):
        return result

    raw = getattr(result, "raw", None)
    if raw is not None:
        return str(raw)

    return str(result)


@dataclass(frozen=True)
class StockPriceCrewApp:
    """Small facade around a CrewAI crew configured for stock lookups."""

    crew: Any
    agent: Any
    task: Any
    tool: BaseTool

    @classmethod
    def build(
        cls,
        *,
        ticker: str = "AAPL",
        company_name: str = "Apple",
        service: StockPriceService | None = None,
        llm: Any = None,
        function_calling_llm: Any = None,
        verbose: bool = False,
    ) -> "StockPriceCrewApp":
        tool = build_stock_price_tool(service, result_as_answer=True)
        agent = build_financial_analyst_agent(
            tools=[tool],
            llm=llm,
            function_calling_llm=function_calling_llm,
            verbose=verbose,
        )
        task = build_stock_price_task(
            ticker=ticker,
            company_name=company_name,
            agent=agent,
            tools=[tool],
        )
        crew = build_financial_crew(
            agent=agent,
            task=task,
            tools=[tool],
            verbose=verbose,
        )
        return cls(crew=crew, agent=agent, task=task, tool=tool)

    def run(self) -> str:
        result = self.crew.kickoff()
        return extract_crew_result_text(result)


def main() -> None:
    load_environment_variables()
    app = StockPriceCrewApp.build()
    print(app.run())


if __name__ == "__main__":
    main()
