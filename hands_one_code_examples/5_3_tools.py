from __future__ import annotations

import ast
import asyncio
import math
import operator
import os
from contextlib import aclosing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from dotenv import load_dotenv

try:
    from google.adk.agents import LlmAgent
    from google.adk.runners import InMemoryRunner
    from google.adk.tools import FunctionTool
    from google.genai import types
except ImportError:
    LlmAgent = InMemoryRunner = FunctionTool = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_APP_NAME = "calculator_tool_demo"
DEFAULT_USER_ID = "demo-user"
DEFAULT_SESSION_ID = "calculator-session"
DEFAULT_INSTRUCTION = (
    "You are a precise calculator agent. "
    "Use the calculate_expression tool for arithmetic and math-function requests. "
    "Translate natural language into a valid expression for the tool, such as "
    "'factorial(10)' for a factorial request. Return only the final numeric result."
)

_ALLOWED_BINARY_OPERATORS: Mapping[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARY_OPERATORS: Mapping[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_FUNCTIONS: Mapping[str, Callable[..., Any]] = {
    "abs": abs,
    "factorial": math.factorial,
    "round": round,
    "sqrt": math.sqrt,
}


def load_environment_variables() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")


def require_google_adk() -> None:
    if LlmAgent is None or InMemoryRunner is None or FunctionTool is None or types is None:
        raise ImportError(
            "google-adk is not installed. Install it with `uv add google-adk` "
            "before running this ADK tool-calling example."
        )


def validate_runtime_environment(model: str = DEFAULT_MODEL) -> None:
    if model.startswith("gemini") and not os.getenv("GOOGLE_API_KEY"):
        raise ValueError(
            "GOOGLE_API_KEY not found. Set it before running this ADK tool-calling example."
        )


@dataclass
class MathExpressionService:
    """Safely evaluate a small subset of arithmetic expressions."""

    allowed_functions: Mapping[str, Callable[..., Any]] = field(
        default_factory=lambda: dict(_ALLOWED_FUNCTIONS)
    )
    max_exponent: int = 12

    @staticmethod
    def normalize_expression(expression: str) -> str:
        normalized = " ".join(expression.strip().split())
        return normalized.replace("^", "**")

    def evaluate(self, expression: str) -> int | float:
        normalized_expression = self.normalize_expression(expression)
        if not normalized_expression:
            raise ValueError("expression must not be empty")

        try:
            parsed = ast.parse(normalized_expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid mathematical expression: {expression!r}") from exc

        result = self._evaluate_node(parsed.body)
        if not isinstance(result, (int, float)):
            raise ValueError("expression did not evaluate to a numeric result")
        return result

    def format_result(self, expression: str) -> str:
        result = self.evaluate(expression)
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return format(result, ".15g")

    def _evaluate_node(self, node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value

        if isinstance(node, ast.BinOp):
            operator_type = type(node.op)
            if operator_type not in _ALLOWED_BINARY_OPERATORS:
                raise ValueError(f"Unsupported operator: {operator_type.__name__}")

            left = self._evaluate_node(node.left)
            right = self._evaluate_node(node.right)
            if operator_type is ast.Pow and abs(right) > self.max_exponent:
                raise ValueError("Exponent is too large for this demo calculator")
            return _ALLOWED_BINARY_OPERATORS[operator_type](left, right)

        if isinstance(node, ast.UnaryOp):
            operator_type = type(node.op)
            if operator_type not in _ALLOWED_UNARY_OPERATORS:
                raise ValueError(f"Unsupported unary operator: {operator_type.__name__}")
            operand = self._evaluate_node(node.operand)
            return _ALLOWED_UNARY_OPERATORS[operator_type](operand)

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function_name = node.func.id
            if function_name not in self.allowed_functions:
                raise ValueError(f"Unsupported function: {function_name}")
            if node.keywords:
                raise ValueError("Keyword arguments are not supported")
            arguments = [self._evaluate_node(argument) for argument in node.args]
            return self.allowed_functions[function_name](*arguments)

        raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def calculate_expression(
    expression: str,
    *,
    service: MathExpressionService | None = None,
) -> str:
    """Return the numeric result for a supported arithmetic expression."""

    active_service = service or MathExpressionService()
    return active_service.format_result(expression)


def build_calculation_tool(service: MathExpressionService | None = None) -> Any:
    """Create a stable ADK FunctionTool for calculator queries."""

    require_google_adk()
    active_service = service or MathExpressionService()

    def calculate_expression_tool(expression: str) -> str:
        """
        Evaluate a mathematical expression and return only the numeric result.

        Args:
            expression (str): Arithmetic expression to evaluate, such as
                "(5 + 7) * 3" or "factorial(10)".
        """

        return active_service.format_result(expression)

    return FunctionTool(func=calculate_expression_tool)


def build_calculator_agent(
    *,
    model: str = DEFAULT_MODEL,
    tool: Any | None = None,
    instruction: str = DEFAULT_INSTRUCTION,
) -> Any:
    """Create an ADK agent that solves math tasks through a single tool."""

    require_google_adk()
    active_tool = tool or build_calculation_tool()
    return LlmAgent(
        name="calculator_agent",
        model=model,
        instruction=instruction,
        description="Solves arithmetic queries by calling a deterministic calculator tool.",
        tools=[active_tool],
    )


def build_runner(
    agent: Any | None = None,
    *,
    app_name: str = DEFAULT_APP_NAME,
) -> Any:
    """Create the recommended in-memory runner for development and tests."""

    require_google_adk()
    active_agent = agent or build_calculator_agent()
    return InMemoryRunner(agent=active_agent, app_name=app_name)


async def ensure_session(
    runner: Any,
    *,
    user_id: str = DEFAULT_USER_ID,
    session_id: str = DEFAULT_SESSION_ID,
) -> Any:
    session = await runner.session_service.get_session(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id,
    )
    if session is not None:
        return session

    return await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id,
    )


def derive_session_id(base_session_id: str, query_index: int) -> str:
    if query_index < 0:
        raise ValueError("query_index must not be negative")
    return f"{base_session_id}-{query_index + 1}"


def build_user_message(query: str) -> Any:
    require_google_adk()
    if not query.strip():
        raise ValueError("query must not be empty")

    content_type = getattr(types, "UserContent", types.Content)
    return content_type(parts=[types.Part(text=query)])


def _content_to_text(content: Any) -> str:
    parts = getattr(content, "parts", None) or []
    text_parts: list[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if isinstance(text, str) and text.strip():
            text_parts.append(text.strip())
    return "".join(text_parts).strip()


def extract_final_response_text(events: Sequence[Any]) -> str:
    for event in reversed(list(events)):
        is_final = getattr(event, "is_final_response", None)
        if callable(is_final) and not is_final():
            continue

        content = getattr(event, "content", None)
        text = _content_to_text(content)
        if text:
            return text

    raise ValueError("No final text response was captured from the agent run.")


@dataclass(frozen=True)
class CalculatorToolAssistant:
    """Small facade around an ADK calculator agent and runner."""

    agent: Any
    runner: Any
    tool: Any
    app_name: str = DEFAULT_APP_NAME
    user_id: str = DEFAULT_USER_ID
    session_id: str = DEFAULT_SESSION_ID

    @classmethod
    def build(
        cls,
        *,
        service: MathExpressionService | None = None,
        model: str = DEFAULT_MODEL,
        app_name: str = DEFAULT_APP_NAME,
        user_id: str = DEFAULT_USER_ID,
        session_id: str = DEFAULT_SESSION_ID,
        instruction: str = DEFAULT_INSTRUCTION,
    ) -> "CalculatorToolAssistant":
        tool = build_calculation_tool(service)
        agent = build_calculator_agent(
            model=model,
            tool=tool,
            instruction=instruction,
        )
        runner = build_runner(agent, app_name=app_name)
        return cls(
            agent=agent,
            runner=runner,
            tool=tool,
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )

    async def ask(self, query: str) -> str:
        await ensure_session(
            self.runner,
            user_id=self.user_id,
            session_id=self.session_id,
        )
        user_message = build_user_message(query)
        events: list[Any] = []
        async with aclosing(
            self.runner.run_async(
                user_id=self.user_id,
                session_id=self.session_id,
                new_message=user_message,
            )
        ) as event_stream:
            async for event in event_stream:
                events.append(event)

        return extract_final_response_text(events)


async def run_demo_queries(
    assistant: CalculatorToolAssistant,
    queries: Sequence[str],
) -> list[str]:
    if not queries:
        return []

    async def run_isolated_query(query: str, query_index: int) -> str:
        isolated_assistant = CalculatorToolAssistant(
            agent=assistant.agent,
            runner=assistant.runner,
            tool=assistant.tool,
            app_name=assistant.app_name,
            user_id=assistant.user_id,
            session_id=derive_session_id(assistant.session_id, query_index),
        )
        return await isolated_assistant.ask(query)

    return list(
        await asyncio.gather(
            *(run_isolated_query(query, index) for index, query in enumerate(queries))
        )
    )


async def main() -> None:
    load_environment_variables()
    validate_runtime_environment()
    assistant = CalculatorToolAssistant.build()

    queries = [
        "Calculate the value of (5 + 7) * 3",
        "What is 10 factorial?",
    ]
    responses = await run_demo_queries(assistant, queries)

    for query, response in zip(queries, responses):
        print(f"Query: {query}")
        print(f"Response: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())
