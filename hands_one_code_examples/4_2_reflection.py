import asyncio
import ast
import inspect
import json
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.workflow import START, Workflow
from google.genai import types


DEFAULT_MAX_REVIEW_ITERATIONS = 3
JUNIOR_MODEL = "gemini-2.5-flash"
SENIOR_MODEL = "gemini-2.5-pro"
NO_PREVIOUS_REVIEWS = "No previous reviews yet."
INITIAL_REVIEW_STATE = json.dumps(
    {"status": "INACCURATE", "reasoning": "No previous review yet."}
)


def load_environment_variables() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dotenv_path = project_root / ".env"
    load_dotenv(dotenv_path=dotenv_path)


load_environment_variables()

TASK_PROMPT = """
Write a Python function named `calculate_factorial`.
Requirements:
1. Accept a single integer `n`.
2. Return the factorial of `n`.
3. Include a clear docstring.
4. Return `1` when `n == 0`.
5. Raise `ValueError` when `n` is negative.
""".strip()


generator = LlmAgent(
    name="JuniorSoftwareEngineer",
    model=JUNIOR_MODEL,
    description="Junior Software Engineer that writes the latest version of the requested Python code.",
    instruction="""
    You are a junior software engineer implementing a Python solution.

    Original request:
    {task_request}

    Latest reviewer feedback:
    {latest_review}

    Prior reviewer feedback:
    {review_history}

    Current code from the previous iteration:
    {current_code}

    Rules:
    - Treat every requested change in the latest reviewer feedback as mandatory unless it conflicts with the original request.
    - If the reviewer suggests a concrete implementation change, apply it in this iteration.
    - Do not repeat the previous code unchanged when the reviewer requested changes.
    - If the latest reviewer feedback says no changes are required, return the current code unchanged.

    Write the next version of the code. Return only Python code.
    """,
    output_key="current_code",
)

reviewer = LlmAgent(
    name="SeniorSoftwareEngineer",
    model=SENIOR_MODEL,
    description="Senior Software Engineer that reviews the latest code while considering earlier reviews.",
    instruction="""
    You are a senior software engineer and expert Python reviewer.

    Original request:
    {task_request}

    Latest junior engineer code:
    {current_code}

    Previous review history:
    {review_history}

    Review the latest code version while considering the previous reviews.
    Only mention changes that are necessary to satisfy the request or to address unresolved earlier review items.
    Do not add praise, optional future improvements, or general commentary. Be critical as much as possible.
    If no changes are required, set status to "ACCURATE" and reasoning to exactly "No changes required."
    Return JSON with exactly these keys:
    - status: "ACCURATE" or "INACCURATE"
    - reasoning: concise required changes only
    """,
    output_key="latest_review",
)

reflection_workflow = Workflow(
    name="ReflectionWorkflow",
    description="Runs one deterministic generate-review pass using ADK Workflow primitives.",
    edges=[
        (START, generator),
        (generator, reviewer),
    ],
)

root_agent = reflection_workflow


def _resolve_maybe_awaitable(value):
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


def _normalize_max_iterations(max_iterations: int) -> int:
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer.")
    return max_iterations


def _parse_review(review_output) -> dict[str, str]:
    if isinstance(review_output, dict):
        return {
            "status": str(review_output.get("status", "")).strip().upper(),
            "reasoning": str(review_output.get("reasoning", "")).strip(),
        }

    if not isinstance(review_output, str):
        return {"status": "", "reasoning": ""}

    if "CODE_IS_PERFECT" in review_output.upper():
        return {"status": "ACCURATE", "reasoning": "No changes required."}

    text = review_output.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return {
                "status": str(parsed.get("status", "")).strip().upper(),
                "reasoning": str(parsed.get("reasoning", "")).strip(),
            }

    return {"status": "", "reasoning": text}


def _format_review_for_display(review_output) -> str:
    parsed_review = _parse_review(review_output)
    status = parsed_review["status"] or "UNKNOWN"
    reasoning = parsed_review["reasoning"] or str(review_output).strip()
    return f"Status: {status}\nReasoning: {reasoning}"


def _format_review_history(review_history: list[str]) -> str:
    if not review_history:
        return NO_PREVIOUS_REVIEWS

    formatted_reviews = []
    for index, review in enumerate(review_history, start=1):
        formatted_reviews.append(
            f"Review iteration {index}:\n{_format_review_for_display(review)}"
        )
    return "\n\n".join(formatted_reviews)


def _build_iteration_message(request: str, iteration: int, max_iterations: int):
    prompt = (
        f"Task request:\n{request}\n\n"
        f"Run reflection iteration {iteration} of {max_iterations}."
    )
    return types.Content(role="user", parts=[types.Part(text=prompt)])


def _get_session(runner: InMemoryRunner, user_id: str, session_id: str):
    return _resolve_maybe_awaitable(
        runner.session_service.get_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )
    )


def _update_session_state(
    runner: InMemoryRunner,
    user_id: str,
    session_id: str,
    updates: dict,
) -> None:
    session = _get_session(runner, user_id, session_id)

    if session and hasattr(session, "state") and isinstance(session.state, dict):
        session.state.update(updates)

    if hasattr(runner.session_service, "state") and isinstance(
        runner.session_service.state, dict
    ):
        runner.session_service.state.update(updates)


def run_review_pipeline(
    runner: InMemoryRunner,
    request: str,
    max_iterations: int = DEFAULT_MAX_REVIEW_ITERATIONS,
) -> str:
    """Run the ADK reflection workflow for a fixed number of iterations."""
    resolved_max_iterations = _normalize_max_iterations(max_iterations)
    print(f"\n--- Running ADK Reflection Workflow with request: '{request}' ---")

    latest_code = ""
    latest_review = INITIAL_REVIEW_STATE
    review_history: list[str] = []

    try:
        user_id = "user_123"
        session_id = str(uuid.uuid4())
        initial_state = {
            "task_request": request,
            "current_code": "",
            "latest_review": INITIAL_REVIEW_STATE,
            "review_history": NO_PREVIOUS_REVIEWS,
            "max_review_iterations": resolved_max_iterations,
        }

        _resolve_maybe_awaitable(
            runner.session_service.create_session(
                app_name=runner.app_name,
                user_id=user_id,
                session_id=session_id,
                state=initial_state,
            )
        )

        for iteration in range(resolved_max_iterations):
            _update_session_state(
                runner,
                user_id,
                session_id,
                {
                    "task_request": request,
                    "current_code": latest_code,
                    "latest_review": latest_review,
                    "review_history": _format_review_history(review_history),
                },
            )

            for _ in runner.run(
                user_id=user_id,
                session_id=session_id,
                new_message=_build_iteration_message(
                    request=request,
                    iteration=iteration + 1,
                    max_iterations=resolved_max_iterations,
                ),
            ):
                pass

            session = _get_session(runner, user_id, session_id)
            if session and hasattr(session, "state"):
                latest_code = session.state.get("current_code", latest_code)
                latest_review = session.state.get("latest_review", latest_review)

            review_history.append(latest_review)

            _update_session_state(
                runner,
                user_id,
                session_id,
                {
                    "current_code": latest_code,
                    "latest_review": latest_review,
                    "review_history": _format_review_history(review_history),
                },
            )

            print("\n" + "=" * 20 + f" ITERATION {iteration + 1} " + "=" * 20)
            print("\nCode version:\n")
            print(latest_code or "[No code returned]")
            print("\nReviewer comments:\n")
            print(_format_review_for_display(latest_review))

        return latest_code or latest_review
    except Exception as exc:
        message = f"An error occurred while processing your request: {exc}"
        print(message)
        return message


def main() -> None:
    print("--- Google ADK Reflection Workflow Example ---")
    print("Note: This requires Google ADK installed and authenticated.")

    runner = InMemoryRunner(root_agent)
    result = run_review_pipeline(runner, TASK_PROMPT)
    print(f"\nFinal Output:\n{result}")


if __name__ == "__main__":
    main()
