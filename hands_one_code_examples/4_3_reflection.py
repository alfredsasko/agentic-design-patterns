import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, NamedTuple

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_ITERATIONS = 3
NO_PREVIOUS_CRITIQUES = "No previous critiques yet."
NO_PREVIOUS_DESCRIPTIONS = "No previous descriptions yet."


class ReflectionPrompts(NamedTuple):
    generation: ChatPromptTemplate
    critique: ChatPromptTemplate
    revision: ChatPromptTemplate


class ReflectionChains(NamedTuple):
    generation: Any
    critique: Any
    revision: Any


class ReflectionStep(NamedTuple):
    iteration: int
    description_before_review: str
    critique: str
    revised_description: str
    accepted: bool


class ReflectionResult(NamedTuple):
    initial_description: str
    final_description: str
    steps: tuple[ReflectionStep, ...]
    iterations_run: int


class CritiqueDecision(NamedTuple):
    status: str
    required_changes: tuple[str, ...]


def load_environment_variables() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dotenv_path = project_root / ".env"
    load_dotenv(dotenv_path=dotenv_path)


load_environment_variables()


def create_llm(model: str = DEFAULT_MODEL, temperature: float = 0.2) -> ChatOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found. Set it before running this example.")
    return ChatOpenAI(model=model, temperature=temperature)


def create_reflection_prompts() -> ReflectionPrompts:
    generation = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You write concise product descriptions. Create one strong first draft "
                    "from the product details. Return only the product description."
                ),
            ),
            ("user", "Product details:\n{product_details}"),
        ]
    )

    critique = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are the reflection reviewer for a product-description writer. "
                    "Only list changes that are required to improve clarity, concision, "
                    "specificity, or customer appeal. Do not include praise, optional ideas, "
                    "or general commentary. Return valid JSON with exactly these keys: "
                    '"status" as "ACCEPTED" or "NEEDS_REVISION", and '
                    '"required_changes" as a list of strings. Use "ACCEPTED" only when '
                    "no required changes remain. Compare the current description against the "
                    "previous critiques and previous description versions. Do not repeat "
                    "critique items that were already addressed. Only surface unresolved or "
                    "newly discovered required changes."
                ),
            ),
            (
                "user",
                (
                    "Product details:\n{product_details}\n\n"
                    "Current description:\n{current_description}\n\n"
                    "Previous description versions:\n{description_history}\n\n"
                    "Previous critiques:\n{critique_history}"
                ),
            ),
        ]
    )

    revision = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You revise product descriptions. Apply every required change from the "
                    "latest critique. Do not ignore or reinterpret critique items. Preserve "
                    "accurate details from the original product details. Return only the "
                    "revised product description."
                ),
            ),
            (
                "user",
                (
                    "Product details:\n{product_details}\n\n"
                    "Current description:\n{current_description}\n\n"
                    "Latest critique:\n{critique}\n\n"
                    "Previous description versions:\n{description_history}\n\n"
                    "Previous critiques:\n{critique_history}"
                ),
            ),
        ]
    )

    return ReflectionPrompts(generation=generation, critique=critique, revision=revision)


def build_reflection_chains(llm_client: Any) -> ReflectionChains:
    prompts = create_reflection_prompts()
    parser = StrOutputParser()
    return ReflectionChains(
        generation=prompts.generation | llm_client | parser,
        critique=prompts.critique | llm_client | parser,
        revision=prompts.revision | llm_client | parser,
    )


def _normalize_max_iterations(max_iterations: int) -> int:
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer.")
    return max_iterations


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def parse_critique_decision(critique: str) -> CritiqueDecision:
    text = _strip_json_fence(critique)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return CritiqueDecision(status="NEEDS_REVISION", required_changes=(text,))

    status = str(payload.get("status", "")).strip().upper()
    raw_changes = payload.get("required_changes", [])
    if isinstance(raw_changes, str):
        required_changes = (raw_changes.strip(),) if raw_changes.strip() else ()
    elif isinstance(raw_changes, list):
        required_changes = tuple(str(change).strip() for change in raw_changes if str(change).strip())
    else:
        required_changes = ()

    return CritiqueDecision(status=status, required_changes=required_changes)


def critique_requires_revision(critique: str) -> bool:
    decision = parse_critique_decision(critique)
    accepted_statuses = {"ACCEPTED", "ACCURATE", "NO_CHANGES_REQUIRED"}
    return decision.status not in accepted_statuses


def format_critique_history(critiques: list[str]) -> str:
    if not critiques:
        return NO_PREVIOUS_CRITIQUES

    formatted = []
    for index, critique in enumerate(critiques, start=1):
        decision = parse_critique_decision(critique)
        changes = "; ".join(decision.required_changes) or "No required changes."
        formatted.append(f"Critique {index}: status={decision.status}; required_changes={changes}")
    return "\n".join(formatted)


def format_description_history(descriptions: list[str]) -> str:
    if not descriptions:
        return NO_PREVIOUS_DESCRIPTIONS

    formatted = []
    for index, description in enumerate(descriptions, start=1):
        formatted.append(f"Description {index}: {description}")
    return "\n".join(formatted)


async def run_reflection_loop(
    product_details: str,
    *,
    chains: ReflectionChains | None = None,
    llm_client: Any | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    verbose: bool = True,
) -> ReflectionResult:
    resolved_max_iterations = _normalize_max_iterations(max_iterations)
    if chains is None:
        chains = build_reflection_chains(llm_client or create_llm())

    initial_description = await chains.generation.ainvoke(
        {"product_details": product_details}
    )
    current_description = initial_description
    critiques: list[str] = []
    prior_descriptions: list[str] = []
    steps: list[ReflectionStep] = []

    if verbose:
        print(f"\n--- Running LangChain Reflection Loop for: '{product_details}' ---")
        print("\nInitial description:\n")
        print(initial_description)

    for iteration in range(1, resolved_max_iterations + 1):
        critique_history = format_critique_history(critiques)
        description_history = format_description_history(prior_descriptions)
        critique = await chains.critique.ainvoke(
            {
                "product_details": product_details,
                "current_description": current_description,
                "description_history": description_history,
                "critique_history": critique_history,
            }
        )
        accepted = not critique_requires_revision(critique)

        if accepted:
            revised_description = current_description
        else:
            revised_description = await chains.revision.ainvoke(
                {
                    "product_details": product_details,
                    "current_description": current_description,
                    "critique": critique,
                    "description_history": description_history,
                    "critique_history": critique_history,
                }
            )

        steps.append(
            ReflectionStep(
                iteration=iteration,
                description_before_review=current_description,
                critique=critique,
                revised_description=revised_description,
                accepted=accepted,
            )
        )

        if verbose:
            print("\n" + "=" * 20 + f" ITERATION {iteration} " + "=" * 20)
            print("\nDescription under review:\n")
            print(current_description)
            print("\nReviewer required changes:\n")
            print(critique)
            print("\nRevised description:\n")
            print(revised_description)

        critiques.append(critique)
        prior_descriptions.append(current_description)
        current_description = revised_description
        if accepted:
            break

    return ReflectionResult(
        initial_description=initial_description,
        final_description=current_description,
        steps=tuple(steps),
        iterations_run=len(steps),
    )


async def run_reflection_example(product_details: str) -> ReflectionResult:
    return await run_reflection_loop(product_details)


if __name__ == "__main__":
    test_product_details = "A mug that keeps coffee hot and can be controlled by a smartphone app."
    result = asyncio.run(run_reflection_example(test_product_details))
    print("\n--- Final Refined Product Description ---")
    print(result.final_description)
