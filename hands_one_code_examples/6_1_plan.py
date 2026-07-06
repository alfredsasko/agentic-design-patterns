from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from crewai import Agent, Crew, LLM, Process, Task
except ImportError:
    Agent = Crew = LLM = Process = Task = None  # type: ignore[assignment]


DEFAULT_TOPIC = "The importance of Reinforcement Learning in AI"
DEFAULT_WRITER_MODEL = "openai/gpt-4o-mini"
DEFAULT_PLANNING_MODEL = "openai/gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.2


def load_environment_variables() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")


def require_crewai() -> None:
    if Agent is None or Crew is None or LLM is None or Process is None or Task is None:
        raise ImportError(
            "crewai is not installed. Install it with `uv add crewai==0.130.0` "
            "before running this planning example."
        )


def _model_requires_openai_api_key(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.startswith("openai/") or normalized.startswith("gpt-")


def validate_runtime_environment(
    *,
    writer_model: str = DEFAULT_WRITER_MODEL,
    planning_model: str = DEFAULT_PLANNING_MODEL,
) -> None:
    if (
        _model_requires_openai_api_key(writer_model)
        or _model_requires_openai_api_key(planning_model)
    ) and not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY not found. Set it before running this CrewAI planning example."
        )


def create_llm(
    model: str = DEFAULT_WRITER_MODEL,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Any:
    require_crewai()
    return LLM(
        model=model,
        temperature=temperature,
    )


def build_planner_writer_agent(
    *,
    llm: Any | None = None,
    verbose: bool = False,
    allow_delegation: bool = False,
) -> Any:
    require_crewai()
    active_llm = llm or create_llm()
    return Agent(
        role="Article Planner and Writer",
        goal="Plan first, then write a concise, engaging technical summary.",
        backstory=(
            "You are an expert technical writer who relies on structured planning "
            "before drafting the final response."
        ),
        llm=active_llm,
        verbose=verbose,
        allow_delegation=allow_delegation,
    )


def build_planning_task(
    *,
    topic: str = DEFAULT_TOPIC,
    agent: Any | None = None,
) -> Any:
    require_crewai()
    if not topic.strip():
        raise ValueError("topic must not be empty")

    active_agent = agent or build_planner_writer_agent()
    return Task(
        name="planning_task",
        description=(
            "Create a short bullet-point plan how to summarize topic: '{topic}'. "
            "Focus on the list of actions and the order they should appear in."
        ),
        expected_output=(
            "A bullet-point plan with the step-by-step procedure to summarize the topic."
        ),
        agent=active_agent,
        markdown=True,
    )


def build_writing_task(
    *,
    topic: str = DEFAULT_TOPIC,
    agent: Any | None = None,
    context: list[Any] | None = None,
    markdown: bool = True,
) -> Any:
    require_crewai()
    if not topic.strip():
        raise ValueError("topic must not be empty")

    active_agent = agent or build_planner_writer_agent()
    active_context = context or []
    return Task(
        name="writing_task",
        description=(
            "Write the final summary about '{topic}' using the plan from the "
            "previous task. Do not repeat the plan verbatim."
        ),
        expected_output=(
            "A concise summary of about 180-220 words that explains the topic, "
            "highlights why it matters, and ends with a short practical takeaway."
        ),
        agent=active_agent,
        context=active_context,
        markdown=markdown,
    )


def build_planning_crew(
    *,
    topic: str = DEFAULT_TOPIC,
    agent: Any | None = None,
    planning_task: Any | None = None,
    writing_task: Any | None = None,
    writer_llm: Any | None = None,
    planning_llm: Any | None = None,
    verbose: bool = False,
) -> Any:
    require_crewai()
    active_writer_llm = writer_llm or create_llm(DEFAULT_WRITER_MODEL)
    active_planning_llm = planning_llm or create_llm(DEFAULT_PLANNING_MODEL)
    active_agent = agent or build_planner_writer_agent(
        llm=active_writer_llm,
        verbose=verbose,
    )
    active_planning_task = planning_task or build_planning_task(
        topic=topic,
        agent=active_agent,
    )
    active_writing_task = writing_task or build_writing_task(
        topic=topic,
        agent=active_agent,
        context=[active_planning_task],
    )
    return Crew(
        agents=[active_agent],
        tasks=[active_planning_task, active_writing_task],
        process=Process.sequential,
        planning=True,
        planning_llm=active_planning_llm,
        verbose=verbose,
    )


@dataclass(frozen=True)
class PlanningWritingApp:
    """OO facade around a CrewAI planning-enabled writing crew."""

    crew: Any
    agent: Any
    planning_task: Any
    writing_task: Any
    writer_llm: Any
    planning_llm: Any
    topic: str = DEFAULT_TOPIC

    @classmethod
    def build(
        cls,
        *,
        topic: str = DEFAULT_TOPIC,
        writer_model: str = DEFAULT_WRITER_MODEL,
        planning_model: str = DEFAULT_PLANNING_MODEL,
        verbose: bool = False,
    ) -> "PlanningWritingApp":
        writer_llm = create_llm(writer_model)
        planning_llm = create_llm(planning_model)
        agent = build_planner_writer_agent(
            llm=writer_llm,
            verbose=verbose,
        )
        planning_task = build_planning_task(
            topic=topic,
            agent=agent,
        )
        writing_task = build_writing_task(
            topic=topic,
            agent=agent,
            context=[planning_task],
        )
        crew = build_planning_crew(
            topic=topic,
            agent=agent,
            planning_task=planning_task,
            writing_task=writing_task,
            writer_llm=writer_llm,
            planning_llm=planning_llm,
            verbose=verbose,
        )
        return cls(
            crew=crew,
            agent=agent,
            planning_task=planning_task,
            writing_task=writing_task,
            writer_llm=writer_llm,
            planning_llm=planning_llm,
            topic=topic,
        )

    def run(self) -> Any:
        return self.crew.kickoff(inputs={"topic": self.topic})

    def print_execution(self) -> Any:
        result = self.run()

        planning_output = getattr(self.planning_task, "output", None)
        planning_text = _extract_task_output_text(planning_output)
        summary_output = getattr(self.writing_task, "output", None)
        summary_text = _extract_task_output_text(summary_output)

        print("## Plan")
        print(planning_text or "No plan output was returned.")
        print()
        print("## Summary")
        print(summary_text or _extract_task_output_text(result) or str(result))
        return result


def _extract_task_output_text(output: Any) -> str:
    if output is None:
        return ""

    for attr_name in ("raw", "result", "output", "text"):
        value = getattr(output, attr_name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    if isinstance(output, str):
        return output.strip()

    return str(output).strip()


def main() -> None:
    load_environment_variables()
    validate_runtime_environment()
    app = PlanningWritingApp.build()
    app.print_execution()


if __name__ == "__main__":
    main()
