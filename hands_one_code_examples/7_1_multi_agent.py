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


DEFAULT_TOPIC = "Emerging AI trends in 2026"
DEFAULT_RESEARCH_MODEL = "gemini/gemini-2.5-flash"
DEFAULT_WRITER_MODEL = "gemini/gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_USER_AUDIENCE = "general technology readers"


def normalize_model_provider_environment() -> None:
    """Bridge repo env naming with LiteLLM's Gemini env lookup."""
    if os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]


def load_environment_variables() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")


def require_crewai() -> None:
    if Agent is None or Crew is None or LLM is None or Process is None or Task is None:
        raise ImportError(
            "crewai is not installed. Install it with `uv add crewai==0.130.0` "
            "before running this multi-agent example."
        )


def _model_requires_google_api_key(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.startswith("gemini/") or normalized.startswith("google/")


def validate_runtime_environment(
    *,
    research_model: str = DEFAULT_RESEARCH_MODEL,
    writer_model: str = DEFAULT_WRITER_MODEL,
) -> None:
    if (
        _model_requires_google_api_key(research_model)
        or _model_requires_google_api_key(writer_model)
    ) and not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        raise ValueError(
            "GEMINI_API_KEY or GOOGLE_API_KEY not found. Set one of them before running "
            "this CrewAI multi-agent example."
        )


def create_llm(
    model: str = DEFAULT_RESEARCH_MODEL,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Any:
    require_crewai()
    return LLM(
        model=model,
        temperature=temperature,
    )


def build_research_agent(
    *,
    llm: Any | None = None,
    verbose: bool = False,
    allow_delegation: bool = False,
) -> Any:
    require_crewai()
    active_llm = llm or create_llm(DEFAULT_RESEARCH_MODEL)
    return Agent(
        role="AI Research Analyst",
        goal="Identify the most relevant recent developments on the assigned AI topic.",
        backstory=(
            "You are a careful research analyst who summarizes complex developments "
            "into concrete findings with practical significance."
        ),
        llm=active_llm,
        verbose=verbose,
        allow_delegation=allow_delegation,
    )


def build_writer_agent(
    *,
    llm: Any | None = None,
    verbose: bool = False,
    allow_delegation: bool = False,
) -> Any:
    require_crewai()
    active_llm = llm or create_llm(DEFAULT_WRITER_MODEL)
    return Agent(
        role="Technical Content Writer",
        goal="Transform research notes into a clear, structured article for the target audience.",
        backstory=(
            "You are a technical writer who converts raw research into concise, "
            "readable content without overstating claims."
        ),
        llm=active_llm,
        verbose=verbose,
        allow_delegation=allow_delegation,
    )


def build_research_task(
    *,
    topic: str = DEFAULT_TOPIC,
    audience: str = DEFAULT_USER_AUDIENCE,
    agent: Any | None = None,
) -> Any:
    require_crewai()
    if not topic.strip():
        raise ValueError("topic must not be empty")
    if not audience.strip():
        raise ValueError("audience must not be empty")

    active_agent = agent or build_research_agent()
    return Task(
        name="research_task",
        description=(
            "Research the topic '{topic}' and identify the three most important recent "
            "developments. Focus on what changed, why it matters, and one concrete example "
            "for each development. Tailor the findings for {audience}."
        ),
        expected_output=(
            "A markdown research brief with three sections. Each section must include: "
            "the development, why it matters, and one practical example."
        ),
        agent=active_agent,
        markdown=True,
    )


def build_writing_task(
    *,
    topic: str = DEFAULT_TOPIC,
    audience: str = DEFAULT_USER_AUDIENCE,
    agent: Any | None = None,
    context: list[Any] | None = None,
) -> Any:
    require_crewai()
    if not topic.strip():
        raise ValueError("topic must not be empty")
    if not audience.strip():
        raise ValueError("audience must not be empty")

    active_agent = agent or build_writer_agent()
    active_context = context or []
    return Task(
        name="writing_task",
        description=(
            "Write a blog-style article about '{topic}' for {audience} using the research "
            "brief from the previous task. Preserve factual accuracy, explain the topic in a "
            "simple way, and avoid repeating the research notes verbatim."
        ),
        expected_output=(
            "A markdown article of about 450-550 words with a title, short introduction, "
            "three themed sections, and a short conclusion."
        ),
        agent=active_agent,
        context=active_context,
        markdown=True,
    )


def build_multi_agent_crew(
    *,
    topic: str = DEFAULT_TOPIC,
    audience: str = DEFAULT_USER_AUDIENCE,
    research_agent: Any | None = None,
    writer_agent: Any | None = None,
    research_task: Any | None = None,
    writing_task: Any | None = None,
    research_llm: Any | None = None,
    writer_llm: Any | None = None,
    verbose: bool = False,
) -> Any:
    require_crewai()
    active_research_llm = research_llm or create_llm(DEFAULT_RESEARCH_MODEL)
    active_writer_llm = writer_llm or create_llm(DEFAULT_WRITER_MODEL)
    active_research_agent = research_agent or build_research_agent(
        llm=active_research_llm,
        verbose=verbose,
    )
    active_writer_agent = writer_agent or build_writer_agent(
        llm=active_writer_llm,
        verbose=verbose,
    )
    active_research_task = research_task or build_research_task(
        topic=topic,
        audience=audience,
        agent=active_research_agent,
    )
    active_writing_task = writing_task or build_writing_task(
        topic=topic,
        audience=audience,
        agent=active_writer_agent,
        context=[active_research_task],
    )
    return Crew(
        agents=[active_research_agent, active_writer_agent],
        tasks=[active_research_task, active_writing_task],
        process=Process.sequential,
        verbose=verbose,
    )


def _extract_output_text(output: Any) -> str:
    if output is None:
        return ""

    for attr_name in ("raw", "result", "output", "text"):
        value = getattr(output, attr_name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    if isinstance(output, str):
        return output.strip()

    return ""


@dataclass(frozen=True)
class MultiAgentBlogApp:
    """OO facade around a CrewAI sequential multi-agent collaboration example."""

    crew: Any
    research_agent: Any
    writer_agent: Any
    research_task: Any
    writing_task: Any
    research_llm: Any
    writer_llm: Any
    topic: str = DEFAULT_TOPIC
    audience: str = DEFAULT_USER_AUDIENCE

    @classmethod
    def build(
        cls,
        *,
        topic: str = DEFAULT_TOPIC,
        audience: str = DEFAULT_USER_AUDIENCE,
        research_model: str = DEFAULT_RESEARCH_MODEL,
        writer_model: str = DEFAULT_WRITER_MODEL,
        verbose: bool = False,
    ) -> "MultiAgentBlogApp":
        research_llm = create_llm(research_model)
        writer_llm = create_llm(writer_model)
        research_agent = build_research_agent(
            llm=research_llm,
            verbose=verbose,
        )
        writer_agent = build_writer_agent(
            llm=writer_llm,
            verbose=verbose,
        )
        research_task = build_research_task(
            topic=topic,
            audience=audience,
            agent=research_agent,
        )
        writing_task = build_writing_task(
            topic=topic,
            audience=audience,
            agent=writer_agent,
            context=[research_task],
        )
        crew = build_multi_agent_crew(
            topic=topic,
            audience=audience,
            research_agent=research_agent,
            writer_agent=writer_agent,
            research_task=research_task,
            writing_task=writing_task,
            research_llm=research_llm,
            writer_llm=writer_llm,
            verbose=verbose,
        )
        return cls(
            crew=crew,
            research_agent=research_agent,
            writer_agent=writer_agent,
            research_task=research_task,
            writing_task=writing_task,
            research_llm=research_llm,
            writer_llm=writer_llm,
            topic=topic,
            audience=audience,
        )

    def run(self) -> Any:
        return self.crew.kickoff(
            inputs={
                "topic": self.topic,
                "audience": self.audience,
            }
        )

    def print_execution(self) -> Any:
        result = self.run()
        research_text = _extract_output_text(getattr(self.research_task, "output", None))
        article_text = _extract_output_text(getattr(self.writing_task, "output", None))

        print("## Research Brief")
        print(research_text or "No research brief was returned.")
        print()
        print("## Final Article")
        print(article_text or _extract_output_text(result) or str(result))
        return result


def main() -> None:
    load_environment_variables()
    normalize_model_provider_environment()
    validate_runtime_environment()
    app = MultiAgentBlogApp.build()
    app.print_execution()


if __name__ == "__main__":
    main()
