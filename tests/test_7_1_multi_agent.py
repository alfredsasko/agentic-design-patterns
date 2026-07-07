import importlib.util
import pathlib
import sys
import types

import pytest


MODULE_PATH = pathlib.Path("hands_one_code_examples/7_1_multi_agent.py")


class FakeLLM:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeAgent:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeTask:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeCrew:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.kickoff_calls = []

    def kickoff(self, inputs=None):
        self.kickoff_calls.append(inputs)
        if self.tasks:
            self.tasks[0].output = types.SimpleNamespace(raw="generated research brief")
        if len(self.tasks) > 1:
            self.tasks[1].output = types.SimpleNamespace(raw="generated final article")
        return "generated final article"


class FakeProcess:
    sequential = "sequential"
    hierarchical = "hierarchical"


def load_module_with_fake_crewai():
    fake_crewai = types.ModuleType("crewai")
    fake_crewai.Agent = FakeAgent
    fake_crewai.Crew = FakeCrew
    fake_crewai.LLM = FakeLLM
    fake_crewai.Process = FakeProcess
    fake_crewai.Task = FakeTask

    sys.modules["crewai"] = fake_crewai

    spec = importlib.util.spec_from_file_location("multi_agent_example", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


multi_agent_example = load_module_with_fake_crewai()


def test_create_llm_uses_crewai_native_llm():
    llm = multi_agent_example.create_llm(
        "gemini/gemini-2.5-flash",
        temperature=0.1,
    )

    assert llm.model == "gemini/gemini-2.5-flash"
    assert llm.temperature == 0.1


def test_build_research_agent_sets_expected_defaults():
    llm = multi_agent_example.create_llm()

    agent = multi_agent_example.build_research_agent(
        llm=llm,
        verbose=True,
    )

    assert agent.role == "AI Research Analyst"
    assert "recent developments" in agent.goal
    assert agent.llm is llm
    assert agent.allow_delegation is False
    assert agent.verbose is True


def test_build_writer_agent_sets_expected_defaults():
    llm = multi_agent_example.create_llm("gemini/gemini-2.5-flash")

    agent = multi_agent_example.build_writer_agent(
        llm=llm,
        verbose=True,
    )

    assert agent.role == "Technical Content Writer"
    assert "target audience" in agent.goal
    assert agent.llm is llm
    assert agent.allow_delegation is False
    assert agent.verbose is True


def test_build_research_task_uses_markdown_and_validates_inputs():
    agent = multi_agent_example.build_research_agent()

    task = multi_agent_example.build_research_task(
        topic="AI trends",
        audience="engineering leaders",
        agent=agent,
    )

    assert task.name == "research_task"
    assert "{topic}" in task.description
    assert "{audience}" in task.description
    assert "markdown research brief" in task.expected_output
    assert task.agent is agent
    assert task.markdown is True

    with pytest.raises(ValueError, match="topic must not be empty"):
        multi_agent_example.build_research_task(topic="   ")

    with pytest.raises(ValueError, match="audience must not be empty"):
        multi_agent_example.build_research_task(audience="   ")


def test_build_writing_task_uses_context_and_validates_inputs():
    research_agent = multi_agent_example.build_research_agent()
    writer_agent = multi_agent_example.build_writer_agent()
    research_task = multi_agent_example.build_research_task(agent=research_agent)

    task = multi_agent_example.build_writing_task(
        topic="AI trends",
        audience="general readers",
        agent=writer_agent,
        context=[research_task],
    )

    assert task.name == "writing_task"
    assert "{topic}" in task.description
    assert "{audience}" in task.description
    assert task.context == [research_task]
    assert task.markdown is True

    with pytest.raises(ValueError, match="topic must not be empty"):
        multi_agent_example.build_writing_task(topic="   ")

    with pytest.raises(ValueError, match="audience must not be empty"):
        multi_agent_example.build_writing_task(audience="   ")


def test_build_multi_agent_crew_uses_sequential_process():
    research_llm = multi_agent_example.create_llm("gemini/gemini-2.5-flash")
    writer_llm = multi_agent_example.create_llm("gemini/gemini-2.5-flash")
    research_agent = multi_agent_example.build_research_agent(llm=research_llm)
    writer_agent = multi_agent_example.build_writer_agent(llm=writer_llm)
    research_task = multi_agent_example.build_research_task(agent=research_agent)
    writing_task = multi_agent_example.build_writing_task(
        agent=writer_agent,
        context=[research_task],
    )

    crew = multi_agent_example.build_multi_agent_crew(
        research_agent=research_agent,
        writer_agent=writer_agent,
        research_task=research_task,
        writing_task=writing_task,
        research_llm=research_llm,
        writer_llm=writer_llm,
        verbose=True,
    )

    assert crew.agents == [research_agent, writer_agent]
    assert crew.tasks == [research_task, writing_task]
    assert crew.process == FakeProcess.sequential
    assert crew.verbose is True


def test_multi_agent_blog_app_builds_consistent_objects_and_runs():
    app = multi_agent_example.MultiAgentBlogApp.build(
        topic="AI trends in 2026",
        audience="startup operators",
        research_model="gemini/gemini-2.5-flash",
        writer_model="gemini/gemini-2.5-flash",
        verbose=True,
    )

    result = app.run()

    assert app.research_agent.llm is app.research_llm
    assert app.writer_agent.llm is app.writer_llm
    assert app.crew.tasks == [app.research_task, app.writing_task]
    assert result == "generated final article"
    assert app.crew.kickoff_calls == [
        {"topic": "AI trends in 2026", "audience": "startup operators"}
    ]


def test_multi_agent_blog_app_print_execution_shows_research_before_article(capsys):
    app = multi_agent_example.MultiAgentBlogApp.build(
        topic="AI trends in 2026",
        audience="general readers",
    )

    result = app.print_execution()
    output = capsys.readouterr().out

    assert result == "generated final article"
    assert "## Research Brief" in output
    assert "## Final Article" in output
    assert output.index("## Research Brief") < output.index("## Final Article")
    assert "generated research brief" in output
    assert "generated final article" in output


def test_validate_runtime_environment_requires_google_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY or GOOGLE_API_KEY not found"):
        multi_agent_example.validate_runtime_environment()


def test_validate_runtime_environment_allows_non_google_models(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    multi_agent_example.validate_runtime_environment(
        research_model="openai/gpt-4o-mini",
        writer_model="openai/gpt-4o-mini",
    )


def test_validate_runtime_environment_allows_gemini_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    multi_agent_example.validate_runtime_environment()


def test_normalize_model_provider_environment_copies_google_key_for_gemini(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    multi_agent_example.normalize_model_provider_environment()

    assert multi_agent_example.os.environ["GEMINI_API_KEY"] == "google-key"


def test_require_crewai_raises_helpful_error_when_dependency_is_missing(monkeypatch):
    monkeypatch.setattr(multi_agent_example, "Agent", None)
    monkeypatch.setattr(multi_agent_example, "Crew", None)
    monkeypatch.setattr(multi_agent_example, "LLM", None)
    monkeypatch.setattr(multi_agent_example, "Process", None)
    monkeypatch.setattr(multi_agent_example, "Task", None)

    with pytest.raises(ImportError, match="crewai is not installed"):
        multi_agent_example.require_crewai()
