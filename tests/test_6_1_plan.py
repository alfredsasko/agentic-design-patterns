import importlib.util
import pathlib
import sys
import types

import pytest


MODULE_PATH = pathlib.Path("hands_one_code_examples/6_1_plan.py")


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
            self.tasks[0].output = types.SimpleNamespace(raw="generated plan")
        if len(self.tasks) > 1:
            self.tasks[1].output = types.SimpleNamespace(raw="generated summary")
        return "planned summary output"


class FakeProcess:
    sequential = "sequential"


def load_module_with_fake_crewai():
    fake_crewai = types.ModuleType("crewai")
    fake_crewai.Agent = FakeAgent
    fake_crewai.Crew = FakeCrew
    fake_crewai.LLM = FakeLLM
    fake_crewai.Process = FakeProcess
    fake_crewai.Task = FakeTask

    sys.modules["crewai"] = fake_crewai

    spec = importlib.util.spec_from_file_location("plan_crewai_example", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plan_crewai_example = load_module_with_fake_crewai()


def test_create_llm_uses_crewai_native_llm():
    llm = plan_crewai_example.create_llm(
        "openai/gpt-4o-mini",
        temperature=0.1,
    )

    assert llm.model == "openai/gpt-4o-mini"
    assert llm.temperature == 0.1


def test_build_planner_writer_agent_sets_safe_defaults():
    llm = plan_crewai_example.create_llm()

    agent = plan_crewai_example.build_planner_writer_agent(
        llm=llm,
        verbose=True,
    )

    assert agent.role == "Article Planner and Writer"
    assert agent.goal == "Plan first, then write a concise, engaging technical summary."
    assert agent.llm is llm
    assert agent.allow_delegation is False
    assert agent.verbose is True


def test_build_planning_task_uses_topic_placeholder_and_markdown():
    agent = plan_crewai_example.build_planner_writer_agent()

    task = plan_crewai_example.build_planning_task(
        topic="Reinforcement Learning",
        agent=agent,
    )

    assert task.name == "planning_task"
    assert "{topic}" in task.description
    assert "bullet-point plan" in task.expected_output
    assert task.agent is agent
    assert task.markdown is True


def test_build_planning_task_rejects_blank_topics():
    with pytest.raises(ValueError, match="topic must not be empty"):
        plan_crewai_example.build_planning_task(topic="   ")


def test_build_writing_task_uses_planning_context():
    agent = plan_crewai_example.build_planner_writer_agent()
    planning_task = plan_crewai_example.build_planning_task(agent=agent)

    writing_task = plan_crewai_example.build_writing_task(
        topic="Reinforcement Learning",
        agent=agent,
        context=[planning_task],
    )

    assert writing_task.name == "writing_task"
    assert "{topic}" in writing_task.description
    assert planning_task in writing_task.context
    assert writing_task.agent is agent


def test_build_writing_task_rejects_blank_topics():
    with pytest.raises(ValueError, match="topic must not be empty"):
        plan_crewai_example.build_writing_task(topic="   ")


def test_build_planning_crew_enables_planning_and_custom_planning_llm():
    writer_llm = plan_crewai_example.create_llm("openai/gpt-4o-mini")
    planning_llm = plan_crewai_example.create_llm("openai/gpt-4o")
    agent = plan_crewai_example.build_planner_writer_agent(llm=writer_llm)
    planning_task = plan_crewai_example.build_planning_task(agent=agent)
    writing_task = plan_crewai_example.build_writing_task(
        agent=agent,
        context=[planning_task],
    )

    crew = plan_crewai_example.build_planning_crew(
        agent=agent,
        planning_task=planning_task,
        writing_task=writing_task,
        writer_llm=writer_llm,
        planning_llm=planning_llm,
        verbose=True,
    )

    assert crew.agents == [agent]
    assert crew.tasks == [planning_task, writing_task]
    assert crew.process == FakeProcess.sequential
    assert crew.planning is True
    assert crew.planning_llm is planning_llm
    assert crew.verbose is True


def test_planning_writing_app_builds_consistent_objects_and_runs():
    app = plan_crewai_example.PlanningWritingApp.build(
        topic="The importance of Reinforcement Learning in AI",
        writer_model="openai/gpt-4o-mini",
        planning_model="openai/gpt-4o",
        verbose=True,
    )

    result = app.run()

    assert app.agent.llm is app.writer_llm
    assert app.crew.planning_llm is app.planning_llm
    assert app.crew.tasks == [app.planning_task, app.writing_task]
    assert result == "planned summary output"
    assert app.crew.kickoff_calls == [
        {"topic": "The importance of Reinforcement Learning in AI"}
    ]


def test_planning_writing_app_print_execution_shows_plan_before_summary(capsys):
    app = plan_crewai_example.PlanningWritingApp.build(
        topic="The importance of Reinforcement Learning in AI",
        writer_model="openai/gpt-4o-mini",
        planning_model="openai/gpt-4o",
    )

    result = app.print_execution()
    output = capsys.readouterr().out

    assert result == "planned summary output"
    assert "## Plan" in output
    assert "## Summary" in output
    assert output.index("## Plan") < output.index("## Summary")
    assert "generated plan" in output
    assert "generated summary" in output


def test_validate_runtime_environment_requires_openai_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY not found"):
        plan_crewai_example.validate_runtime_environment()


def test_validate_runtime_environment_allows_non_openai_models(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    plan_crewai_example.validate_runtime_environment(
        writer_model="gemini/gemini-2.5-flash",
        planning_model="gemini/gemini-2.5-pro",
    )


def test_require_crewai_raises_helpful_error_when_dependency_is_missing(monkeypatch):
    monkeypatch.setattr(plan_crewai_example, "Agent", None)
    monkeypatch.setattr(plan_crewai_example, "Crew", None)
    monkeypatch.setattr(plan_crewai_example, "LLM", None)
    monkeypatch.setattr(plan_crewai_example, "Process", None)
    monkeypatch.setattr(plan_crewai_example, "Task", None)

    with pytest.raises(ImportError, match="crewai is not installed"):
        plan_crewai_example.require_crewai()
