import importlib.util
import pathlib
import sys
import types

import pytest


MODULE_PATH = pathlib.Path("hands_one_code_examples/11_2_goal_monitoring.py")


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
        return "live crew result"


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

    spec = importlib.util.spec_from_file_location("goal_monitoring_example", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


goal_monitoring_example = load_module_with_fake_crewai()


def test_role_definitions_cover_requested_role_set():
    roles = goal_monitoring_example.build_role_definitions()

    assert tuple(role.name for role in roles) == (
        "Peer Programmer",
        "Code Reviewer",
        "Documenter",
        "Test Writer",
        "Prompt Refiner",
    )
    assert all(role.goal for role in roles)
    assert all(role.backstory for role in roles)
    assert all(role.responsibility for role in roles)


def test_goal_contract_validates_goal_criteria_and_duplicate_keys():
    criterion = goal_monitoring_example.SuccessCriterion("done", "Done")

    contract = goal_monitoring_example.build_goal_contract(
        "Build a thing",
        criteria=(criterion,),
    )

    assert contract.goal == "Build a thing"
    assert contract.criteria == (criterion,)

    with pytest.raises(ValueError, match="goal must not be empty"):
        goal_monitoring_example.build_goal_contract("   ")

    with pytest.raises(ValueError, match="at least one success criterion"):
        goal_monitoring_example.GoalContract(goal="Build", criteria=())

    with pytest.raises(ValueError, match="duplicate success criteria"):
        goal_monitoring_example.GoalContract(
            goal="Build",
            criteria=(criterion, criterion),
        )


def test_goal_monitor_reports_missing_and_complete_criteria():
    contract = goal_monitoring_example.build_goal_contract("Build a thing")
    monitor = goal_monitoring_example.GoalMonitor(contract)

    partial = monitor.evaluate(
        [
            goal_monitoring_example.CRITERION_IMPLEMENTATION,
            goal_monitoring_example.CRITERION_REVIEW,
        ]
    )
    complete = monitor.evaluate(criterion.key for criterion in contract.criteria)

    assert partial.status == "needs_revision"
    assert partial.score == 0.4
    assert goal_monitoring_example.CRITERION_TESTS in partial.missing_criteria
    assert complete.status == "complete"
    assert complete.score == 1.0
    assert complete.missing_criteria == ()


def test_workflow_runs_reflection_style_role_conversation_to_completion():
    result = goal_monitoring_example.run_goal_monitoring_demo("Build BinaryGap")

    speakers = [message.speaker for message in result.conversation]
    stages = [message.stage for message in result.conversation]

    assert result.evaluation.is_complete is True
    assert result.evaluation.score == 1.0
    assert result.evaluation.missing_criteria == ()
    assert speakers == [
        "Peer Programmer",
        "Code Reviewer",
        "Peer Programmer",
        "Test Writer",
        "Documenter",
        "Prompt Refiner",
        "Code Reviewer",
    ]
    assert "initial implementation" in stages
    assert "review" in stages
    assert "revision" in stages
    assert "implementation" in result.artifacts
    assert "ValueError" in result.artifacts["implementation"]


def test_format_goal_monitoring_result_includes_conversation_and_monitoring_summary():
    result = goal_monitoring_example.run_goal_monitoring_demo("Build BinaryGap")

    text = goal_monitoring_example.format_goal_monitoring_result(result)

    assert "# Role Conversation" in text
    assert "Peer Programmer [initial implementation]" in text
    assert "Code Reviewer [final monitoring check]" in text
    assert "Status: complete" in text
    assert "Missing: none" in text
    assert "refined_prompt" in text


def test_print_demo_outputs_whole_role_conversation(capsys):
    app = goal_monitoring_example.GoalMonitoringCrewApp(goal="Build BinaryGap")

    result = app.print_demo()
    output = capsys.readouterr().out

    assert result.evaluation.is_complete is True
    assert output.index("Peer Programmer") < output.index("Code Reviewer")
    assert "Goal Monitoring Summary" in output
    assert "Prompt Refiner" in output


def test_crewai_builders_create_agents_tasks_and_sequential_crew():
    llm = goal_monitoring_example.create_llm(
        "gemini/gemini-2.5-flash",
        temperature=0.1,
    )
    agents = goal_monitoring_example.build_crewai_agents(llm=llm, verbose=True)
    tasks = goal_monitoring_example.build_crewai_tasks(agents=agents)
    crew = goal_monitoring_example.build_goal_monitoring_crew(
        goal="Build BinaryGap",
        llm=llm,
        verbose=True,
    )

    assert llm.model == "gemini/gemini-2.5-flash"
    assert llm.temperature == 0.1
    assert set(agents) == {
        "Peer Programmer",
        "Code Reviewer",
        "Documenter",
        "Test Writer",
        "Prompt Refiner",
    }
    assert all(agent.llm is llm for agent in agents.values())
    assert all(agent.allow_delegation is False for agent in agents.values())
    assert [task.name for task in tasks] == [
        "peer_programming_task",
        "code_review_task",
        "test_writing_task",
        "documentation_task",
        "prompt_refinement_task",
    ]
    assert tasks[1].context == [tasks[0]]
    assert tasks[-1].context == tasks[:-1]
    assert crew.process == FakeProcess.sequential
    assert len(crew.agents) == 5
    assert len(crew.tasks) == 5
    assert crew.verbose is True


def test_live_crew_app_uses_kickoff_inputs(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    app = goal_monitoring_example.GoalMonitoringCrewApp(
        goal="Build BinaryGap",
        verbose=True,
    )

    result = app.run_live_crew()

    assert result == "live crew result"


def test_runtime_environment_validation_and_google_key_normalization(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY or GOOGLE_API_KEY"):
        goal_monitoring_example.validate_runtime_environment()

    goal_monitoring_example.validate_runtime_environment("openai/gpt-4o-mini")

    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    goal_monitoring_example.normalize_model_provider_environment()

    assert goal_monitoring_example.os.environ["GEMINI_API_KEY"] == "google-key"


def test_require_crewai_raises_helpful_error_when_dependency_missing(monkeypatch):
    monkeypatch.setattr(goal_monitoring_example, "Agent", None)
    monkeypatch.setattr(goal_monitoring_example, "Crew", None)
    monkeypatch.setattr(goal_monitoring_example, "LLM", None)
    monkeypatch.setattr(goal_monitoring_example, "Process", None)
    monkeypatch.setattr(goal_monitoring_example, "Task", None)

    with pytest.raises(ImportError, match="crewai is not installed"):
        goal_monitoring_example.require_crewai()
