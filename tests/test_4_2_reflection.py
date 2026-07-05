import importlib.util
import json
import pathlib
import sys
import types as builtin_types

import pytest


def _install_fake_google_adk_modules():
    google_module = builtin_types.ModuleType("google")
    adk_module = builtin_types.ModuleType("google.adk")
    agents_module = builtin_types.ModuleType("google.adk.agents")
    runners_module = builtin_types.ModuleType("google.adk.runners")
    workflow_module = builtin_types.ModuleType("google.adk.workflow")
    genai_module = builtin_types.ModuleType("google.genai")

    class FakeLlmAgent:
        def __init__(self, name, instruction, output_key=None, description=None, model=None):
            self.name = name
            self.instruction = instruction
            self.output_key = output_key
            self.description = description
            self.model = model

    class FakeInMemoryRunner:
        def __init__(self, agent):
            self.agent = agent

    class FakeWorkflow:
        def __init__(self, name, edges, description=""):
            self.name = name
            self.edges = edges
            self.description = description

    class FakePart:
        def __init__(self, text=None):
            self.text = text

    class FakeContent:
        def __init__(self, role=None, parts=None, text=None):
            self.role = role
            self.parts = parts or []
            self.text = text

    agents_module.LlmAgent = FakeLlmAgent
    runners_module.InMemoryRunner = FakeInMemoryRunner
    workflow_module.Workflow = FakeWorkflow
    workflow_module.START = "START"
    genai_module.types = builtin_types.SimpleNamespace(Content=FakeContent, Part=FakePart)

    google_module.adk = adk_module
    google_module.genai = genai_module
    adk_module.agents = agents_module
    adk_module.runners = runners_module
    adk_module.workflow = workflow_module

    sys.modules["google"] = google_module
    sys.modules["google.adk"] = adk_module
    sys.modules["google.adk.agents"] = agents_module
    sys.modules["google.adk.runners"] = runners_module
    sys.modules["google.adk.workflow"] = workflow_module
    sys.modules["google.genai"] = genai_module


_install_fake_google_adk_modules()

MODULE_PATH = pathlib.Path("hands_one_code_examples/4_2_reflection.py")

spec = importlib.util.spec_from_file_location("reflection_adk_example", MODULE_PATH)
reflection_adk_example = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(reflection_adk_example)


class DummySessionService:
    def __init__(self):
        self.calls = []
        self.state = {}

    def create_session(self, app_name, user_id, session_id, state=None):
        self.calls.append(
            {
                "app_name": app_name,
                "user_id": user_id,
                "session_id": session_id,
                "state": state,
            }
        )
        self.state = dict(state or {})

    def get_session(self, app_name, user_id, session_id):
        _ = (app_name, user_id, session_id)

        class DummySession:
            def __init__(self, state):
                self.state = state

        return DummySession(self.state)


class DummyPart:
    def __init__(self, text):
        self.text = text


class DummyContent:
    def __init__(self, text=None, parts=None):
        self.text = text
        self.parts = parts or []


class DummyEvent:
    def __init__(self, is_final, content):
        self._is_final = is_final
        self.content = content

    def is_final_response(self):
        return self._is_final


class DummyRunner:
    def __init__(self, run_steps):
        self.app_name = "test_app"
        self.session_service = DummySessionService()
        self._run_steps = list(run_steps)
        self.run_calls = []
        self.state_before_runs = []

    def run(self, **kwargs):
        self.run_calls.append(kwargs)
        self.state_before_runs.append(dict(self.session_service.state))
        step_index = len(self.run_calls) - 1
        step = self._run_steps[step_index]
        self.session_service.state.update(step.get("session_state_update", {}))
        for event in step.get("events", []):
            yield event


class ExplodingRunner:
    def __init__(self):
        self.app_name = "test_app"
        self.session_service = DummySessionService()

    def run(self, **kwargs):
        _ = kwargs
        raise RuntimeError("runner failed")


def _message_text(content):
    return "".join(part.text for part in content.parts if part.text)


def test_generator_agent_uses_latest_review_and_prior_history():
    assert reflection_adk_example.generator.name == "JuniorSoftwareEngineer"
    assert reflection_adk_example.generator.output_key == "current_code"
    assert reflection_adk_example.generator.model == reflection_adk_example.JUNIOR_MODEL
    assert "{latest_review}" in reflection_adk_example.generator.instruction
    assert "{review_history}" in reflection_adk_example.generator.instruction
    assert "Treat every requested change" in reflection_adk_example.generator.instruction
    assert "Do not repeat the previous code unchanged" in reflection_adk_example.generator.instruction


def test_reviewer_agent_reads_latest_code_and_previous_reviews():
    assert reflection_adk_example.reviewer.name == "SeniorSoftwareEngineer"
    assert reflection_adk_example.reviewer.output_key == "latest_review"
    assert reflection_adk_example.reviewer.model == reflection_adk_example.SENIOR_MODEL
    assert "{current_code}" in reflection_adk_example.reviewer.instruction
    assert "{review_history}" in reflection_adk_example.reviewer.instruction
    assert "Only mention changes that are necessary" in reflection_adk_example.reviewer.instruction
    assert 'reasoning to exactly "No changes required."' in reflection_adk_example.reviewer.instruction


def test_model_tier_constants_are_configured_for_junior_and_senior():
    assert reflection_adk_example.JUNIOR_MODEL == "gemini-2.5-flash"
    assert reflection_adk_example.SENIOR_MODEL == "gemini-2.5-pro"


def test_root_agent_points_to_workflow_reflection_graph():
    workflow = reflection_adk_example.reflection_workflow

    assert reflection_adk_example.root_agent is workflow
    assert workflow.name == "ReflectionWorkflow"
    assert workflow.edges[0][0] == "START"
    assert workflow.edges[0][1] is reflection_adk_example.generator
    assert workflow.edges[1][0] is reflection_adk_example.generator
    assert workflow.edges[1][1] is reflection_adk_example.reviewer


def test_default_max_review_iterations_is_three():
    assert reflection_adk_example.DEFAULT_MAX_REVIEW_ITERATIONS == 3


def test_parse_review_formats_json_and_perfect_phrase():
    accurate = reflection_adk_example._parse_review(
        '{"status": "ACCURATE", "reasoning": "Looks good."}'
    )
    perfect = reflection_adk_example._parse_review("CODE_IS_PERFECT")

    assert accurate == {"status": "ACCURATE", "reasoning": "Looks good."}
    assert perfect == {
        "status": "ACCURATE",
        "reasoning": "No changes required.",
    }


def test_run_review_pipeline_seeds_reflection_state():
    runner = DummyRunner(
        [
            {
                "session_state_update": {
                    "current_code": "def calculate_factorial(n):\n    return 1",
                    "latest_review": json.dumps(
                        {"status": "ACCURATE", "reasoning": "Looks good."}
                    ),
                },
                "events": [DummyEvent(True, DummyContent(text="ignored"))],
            },
            {
                "session_state_update": {
                    "current_code": "same code",
                    "latest_review": json.dumps(
                        {"status": "ACCURATE", "reasoning": "Still good."}
                    ),
                },
                "events": [DummyEvent(True, DummyContent(text="ignored"))],
            },
            {
                "session_state_update": {
                    "current_code": "same code",
                    "latest_review": json.dumps(
                        {"status": "ACCURATE", "reasoning": "Still good."}
                    ),
                },
                "events": [DummyEvent(True, DummyContent(text="ignored"))],
            },
        ]
    )

    reflection_adk_example.run_review_pipeline(runner, "Implement factorial")

    created_state = runner.session_service.calls[0]["state"]
    assert created_state["task_request"] == "Implement factorial"
    assert created_state["current_code"] == ""
    assert created_state["review_history"] == "No previous reviews yet."
    assert created_state["max_review_iterations"] == 3


def test_run_review_pipeline_runs_full_default_iterations_even_if_first_review_is_accurate(capsys):
    accurate_review = json.dumps(
        {"status": "ACCURATE", "reasoning": "The code already satisfies the requirements."}
    )
    runner = DummyRunner(
        [
            {
                "session_state_update": {
                    "current_code": "code v1",
                    "latest_review": accurate_review,
                },
                "events": [DummyEvent(True, DummyContent(text=accurate_review))],
            },
            {
                "session_state_update": {
                    "current_code": "code v2",
                    "latest_review": accurate_review,
                },
                "events": [DummyEvent(True, DummyContent(text=accurate_review))],
            },
            {
                "session_state_update": {
                    "current_code": "code v3",
                    "latest_review": accurate_review,
                },
                "events": [DummyEvent(True, DummyContent(text=accurate_review))],
            },
        ]
    )

    result = reflection_adk_example.run_review_pipeline(runner, "Implement factorial")

    assert result == "code v3"
    assert len(runner.run_calls) == 3

    captured = capsys.readouterr()
    assert "ITERATION 1" in captured.out
    assert "ITERATION 2" in captured.out
    assert "ITERATION 3" in captured.out
    assert "Code version:" in captured.out
    assert "Reviewer comments:" in captured.out


def test_run_review_pipeline_passes_latest_code_and_prior_reviews_between_iterations():
    first_review = json.dumps(
        {"status": "INACCURATE", "reasoning": "Add a docstring and negative guard."}
    )
    second_review = json.dumps(
        {"status": "INACCURATE", "reasoning": "The docstring is better, but the loop should use clearer names."}
    )
    third_review = json.dumps(
        {"status": "ACCURATE", "reasoning": "All requirements are now satisfied."}
    )
    runner = DummyRunner(
        [
            {
                "session_state_update": {
                    "current_code": "def calculate_factorial(n):\n    return n",
                    "latest_review": first_review,
                },
                "events": [DummyEvent(True, DummyContent(text=first_review))],
            },
            {
                "session_state_update": {
                    "current_code": "def calculate_factorial(n):\n    return 1",
                    "latest_review": second_review,
                },
                "events": [DummyEvent(True, DummyContent(text=second_review))],
            },
            {
                "session_state_update": {
                    "current_code": "def calculate_factorial(n):\n    return 2",
                    "latest_review": third_review,
                },
                "events": [DummyEvent(True, DummyContent(text=third_review))],
            },
        ]
    )

    result = reflection_adk_example.run_review_pipeline(
        runner,
        "Implement factorial",
        max_iterations=3,
    )

    assert result == "def calculate_factorial(n):\n    return 2"
    assert len(runner.run_calls) == 3
    assert runner.state_before_runs[1]["current_code"] == "def calculate_factorial(n):\n    return n"
    assert "Add a docstring and negative guard." in runner.state_before_runs[1]["review_history"]
    assert "Add a docstring and negative guard." in runner.state_before_runs[2]["review_history"]
    assert "The docstring is better" in runner.state_before_runs[2]["review_history"]
    assert "Run reflection iteration 2 of 3." in _message_text(runner.run_calls[1]["new_message"])


def test_run_review_pipeline_respects_custom_max_iterations(capsys):
    inaccurate_review = json.dumps(
        {"status": "INACCURATE", "reasoning": "Still missing validation."}
    )
    runner = DummyRunner(
        [
            {
                "session_state_update": {
                    "current_code": "draft one",
                    "latest_review": inaccurate_review,
                },
                "events": [DummyEvent(True, DummyContent(text=inaccurate_review))],
            },
            {
                "session_state_update": {
                    "current_code": "draft two",
                    "latest_review": inaccurate_review,
                },
                "events": [DummyEvent(True, DummyContent(text=inaccurate_review))],
            },
        ]
    )

    result = reflection_adk_example.run_review_pipeline(
        runner,
        "Implement factorial",
        max_iterations=2,
    )

    assert result == "draft two"
    assert len(runner.run_calls) == 2
    assert runner.session_service.calls[0]["state"]["max_review_iterations"] == 2

    captured = capsys.readouterr()
    assert "ITERATION 1" in captured.out
    assert "ITERATION 2" in captured.out
    assert "ITERATION 3" not in captured.out


def test_run_review_pipeline_rejects_non_positive_max_iterations():
    runner = DummyRunner([])

    with pytest.raises(ValueError):
        reflection_adk_example.run_review_pipeline(
            runner,
            "Implement factorial",
            max_iterations=0,
        )


def test_run_review_pipeline_handles_runner_exception():
    runner = ExplodingRunner()

    result = reflection_adk_example.run_review_pipeline(runner, "Any request")

    assert "An error occurred while processing your request" in result
