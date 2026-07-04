import importlib.util
import pathlib


MODULE_PATH = pathlib.Path("hands_one_code_examples/3_3_parallelization.py")

spec = importlib.util.spec_from_file_location("parallelization_adk_workflow_example", MODULE_PATH)
parallelization_adk_workflow_example = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(parallelization_adk_workflow_example)


class DummySessionService:
    def __init__(self):
        self.calls = []

    def create_session(self, app_name, user_id, session_id):
        self.calls.append(
            {"app_name": app_name, "user_id": user_id, "session_id": session_id}
        )


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
    def __init__(self, events):
        self.app_name = "test_app"
        self.session_service = DummySessionService()
        self._events = events

    def run(self, **kwargs):
        _ = kwargs
        for event in self._events:
            yield event


class ExplodingRunner:
    def __init__(self):
        self.app_name = "test_app"
        self.session_service = DummySessionService()

    def run(self, **kwargs):
        _ = kwargs
        raise RuntimeError("runner failed")


def test_model_constant_is_current_supported_version():
    assert parallelization_adk_workflow_example.GEMINI_MODEL == "gemini-2.5-flash"


def test_workflow_root_agent_name():
    root = parallelization_adk_workflow_example.root_agent
    assert root.name == "ResearchAndSynthesisWorkflow"


def test_workflow_has_expected_research_and_synthesis_nodes():
    workflow = parallelization_adk_workflow_example.workflow_pipeline
    node_names = {node.name for node in workflow.graph.nodes}

    assert "RenewableEnergyResearcher" in node_names
    assert "EVResearcher" in node_names
    assert "CarbonCaptureResearcher" in node_names
    assert "JoinResearchOutputs" in node_names
    assert "SynthesisAgent" in node_names


def test_run_pipeline_uses_content_text_when_available():
    events = [DummyEvent(True, DummyContent(text="Final answer from workflow"))]
    runner = DummyRunner(events)

    result = parallelization_adk_workflow_example.run_pipeline(runner, "test topic")

    assert result == "Final answer from workflow"
    assert len(runner.session_service.calls) == 1


def test_run_pipeline_falls_back_to_parts():
    events = [DummyEvent(True, DummyContent(parts=[DummyPart("Hello "), DummyPart("workflow")]))]
    runner = DummyRunner(events)

    result = parallelization_adk_workflow_example.run_pipeline(runner, "test topic")

    assert result == "Hello workflow"


def test_run_pipeline_handles_runner_exception():
    runner = ExplodingRunner()

    result = parallelization_adk_workflow_example.run_pipeline(runner, "Any request")

    assert "An error occurred while processing your request" in result
