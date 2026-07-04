import importlib.util
import pathlib


MODULE_PATH = pathlib.Path("hands_one_code_examples/3_2_parallelization.py")

spec = importlib.util.spec_from_file_location("parallelization_adk_example", MODULE_PATH)
parallelization_adk_example = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(parallelization_adk_example)


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
    assert parallelization_adk_example.GEMINI_MODEL == "gemini-2.5-flash"


def test_parallel_agent_has_three_research_sub_agents():
    agent = parallelization_adk_example.parallel_research_agent
    assert agent.name == "ParallelWebResearchAgent"
    assert len(agent.sub_agents) == 3


def test_researcher_output_keys_are_defined_for_state_passing():
    sub_agents = parallelization_adk_example.parallel_research_agent.sub_agents
    output_keys = {sub_agent.output_key for sub_agent in sub_agents}

    assert output_keys == {
        "renewable_energy_result",
        "ev_technology_result",
        "carbon_capture_result",
    }


def test_sequential_pipeline_order_is_parallel_then_merger():
    pipeline = parallelization_adk_example.sequential_pipeline_agent
    assert pipeline.name == "ResearchAndSynthesisPipeline"
    assert len(pipeline.sub_agents) == 2
    assert pipeline.sub_agents[0].name == "ParallelWebResearchAgent"
    assert pipeline.sub_agents[1].name == "SynthesisAgent"


def test_root_agent_points_to_sequential_pipeline():
    assert parallelization_adk_example.root_agent is parallelization_adk_example.sequential_pipeline_agent


def test_run_pipeline_uses_content_text_when_available():
    events = [DummyEvent(True, DummyContent(text="Final answer from pipeline"))]
    runner = DummyRunner(events)

    result = parallelization_adk_example.run_pipeline(runner, "test topic")

    assert result == "Final answer from pipeline"
    assert len(runner.session_service.calls) == 1


def test_run_pipeline_falls_back_to_parts():
    events = [DummyEvent(True, DummyContent(parts=[DummyPart("Hello "), DummyPart("world")]))]
    runner = DummyRunner(events)

    result = parallelization_adk_example.run_pipeline(runner, "test topic")

    assert result == "Hello world"


def test_run_pipeline_handles_runner_exception():
    runner = ExplodingRunner()

    result = parallelization_adk_example.run_pipeline(runner, "Any request")

    assert "An error occurred while processing your request" in result
