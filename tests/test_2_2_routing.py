import importlib.util
import pathlib


MODULE_PATH = pathlib.Path("hands_one_code_examples/2_2_routing.py")

spec = importlib.util.spec_from_file_location("adk_routing", MODULE_PATH)
adk_routing = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(adk_routing)


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


def test_booking_handler_output():
    result = adk_routing.booking_handler("Book me a hotel")
    assert "Booking action" in result


def test_info_handler_output():
    result = adk_routing.info_handler("What is the capital of Italy?")
    assert "Simulated information retrieval" in result


def test_unclear_handler_output():
    result = adk_routing.unclear_handler("???")
    assert "Please clarify" in result


def test_run_coordinator_uses_content_text_when_available():
    events = [DummyEvent(True, DummyContent(text="Final answer from coordinator"))]
    runner = DummyRunner(events)

    result = adk_routing.run_coordinator(runner, "Book me a hotel in Paris")

    assert result == "Final answer from coordinator"
    assert len(runner.session_service.calls) == 1


def test_run_coordinator_falls_back_to_parts():
    events = [
        DummyEvent(
            True,
            DummyContent(parts=[DummyPart("Hello "), DummyPart("world")]),
        )
    ]
    runner = DummyRunner(events)

    result = adk_routing.run_coordinator(runner, "Tell me something")
    assert result == "Hello world"


def test_run_coordinator_handles_runner_exception():
    runner = ExplodingRunner()

    result = adk_routing.run_coordinator(runner, "Any request")

    assert "An error occurred while processing your request" in result
