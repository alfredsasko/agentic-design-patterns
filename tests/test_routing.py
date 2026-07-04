import importlib.util
import pathlib

from langchain_core.runnables import RunnableLambda


MODULE_PATH = pathlib.Path("hands_one_code_examples/2_routing/2_1_routing.py")

spec = importlib.util.spec_from_file_location("routing_example", MODULE_PATH)
routing_example = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(routing_example)


def _decision_router(payload):
    request = payload["request"].lower()
    if "book" in request or "flight" in request or "hotel" in request:
        return "booker"
    if "capital" in request or "what is" in request:
        return "info"
    return "unclear"


def test_booking_handler_direct():
    result = routing_example.booking_handler("Book me a flight to London.")
    assert "Booking Handler processed request" in result
    assert "Simulated booking action" in result


def test_info_handler_direct():
    result = routing_example.info_handler("What is the capital of Italy?")
    assert "Info Handler processed request" in result
    assert "Simulated information retrieval" in result


def test_unclear_handler_direct():
    result = routing_example.unclear_handler("Tell me something")
    assert "Coordinator could not delegate request" in result


def test_coordinator_routes_booking_requests():
    agent = routing_example.build_coordinator_agent(RunnableLambda(_decision_router))
    result = agent.invoke({"request": "Please book a hotel in Rome."})
    assert "Booking Handler processed request" in result


def test_coordinator_routes_info_requests():
    agent = routing_example.build_coordinator_agent(RunnableLambda(_decision_router))
    result = agent.invoke({"request": "What is the capital of Japan?"})
    assert "Info Handler processed request" in result


def test_coordinator_routes_unclear_requests():
    agent = routing_example.build_coordinator_agent(RunnableLambda(_decision_router))
    result = agent.invoke({"request": "Quantum gravity?"})
    assert "Coordinator could not delegate request" in result


def test_coordinator_strips_router_whitespace():
    whitespace_router = RunnableLambda(lambda _: " booker  ")
    agent = routing_example.build_coordinator_agent(whitespace_router)
    result = agent.invoke({"request": "Book me a flight."})
    assert "Booking Handler processed request" in result
