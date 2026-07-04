import importlib.util
import asyncio
import os
import pathlib

MODULE_PATH = pathlib.Path("hands_one_code_examples/3_1_parallelization.py")

os.environ.setdefault("OPENAI_API_KEY", "test-key")

spec = importlib.util.spec_from_file_location("parallelization_example", MODULE_PATH)
parallelization_example = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(parallelization_example)


class _FakeChain:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def ainvoke(self, topic):
        self.calls.append(topic)
        return self.response


def test_run_parallel_example_prints_final_response_when_llm_initialized(monkeypatch, capsys):
    fake_chain = _FakeChain("Synthesized answer")

    monkeypatch.setattr(parallelization_example, "llm", object())
    monkeypatch.setattr(parallelization_example, "full_parallel_chain", fake_chain)

    topic = "The history of space exploration"
    asyncio.run(parallelization_example.run_parallel_example(topic))

    captured = capsys.readouterr()

    assert fake_chain.calls == [topic]
    assert "Running Parallel LangChain Example" in captured.out
    assert "Final Response" in captured.out
    assert "Synthesized answer" in captured.out


def test_run_parallel_example_returns_early_when_llm_not_initialized(monkeypatch, capsys):
    fake_chain = _FakeChain("Should not be used")

    monkeypatch.setattr(parallelization_example, "llm", None)
    monkeypatch.setattr(parallelization_example, "full_parallel_chain", fake_chain)

    asyncio.run(parallelization_example.run_parallel_example("Any topic"))

    captured = capsys.readouterr()

    assert fake_chain.calls == []
    assert "LLM not initialized. Cannot run example." in captured.out


def test_run_parallel_example_handles_chain_exception(monkeypatch, capsys):
    class _FailingChain:
        async def ainvoke(self, _topic):
            raise RuntimeError("chain exploded")

    monkeypatch.setattr(parallelization_example, "llm", object())
    monkeypatch.setattr(parallelization_example, "full_parallel_chain", _FailingChain())

    asyncio.run(parallelization_example.run_parallel_example("Topic"))

    captured = capsys.readouterr()
    assert "An error occurred during chain execution: chain exploded" in captured.out
