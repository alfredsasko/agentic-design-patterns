import importlib.util
import pathlib


MODULE_PATH = pathlib.Path("hands_one_code_examples/4_1_reflection.py")

spec = importlib.util.spec_from_file_location("reflection_example", MODULE_PATH)
reflection_example = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(reflection_example)


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, responses):
        self._responses = [_FakeResponse(r) for r in responses]
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if not self._responses:
            raise RuntimeError("No more fake responses configured")
        return self._responses.pop(0)


def test_run_reflection_loop_stops_early_when_code_is_perfect(capsys):
    fake_llm = _FakeLLM(
        [
            "def calculate_factorial(n):\n    return 1",
            "CODE_IS_PERFECT",
        ]
    )

    final_code = reflection_example.run_reflection_loop(llm_client=fake_llm, max_iterations=3)

    assert final_code == "def calculate_factorial(n):\n    return 1"
    assert len(fake_llm.calls) == 2

    captured = capsys.readouterr()
    assert "No further critiques found. The code is satisfactory." in captured.out


def test_run_reflection_loop_refines_until_second_iteration(capsys):
    fake_llm = _FakeLLM(
        [
            "def calculate_factorial(n):\n    return n",
            "- Missing edge-case handling for 0\n- Missing ValueError for negatives",
            "def calculate_factorial(n):\n    if n < 0:\n        raise ValueError('Negative not allowed')\n    if n == 0:\n        return 1\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result",
            "CODE_IS_PERFECT",
        ]
    )

    final_code = reflection_example.run_reflection_loop(llm_client=fake_llm, max_iterations=3)

    assert "raise ValueError" in final_code
    assert "if n == 0" in final_code
    assert len(fake_llm.calls) == 4

    captured = capsys.readouterr()
    assert "REFLECTION LOOP: ITERATION 2" in captured.out


def test_create_llm_raises_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    try:
        reflection_example.create_llm()
        raised = False
    except ValueError as exc:
        raised = True
        assert "OPENAI_API_KEY not found" in str(exc)

    assert raised is True
