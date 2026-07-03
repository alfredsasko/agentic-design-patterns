import importlib.util
import pathlib

from langchain_core.runnables import RunnableLambda


MODULE_PATH = pathlib.Path(
    "hands_one_code_examples/1_prompt_chaning/1_1_prompt_chaining.py"
)


spec = importlib.util.spec_from_file_location("prompt_chaining", MODULE_PATH)
prompt_chaining = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(prompt_chaining)


def _fake_llm_fn(prompt_value):
    prompt_text = (
        prompt_value.to_string()
        if hasattr(prompt_value, "to_string")
        else str(prompt_value)
    )

    if "Extract the technical specifications" in prompt_text:
        return "CPU: 3.5 GHz octa-core processor; Memory: 16GB RAM; Storage: 1TB NVMe SSD"

    if "Transform the following specifications" in prompt_text:
        return '{"cpu": "3.5 GHz octa-core processor", "memory": "16GB RAM", "storage": "1TB NVMe SSD"}'

    return ""


def test_validate_json_structure_success():
    ok, parsed, message = prompt_chaining.validate_json_structure(
        '{"cpu":"3.5 GHz","memory":"16GB","storage":"1TB NVMe SSD"}'
    )

    assert ok is True
    assert parsed["cpu"] == "3.5 GHz"
    assert message == "JSON structure is valid."


def test_validate_json_structure_missing_key():
    ok, parsed, message = prompt_chaining.validate_json_structure(
        '{"cpu":"3.5 GHz","memory":"16GB"}'
    )

    assert ok is False
    assert parsed is not None
    assert "Unexpected keys" in message


def test_validate_expected_values_success():
    parsed = {
        "cpu": "3.5 GHz octa-core processor",
        "memory": "16GB RAM",
        "storage": "1TB NVMe SSD",
    }
    expected = {
        "cpu": "3.5 GHz octa-core processor",
        "memory": "16GB",
        "storage": "1TB NVMe SSD",
    }

    ok, message = prompt_chaining.validate_expected_values(parsed, expected)
    assert ok is True
    assert message == "Values look correct."


def test_full_chain_returns_expected_json_shape_and_values():
    fake_llm = RunnableLambda(_fake_llm_fn)
    chain = prompt_chaining.build_chain(fake_llm)

    input_text = (
        "The new laptop model features a 3.5 GHz octa-core processor, "
        "16GB of RAM, and a 1TB NVMe SSD."
    )

    result = chain.invoke({"text_input": input_text})
    structure_ok, parsed, _ = prompt_chaining.validate_json_structure(result)

    assert structure_ok is True
    assert parsed is not None

    values_ok, values_message = prompt_chaining.validate_expected_values(
        parsed,
        {
            "cpu": "3.5 GHz octa-core processor",
            "memory": "16GB",
            "storage": "1TB NVMe SSD",
        },
    )
    assert values_ok is True, values_message
