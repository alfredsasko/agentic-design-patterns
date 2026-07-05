import asyncio
import importlib.util
import pathlib

import pytest


MODULE_PATH = pathlib.Path("hands_one_code_examples/4_3_reflection.py")

spec = importlib.util.spec_from_file_location("reflection_langchain_example", MODULE_PATH)
reflection_langchain_example = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(reflection_langchain_example)


class FakeChain:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def ainvoke(self, payload):
        self.calls.append(payload)
        if not self.responses:
            raise RuntimeError("No fake response configured")
        return self.responses.pop(0)


def _run(coro):
    return asyncio.run(coro)


def test_prompt_templates_enforce_reflection_contract():
    prompts = reflection_langchain_example.create_reflection_prompts()

    critique_prompt = str(prompts.critique)
    revision_prompt = str(prompts.revision)

    assert "Only list changes that are required" in critique_prompt
    assert "Do not include praise" in critique_prompt
    assert "Return valid JSON" in critique_prompt
    assert "Do not repeat critique items that were already addressed" in critique_prompt
    assert "{description_history}" in critique_prompt
    assert "Apply every required change" in revision_prompt
    assert "Do not ignore or reinterpret critique items" in revision_prompt
    assert "{description_history}" in revision_prompt


def test_run_reflection_loop_refines_until_accepted():
    chains = reflection_langchain_example.ReflectionChains(
        generation=FakeChain(["Warm coffee, from your phone."]),
        critique=FakeChain(
            [
                '{"status": "NEEDS_REVISION", "required_changes": ["Mention app temperature control."]}',
                '{"status": "ACCEPTED", "required_changes": []}',
            ]
        ),
        revision=FakeChain(["Keep coffee at your chosen temperature with app control."]),
    )

    result = _run(
        reflection_langchain_example.run_reflection_loop(
            "Smart mug with phone controls",
            chains=chains,
            max_iterations=3,
            verbose=False,
        )
    )

    assert result.final_description == "Keep coffee at your chosen temperature with app control."
    assert result.initial_description == "Warm coffee, from your phone."
    assert result.iterations_run == 2
    assert len(result.steps) == 2
    assert chains.revision.calls[0]["current_description"] == "Warm coffee, from your phone."
    assert "Mention app temperature control." in chains.revision.calls[0]["critique"]
    assert chains.critique.calls[1]["current_description"] == result.final_description


def test_run_reflection_loop_passes_review_history_to_later_iterations():
    first_critique = (
        '{"status": "NEEDS_REVISION", '
        '"required_changes": ["Add battery life.", "Remove vague wording."]}'
    )
    second_critique = (
        '{"status": "NEEDS_REVISION", '
        '"required_changes": ["Mention leak-resistant lid."]}'
    )
    chains = reflection_langchain_example.ReflectionChains(
        generation=FakeChain(["Draft v1"]),
        critique=FakeChain([first_critique, second_critique]),
        revision=FakeChain(["Draft v2", "Draft v3"]),
    )

    result = _run(
        reflection_langchain_example.run_reflection_loop(
            "Smart mug",
            chains=chains,
            max_iterations=2,
            verbose=False,
        )
    )

    assert result.final_description == "Draft v3"
    assert len(chains.critique.calls) == 2
    assert "Add battery life." in chains.critique.calls[1]["critique_history"]
    assert "Description 1: Draft v1" in chains.critique.calls[1]["description_history"]
    assert "Remove vague wording." in chains.revision.calls[1]["critique_history"]
    assert "Description 1: Draft v1" in chains.revision.calls[1]["description_history"]
    assert chains.revision.calls[1]["current_description"] == "Draft v2"


def test_run_reflection_loop_respects_max_iterations_when_not_accepted():
    critique = '{"status": "NEEDS_REVISION", "required_changes": ["Make it shorter."]}'
    chains = reflection_langchain_example.ReflectionChains(
        generation=FakeChain(["Draft v1"]),
        critique=FakeChain([critique]),
        revision=FakeChain(["Draft v2"]),
    )

    result = _run(
        reflection_langchain_example.run_reflection_loop(
            "Smart mug",
            chains=chains,
            max_iterations=1,
            verbose=False,
        )
    )

    assert result.final_description == "Draft v2"
    assert result.iterations_run == 1
    assert len(chains.critique.calls) == 1
    assert len(chains.revision.calls) == 1


def test_run_reflection_loop_does_not_revise_when_initial_draft_is_accepted():
    chains = reflection_langchain_example.ReflectionChains(
        generation=FakeChain(["Draft v1"]),
        critique=FakeChain(['{"status": "ACCEPTED", "required_changes": []}']),
        revision=FakeChain([]),
    )

    result = _run(
        reflection_langchain_example.run_reflection_loop(
            "Smart mug",
            chains=chains,
            verbose=False,
        )
    )

    assert result.final_description == "Draft v1"
    assert result.iterations_run == 1
    assert chains.revision.calls == []


def test_run_reflection_loop_rejects_invalid_max_iterations():
    chains = reflection_langchain_example.ReflectionChains(
        generation=FakeChain([]),
        critique=FakeChain([]),
        revision=FakeChain([]),
    )

    with pytest.raises(ValueError, match="positive integer"):
        _run(
            reflection_langchain_example.run_reflection_loop(
                "Smart mug",
                chains=chains,
                max_iterations=0,
                verbose=False,
            )
        )
