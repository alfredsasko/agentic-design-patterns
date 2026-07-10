from __future__ import annotations

import pathlib
import re

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from tests._support import load_module_from_path


MODULE_PATH = pathlib.Path("hands_one_code_examples/8_2_memory.py")
memory_long_term_example = load_module_from_path("memory_long_term_example", MODULE_PATH)


class RuleBasedResponseModel:
    def invoke(self, messages, config=None):
        system_prompt = messages[0].content if isinstance(messages[0], SystemMessage) else ""
        latest_user_message = next(
            message.content for message in reversed(messages) if isinstance(message, HumanMessage)
        )
        normalized_user_message = latest_user_message.lower()
        use_one_line = "summarize in one line" in system_prompt.lower()
        use_simple_words = "simple words" in system_prompt.lower()
        bullet_match = re.search(r"use (\d+) bullets", system_prompt.lower())
        bullet_count = int(bullet_match.group(1)) if bullet_match else None
        use_bullets = bullet_count is not None

        if "one line" in normalized_user_message or "onne line" in normalized_user_message:
            return AIMessage(content="Understood. I will answer in one line.")

        if "simple words" in normalized_user_message:
            return AIMessage(content="Understood. I will use simple words.")

        if "bullets" in normalized_user_message:
            return AIMessage(content="Understood. I will use bullets.")

        topic_match = re.search(r"(?:how do i make|explain) (.+?)(?:[?.]|$)", latest_user_message, flags=re.I)
        topic = topic_match.group(1) if topic_match else "the topic"
        action_word = "put" if use_simple_words else "place"
        heat_word = "hot" if use_simple_words else "heated"

        if use_bullets:
            detail_word = "more detail" if "be more verbose" in system_prompt.lower() else "less detail"
            return AIMessage(
                content=(
                    "\n".join(
                        [
                            f"{index}. {text}"
                            for index, text in enumerate(
                                [
                                    f"{action_word.title()} {topic} parts together.",
                                    f"Add {heat_word} water or heat if needed.",
                                    f"Finish and serve with {detail_word}.",
                                    "Add extras if you want.",
                                ][:bullet_count],
                                start=1,
                            )
                        ]
                    )
                )
            )

        if use_one_line:
            return AIMessage(
                content=f"Mix {topic}, cook until ready, and serve."
            )

        if not use_simple_words:
            return AIMessage(
                content=(
                    f"To make {topic}, prepare the ingredients, cook gently, and serve."
                )
            )

        return AIMessage(
            content=f"Put {topic} together. Cook if needed. Eat."
        )


class RuleBasedReflectionModel:
    def invoke(self, messages, config=None):
        reflection_prompt = messages[-1].content
        current_instructions = reflection_prompt.split("Current instructions:\n", 1)[1].split(
            "\n\nLatest user message:",
            1,
        )[0].strip()
        latest_user_message = reflection_prompt.split("Latest user message:\n", 1)[1].split(
            "\n\nConversation:",
            1,
        )[0].strip().lower().replace("onne", "one")

        if "one line" in latest_user_message and "summarize in one line" not in current_instructions.lower():
            return memory_long_term_example.InstructionReflection(
                should_update=True,
                summary="Saved a one-line preference.",
                new_instructions="Use short sentences. Summarize in one line. Be concise.",
            )

        if "simple words" in latest_user_message and "simple words" not in current_instructions.lower():
            return memory_long_term_example.InstructionReflection(
                should_update=True,
                summary="Saved a simple-words preference.",
                new_instructions="Use short sentences. Summarize in one line. Use simple words. Be concise.",
            )

        if "3 bullets" in latest_user_message and "use 3 bullets" not in current_instructions.lower():
            return memory_long_term_example.InstructionReflection(
                should_update=True,
                summary="Saved a three-bullets preference.",
                new_instructions="Use short sentences. Use 3 bullets. Be more verbose.",
            )

        if "4 bullets" in latest_user_message and "use 4 bullets" not in current_instructions.lower():
            return memory_long_term_example.InstructionReflection(
                should_update=True,
                summary="Saved a four-bullets preference.",
                new_instructions="Use short sentences. Use 4 bullets. Be concise.",
            )

        return memory_long_term_example.InstructionReflection(
            should_update=False,
            summary="No durable instruction update detected.",
            new_instructions=current_instructions,
        )


def build_assistant(user_id: str = "learner-1"):
    return memory_long_term_example.PromptImprovementAssistant.build(
        user_id=user_id,
        response_model=RuleBasedResponseModel(),
        reflection_model=RuleBasedReflectionModel(),
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
    )


def test_create_chat_model_requires_openai_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY not found"):
        memory_long_term_example.create_chat_model()


def test_get_or_create_instruction_profile_seeds_default_value():
    profile = memory_long_term_example.get_or_create_instruction_profile(
        InMemoryStore(),
        user_id="learner-1",
    )

    assert profile.instructions == memory_long_term_example.DEFAULT_PROFILE_INSTRUCTIONS
    assert profile.version == 1
    assert profile.last_update_summary == "Seeded default instructions."


def test_long_term_memory_updates_and_applies_instructions_across_threads():
    assistant = build_assistant()

    first_turn = assistant.ask("thread-1", "How do I make oatmeal for breakfast?")
    preference_turn = assistant.ask(
        "thread-1",
        "Summarize in onne line.",
    )
    question_turn = assistant.ask("thread-2", "How do I make tea?")
    reused_turn = assistant.ask("thread-2", "Use simple words.")

    assert first_turn.memory_changed is False
    assert preference_turn.memory_changed is True
    assert "summarize in one line" in preference_turn.instructions_after.lower()
    assert question_turn.thread_id == "thread-2"
    assert question_turn.instructions_before == preference_turn.instructions_after
    assert question_turn.memory_changed is False
    assert question_turn.instructions_after == preference_turn.instructions_after
    assert reused_turn.thread_id == "thread-2"
    assert reused_turn.instructions_before == preference_turn.instructions_after
    assert reused_turn.memory_changed is True
    assert "simple words" in assistant.instruction_profile().instructions.lower()
    assert "simple words" in reused_turn.assistant_message.lower()


def test_instruction_update_history_tracks_durable_changes_only():
    assistant = build_assistant()

    assistant.ask("thread-1", "How do I make oatmeal for breakfast?")
    assistant.ask(
        "thread-1",
        "Summarize in onne line.",
    )
    assistant.ask("thread-2", "How do I make tea?")
    assistant.ask("thread-2", "Use simple words.")
    assistant.ask("thread-2", "Use 3 bullets. Be more verbose.")
    assistant.ask("thread-3", "Use 4 bullets be concise.")
    final_turn = assistant.ask("thread-3", "How do I make fruit salad?")

    updates = assistant.instruction_update_history()

    assert [record.version for record in updates] == [2, 3, 4, 5]
    assert "summarize in one line" in updates[0].new_instructions.lower()
    assert "simple words" in updates[1].new_instructions.lower()
    assert "use 3 bullets" in updates[2].new_instructions.lower()
    assert "use 4 bullets" in updates[3].new_instructions.lower()
    assert "4." in final_turn.assistant_message


def test_print_improvement_report_displays_memory_state_and_updates(capsys):
    assistant = build_assistant()
    turns = memory_long_term_example.run_demo_turns(
        assistant,
        turns=(
            memory_long_term_example.DemoTurn(
                thread_id="thread-1",
                user_message="How do I make toast?",
            ),
            memory_long_term_example.DemoTurn(
                thread_id="thread-1",
                user_message="Summarize in onne line.",
            ),
            memory_long_term_example.DemoTurn(
                thread_id="thread-1",
                user_message="How do I make oatmeal?",
            ),
            memory_long_term_example.DemoTurn(
                thread_id="thread-1",
                user_message="Use 3 bullets. Be more verbose.",
            ),
        ),
    )

    memory_long_term_example.print_improvement_report(turns, assistant)
    output = capsys.readouterr().out

    assert "Running LangGraph Long-Term Memory Example" in output
    assert "Last Instructions: Use short sentences." in output
    assert "Turn 2" in output
    assert "User Question: Summarize in onne line." in output
    assert "User Instructions: Summarize in onne line." in output
    assert "Updated Instructions: Use short sentences. Summarize in one line. Be concise." in output
    assert "Last Instructions: Use short sentences. Summarize in one line. Be concise." in output
    assert "User Instructions: -" in output
    assert "User Instructions: Use 3 bullets. Be more verbose." in output


def test_assistant_rejects_blank_user_messages():
    assistant = build_assistant()

    with pytest.raises(ValueError, match="user_message must not be empty"):
        assistant.ask("thread-1", "   ")
