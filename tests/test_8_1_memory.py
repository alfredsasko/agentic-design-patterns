import importlib.util
import pathlib
import re
import sys

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver


MODULE_PATH = pathlib.Path("hands_one_code_examples/8_1_memory.py")

spec = importlib.util.spec_from_file_location("memory_langgraph_example", MODULE_PATH)
memory_langgraph_example = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = memory_langgraph_example
spec.loader.exec_module(memory_langgraph_example)


class RuleBasedMemoryModel:
    def __init__(self):
        self.calls = []

    def invoke(self, messages, config=None):
        self.calls.append([message.content for message in messages])
        human_messages = [message for message in messages if isinstance(message, HumanMessage)]
        latest_user_message = human_messages[-1].content.lower()

        if "what city did i say" in latest_user_message:
            intro_message = human_messages[0].content
            dog_message = next(
                message.content for message in human_messages if "dog's name is" in message.content
            )
            city = re.search(r"live in ([A-Za-z]+)", intro_message)
            dog_name = re.search(r"dog's name is ([A-Za-z]+)", dog_message)
            return AIMessage(
                content=(
                    f"You said you live in {city.group(1)} and your dog's name is "
                    f"{dog_name.group(1)}."
                )
            )

        if "booking" in latest_user_message:
            return AIMessage(content="Understood. I noted the upcoming vet appointment.")

        if "squeaky toys" in latest_user_message:
            return AIMessage(content="Pixel sounds energetic.")

        if "dog's name is" in latest_user_message:
            return AIMessage(content="Noted. Your dog's name is Pixel.")

        return AIMessage(content="Nice to meet you, Jane.")


def test_create_chat_model_requires_openai_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY not found"):
        memory_langgraph_example.create_chat_model()


def test_build_thread_config_rejects_blank_thread_ids():
    with pytest.raises(ValueError, match="thread_id must not be empty"):
        memory_langgraph_example.build_thread_config("   ")


def test_extract_final_text_supports_messages_and_content_blocks():
    message_result = {"messages": [AIMessage(content="Bratislava")]}
    block_result = {"messages": [{"content": [{"type": "text", "text": "Pixel"}]}]}

    assert memory_langgraph_example.extract_final_text(message_result) == "Bratislava"
    assert memory_langgraph_example.extract_final_text(block_result) == "Pixel"


def test_run_conversation_turns_recalls_information_from_turn_one():
    assistant = memory_langgraph_example.ShortTermMemoryAssistant.build(
        model=RuleBasedMemoryModel(),
        checkpointer=InMemorySaver(),
    )

    transcript = memory_langgraph_example.run_conversation_turns(
        assistant,
        thread_id="jane-thread",
        user_turns=memory_langgraph_example.DEFAULT_USER_TURNS,
    )

    assert [turn.user_message for turn in transcript] == list(
        memory_langgraph_example.DEFAULT_USER_TURNS
    )
    assert transcript[-1].assistant_message == (
        "You said you live in Bratislava and your dog's name is Pixel."
    )

    history = assistant.conversation_history("jane-thread")
    assert len(history) == 10
    assert history[0].content == "Hi, I'm Jane and I live in Bratislava."
    assert history[-1].content == "You said you live in Bratislava and your dog's name is Pixel."


def test_short_term_memory_is_isolated_per_thread():
    assistant = memory_langgraph_example.ShortTermMemoryAssistant.build(
        model=RuleBasedMemoryModel(),
        checkpointer=InMemorySaver(),
    )

    jane_reply = assistant.ask("thread-jane", "Hi, I'm Jane and I live in Bratislava.")
    mark_reply = assistant.ask("thread-mark", "Hi, I'm Mark and I live in Prague.")
    assistant.ask("thread-jane", "My dog's name is Pixel.")
    assistant.ask("thread-mark", "My dog's name is Nova.")

    jane_result = assistant.ask(
        "thread-jane",
        "What city did I say I live in in my first message, and what is my dog's name?",
    )
    mark_result = assistant.ask(
        "thread-mark",
        "What city did I say I live in in my first message, and what is my dog's name?",
    )

    assert jane_reply == "Nice to meet you, Jane."
    assert mark_reply == "Nice to meet you, Jane."
    assert jane_result == "You said you live in Bratislava and your dog's name is Pixel."
    assert mark_result == "You said you live in Prague and your dog's name is Nova."


def test_assistant_rejects_blank_user_messages():
    assistant = memory_langgraph_example.ShortTermMemoryAssistant.build(
        model=RuleBasedMemoryModel(),
        checkpointer=InMemorySaver(),
    )

    with pytest.raises(ValueError, match="user_message must not be empty"):
        assistant.ask("thread-1", "   ")
