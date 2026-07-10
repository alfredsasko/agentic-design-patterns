from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from hands_one_code_examples._shared.adk_runtime import load_environment_variables
from hands_one_code_examples._shared.langgraph_memory import (
    build_thread_config,
    coerce_ai_message,
    extract_final_text,
    validate_openai_runtime_environment,
)


DEFAULT_MODEL = "openai:gpt-4.1-mini"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the full conversation in the current thread "
    "to answer follow-up questions and recall details the user shared earlier."
)
DEFAULT_THREAD_ID = "demo-short-term-memory"
DEFAULT_USER_TURNS = (
    "Hi, I'm Jane and I live in Bratislava.",
    "My dog's name is Pixel.",
    "Pixel loves squeaky toys and hiking trails.",
    "Next month I'm booking Pixel a vet appointment.",
    "What city did I say I live in in my first message, and what is my dog's name?",
)


class ChatModel(Protocol):
    def invoke(self, input: Any, config: Mapping[str, Any] | None = None) -> Any:
        ...


@dataclass(frozen=True)
class ConversationTurn:
    user_message: str
    assistant_message: str


def create_chat_model(model: str = DEFAULT_MODEL, temperature: float = 0.0) -> ChatModel:
    """Create a provider-backed LangChain chat model."""

    validate_openai_runtime_environment(
        model,
        example_name="short-term memory example",
    )

    return init_chat_model(model=model, temperature=temperature)


def build_short_term_memory_app(
    *,
    model: str | ChatModel = DEFAULT_MODEL,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    checkpointer: InMemorySaver | None = None,
) -> Any:
    """Build a LangGraph conversation app with thread-scoped short-term memory."""

    if not system_prompt.strip():
        raise ValueError("system_prompt must not be empty")

    active_model = create_chat_model(model) if isinstance(model, str) else model
    active_checkpointer = checkpointer or InMemorySaver()

    def call_model(state: MessagesState) -> dict[str, list[AIMessage]]:
        messages = [SystemMessage(content=system_prompt), *state["messages"]]
        response = active_model.invoke(messages)
        return {"messages": [coerce_ai_message(response)]}

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_edge(START, "call_model")
    return builder.compile(checkpointer=active_checkpointer)


@dataclass(frozen=True)
class ShortTermMemoryAssistant:
    """Small OO facade over a compiled LangGraph app."""

    app: Any
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @classmethod
    def build(
        cls,
        *,
        model: str | ChatModel = DEFAULT_MODEL,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        checkpointer: InMemorySaver | None = None,
    ) -> "ShortTermMemoryAssistant":
        app = build_short_term_memory_app(
            model=model,
            system_prompt=system_prompt,
            checkpointer=checkpointer,
        )
        return cls(app=app, system_prompt=system_prompt)

    def ask(self, thread_id: str, user_message: str) -> str:
        normalized_message = user_message.strip()
        if not normalized_message:
            raise ValueError("user_message must not be empty")

        result = self.app.invoke(
            {"messages": [HumanMessage(content=normalized_message)]},
            build_thread_config(thread_id),
        )
        return extract_final_text(result)

    def conversation_history(self, thread_id: str) -> tuple[BaseMessage, ...]:
        state = self.app.get_state(build_thread_config(thread_id))
        messages = state.values.get("messages", [])
        return tuple(messages)


def run_conversation_turns(
    assistant: ShortTermMemoryAssistant,
    *,
    thread_id: str,
    user_turns: Sequence[str],
) -> list[ConversationTurn]:
    transcript: list[ConversationTurn] = []
    for user_message in user_turns:
        assistant_message = assistant.ask(thread_id, user_message)
        transcript.append(
            ConversationTurn(
                user_message=user_message,
                assistant_message=assistant_message,
            )
        )

    return transcript


def print_transcript(transcript: Sequence[ConversationTurn]) -> None:
    for index, turn in enumerate(transcript, start=1):
        print(f"\nTurn {index} user: {turn.user_message}")
        print(f"Turn {index} assistant: {turn.assistant_message}")


def main() -> None:
    load_environment_variables()
    assistant = ShortTermMemoryAssistant.build()
    transcript = run_conversation_turns(
        assistant,
        thread_id=DEFAULT_THREAD_ID,
        user_turns=DEFAULT_USER_TURNS,
    )

    print("--- Running LangGraph Short-Term Memory Example ---")
    print_transcript(transcript)


if __name__ == "__main__":
    main()
