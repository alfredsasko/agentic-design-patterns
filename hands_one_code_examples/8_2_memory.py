from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from hands_one_code_examples._shared.adk_runtime import load_environment_variables
from hands_one_code_examples._shared.langgraph_memory import (
    build_thread_config,
    coerce_ai_message,
    content_to_text,
    extract_final_text,
    validate_openai_runtime_environment,
)


DEFAULT_MODEL = "openai:gpt-4.1-mini"
DEFAULT_USER_ID = "home-cook-user"
DEFAULT_PROFILE_KEY = "active-instructions"
DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful cooking coach.\n"
    "Follow these long-term response instructions:\n"
    "{instructions}"
)
DEFAULT_PROFILE_INSTRUCTIONS = (
    "Use short sentences."
)
DEFAULT_DEMO_TURNS = (
    ("cooking-session", "How do I make toast?"),
    ("cooking-session", "Summarize in onne line."),
    ("cooking-session", "How do I make oatmeal?"),
    ("cooking-session", "Use 3 bullets be more verbose."),
    ("cooking-session", "How do I make tea?"),
    ("cooking-session", "Use 4 bullets be concise."),
    ("cooking-session", "How do I make fruit salad?"),
)
REFLECTION_SYSTEM_PROMPT = """
You maintain procedural long-term memory for a LangGraph assistant.

Your job is to decide whether the latest user message is:
1. a normal task question that should NOT change long-term memory, or
2. a durable response preference that SHOULD update long-term memory.

Update long-term memory only for stable response preferences such as:
- one line summaries
- simpler words
- bullet counts
- concise vs verbose style

Do not update long-term memory for ordinary task requests such as:
- "How do I make oatmeal?"
- "How do I make tea?"

Normalize obvious typos in user preferences. Example:
- "Summarize in onne line." means "Summarize in one line."

When the user gives a new preference, rewrite the full instruction string so it is clean,
short, and internally consistent. Replace conflicting old preferences instead of stacking
contradictions.

Examples:
- Current: "Use short sentences."
  Latest user message: "Summarize in onne line."
  Update: yes
  New instructions: "Use short sentences. Summarize in one line. Be concise."

- Current: "Use short sentences. Summarize in one line. Be concise."
  Latest user message: "Use 3 bullets be more verbose."
  Update: yes
  New instructions: "Use 3 bullets. Be more verbose."

- Current: "Use 3 bullets. Be more verbose."
  Latest user message: "Use 4 bullets be concise."
  Update: yes
  New instructions: "Use 4 bullets. Be concise."

- Current: "Use 4 bullets. Be concise."
  Latest user message: "How do I make fruit salad?"
  Update: no
  New instructions: keep the current instructions unchanged.
""".strip()


class ChatModel(Protocol):
    def invoke(self, input: Any, config: Mapping[str, Any] | None = None) -> Any:
        ...


class PromptImprovementState(MessagesState):
    instructions_before: str
    instructions_after: str
    memory_update_summary: str
    memory_changed: bool
    memory_version: int


@dataclass(frozen=True)
class LongTermMemoryContext:
    user_id: str


class InstructionProfile(BaseModel):
    instructions: str = Field(min_length=1)
    version: int = Field(default=1, ge=1)
    last_update_summary: str = Field(default="Seeded default instructions.")
    last_user_feedback: str = Field(default="")


class InstructionReflection(BaseModel):
    should_update: bool
    summary: str = Field(min_length=1)
    new_instructions: str = Field(min_length=1)


class InstructionUpdateRecord(BaseModel):
    version: int = Field(ge=2)
    previous_instructions: str = Field(min_length=1)
    new_instructions: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    triggering_user_message: str = Field(min_length=1)


@dataclass(frozen=True)
class ImprovementTurn:
    thread_id: str
    user_message: str
    assistant_message: str
    instructions_before: str
    instructions_after: str
    memory_update_summary: str
    memory_changed: bool
    memory_version: int


@dataclass(frozen=True)
class DemoTurn:
    thread_id: str
    user_message: str


def create_chat_model(model: str = DEFAULT_MODEL, temperature: float = 0.0) -> ChatModel:
    """Create a provider-backed LangChain chat model for response generation."""

    validate_openai_runtime_environment(
        model,
        example_name="long-term memory example",
    )
    return init_chat_model(model=model, temperature=temperature)


def create_reflection_model(
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> ChatModel:
    """Create a structured-output model that maintains procedural memory."""

    validate_openai_runtime_environment(
        model,
        example_name="long-term memory example",
    )
    return init_chat_model(model=model, temperature=temperature).with_structured_output(
        InstructionReflection
    )


def build_profile_namespace(user_id: str) -> tuple[str, str, str]:
    normalized_user_id = user_id.strip()
    if not normalized_user_id:
        raise ValueError("user_id must not be empty")

    return ("long_term_memory", normalized_user_id, "instruction_profile")


def build_update_namespace(user_id: str) -> tuple[str, str, str]:
    normalized_user_id = user_id.strip()
    if not normalized_user_id:
        raise ValueError("user_id must not be empty")

    return ("long_term_memory", normalized_user_id, "instruction_updates")


def build_demo_turns() -> tuple[DemoTurn, ...]:
    return tuple(
        DemoTurn(thread_id=thread_id, user_message=user_message)
        for thread_id, user_message in DEFAULT_DEMO_TURNS
    )


def compose_system_prompt(instructions: str) -> str:
    return DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(instructions=instructions.strip())


def _message_role(message: BaseMessage) -> str:
    if isinstance(message, HumanMessage):
        return "user"
    return "assistant"


def _format_conversation(messages: Sequence[BaseMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        lines.append(f"{_message_role(message)}: {content_to_text(message.content)}")
    return "\n".join(lines).strip()


def _coerce_reflection_output(output: Any) -> InstructionReflection:
    if isinstance(output, InstructionReflection):
        return output
    if isinstance(output, Mapping):
        return InstructionReflection.model_validate(output)
    if hasattr(output, "model_dump"):
        return InstructionReflection.model_validate(output.model_dump())
    if isinstance(output, str):
        return InstructionReflection.model_validate_json(output)
    raise TypeError("Unsupported reflection model output type.")


def get_or_create_instruction_profile(
    store: BaseStore,
    *,
    user_id: str,
    default_instructions: str = DEFAULT_PROFILE_INSTRUCTIONS,
) -> InstructionProfile:
    namespace = build_profile_namespace(user_id)
    item = store.get(namespace, DEFAULT_PROFILE_KEY)
    if item is not None:
        return InstructionProfile.model_validate(item.value)

    profile = InstructionProfile(
        instructions=default_instructions.strip(),
        version=1,
        last_update_summary="Seeded default instructions.",
        last_user_feedback="",
    )
    store.put(namespace, DEFAULT_PROFILE_KEY, profile.model_dump())
    return profile


def list_instruction_updates(store: BaseStore, *, user_id: str) -> tuple[InstructionUpdateRecord, ...]:
    items = store.search(build_update_namespace(user_id), limit=100)
    records = [
        InstructionUpdateRecord.model_validate(item.value)
        for item in items
    ]
    return tuple(sorted(records, key=lambda record: record.version))


def build_long_term_memory_app(
    *,
    response_model: str | ChatModel = DEFAULT_MODEL,
    reflection_model: str | ChatModel = DEFAULT_MODEL,
    checkpointer: InMemorySaver | None = None,
    store: BaseStore | None = None,
) -> tuple[Any, BaseStore]:
    """Build a LangGraph app that improves its own prompt instructions over time."""

    active_response_model = (
        create_chat_model(response_model)
        if isinstance(response_model, str)
        else response_model
    )
    active_reflection_model = (
        create_reflection_model(reflection_model)
        if isinstance(reflection_model, str)
        else reflection_model
    )
    active_checkpointer = checkpointer or InMemorySaver()
    active_store = store or InMemoryStore()

    def respond(
        state: PromptImprovementState,
        runtime: Runtime[LongTermMemoryContext],
    ) -> dict[str, Any]:
        profile = get_or_create_instruction_profile(
            runtime.store,
            user_id=runtime.context.user_id,
        )
        messages = [
            SystemMessage(content=compose_system_prompt(profile.instructions)),
            *state["messages"],
        ]
        response = active_response_model.invoke(messages)
        return {
            "messages": [coerce_ai_message(response)],
            "instructions_before": profile.instructions,
            "memory_version": profile.version,
        }

    def reflect_and_update_memory(
        state: PromptImprovementState,
        runtime: Runtime[LongTermMemoryContext],
    ) -> dict[str, Any]:
        profile = get_or_create_instruction_profile(
            runtime.store,
            user_id=runtime.context.user_id,
        )
        latest_user_message = next(
            content_to_text(message.content)
            for message in reversed(state["messages"])
            if isinstance(message, HumanMessage)
        )

        reflection_prompt = (
            f"Current instructions:\n{profile.instructions}\n\n"
            f"Latest user message:\n{latest_user_message}\n\n"
            f"Conversation:\n{_format_conversation(state['messages'])}"
        )
        reflection = _coerce_reflection_output(
            active_reflection_model.invoke(
                [
                    SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
                    HumanMessage(content=reflection_prompt),
                ]
            )
        )

        if not reflection.should_update:
            return {
                "instructions_after": profile.instructions,
                "memory_update_summary": reflection.summary,
                "memory_changed": False,
                "memory_version": profile.version,
            }

        updated_profile = InstructionProfile(
            instructions=reflection.new_instructions.strip(),
            version=profile.version + 1,
            last_update_summary=reflection.summary.strip(),
            last_user_feedback=latest_user_message,
        )
        runtime.store.put(
            build_profile_namespace(runtime.context.user_id),
            DEFAULT_PROFILE_KEY,
            updated_profile.model_dump(),
        )
        runtime.store.put(
            build_update_namespace(runtime.context.user_id),
            f"update-{updated_profile.version:03d}",
            InstructionUpdateRecord(
                version=updated_profile.version,
                previous_instructions=profile.instructions,
                new_instructions=updated_profile.instructions,
                summary=reflection.summary.strip(),
                triggering_user_message=latest_user_message,
            ).model_dump(),
        )
        return {
            "instructions_after": updated_profile.instructions,
            "memory_update_summary": reflection.summary,
            "memory_changed": True,
            "memory_version": updated_profile.version,
        }

    builder = StateGraph(PromptImprovementState, context_schema=LongTermMemoryContext)
    builder.add_node("respond", respond)
    builder.add_node("reflect_and_update_memory", reflect_and_update_memory)
    builder.add_edge(START, "respond")
    builder.add_edge("respond", "reflect_and_update_memory")
    builder.add_edge("reflect_and_update_memory", END)
    return (
        builder.compile(checkpointer=active_checkpointer, store=active_store),
        active_store,
    )


@dataclass(frozen=True)
class PromptImprovementAssistant:
    """OO facade over a LangGraph app that stores procedural memory in a long-term store."""

    app: Any
    store: BaseStore
    user_id: str = DEFAULT_USER_ID

    @classmethod
    def build(
        cls,
        *,
        user_id: str = DEFAULT_USER_ID,
        response_model: str | ChatModel = DEFAULT_MODEL,
        reflection_model: str | ChatModel = DEFAULT_MODEL,
        checkpointer: InMemorySaver | None = None,
        store: BaseStore | None = None,
    ) -> "PromptImprovementAssistant":
        app, active_store = build_long_term_memory_app(
            response_model=response_model,
            reflection_model=reflection_model,
            checkpointer=checkpointer,
            store=store,
        )
        return cls(app=app, store=active_store, user_id=user_id)

    def ask(self, thread_id: str, user_message: str) -> ImprovementTurn:
        normalized_message = user_message.strip()
        if not normalized_message:
            raise ValueError("user_message must not be empty")

        result = self.app.invoke(
            {"messages": [HumanMessage(content=normalized_message)]},
            config=build_thread_config(thread_id),
            context=LongTermMemoryContext(user_id=self.user_id),
        )
        return ImprovementTurn(
            thread_id=thread_id,
            user_message=normalized_message,
            assistant_message=extract_final_text(result),
            instructions_before=result["instructions_before"],
            instructions_after=result["instructions_after"],
            memory_update_summary=result["memory_update_summary"],
            memory_changed=bool(result["memory_changed"]),
            memory_version=int(result["memory_version"]),
        )

    def conversation_history(self, thread_id: str) -> tuple[BaseMessage, ...]:
        state = self.app.get_state(
            build_thread_config(thread_id),
            context=LongTermMemoryContext(user_id=self.user_id),
        )
        return tuple(state.values.get("messages", []))

    def instruction_profile(self) -> InstructionProfile:
        return get_or_create_instruction_profile(self.store, user_id=self.user_id)

    def instruction_update_history(self) -> tuple[InstructionUpdateRecord, ...]:
        return list_instruction_updates(self.store, user_id=self.user_id)


def run_demo_turns(
    assistant: PromptImprovementAssistant,
    *,
    turns: Sequence[DemoTurn],
) -> list[ImprovementTurn]:
    return [
        assistant.ask(turn.thread_id, turn.user_message)
        for turn in turns
    ]


def print_improvement_report(
    turns: Sequence[ImprovementTurn],
    assistant: PromptImprovementAssistant,
) -> None:
    print("--- Running LangGraph Long-Term Memory Example ---")
    for index, turn in enumerate(turns, start=1):
        print(f"\nLast Instructions: {turn.instructions_before}")
        print(f"Turn {index}")
        print(f"User Question: {turn.user_message}")
        print(f"Agent Answer: {turn.assistant_message}")
        if turn.memory_changed:
            print(f"User Instructions: {turn.user_message}")
        else:
            print("User Instructions: -")
        print(f"Updated Instructions: {turn.instructions_after}")


def main() -> None:
    load_environment_variables()
    assistant = PromptImprovementAssistant.build()
    turns = run_demo_turns(assistant, turns=build_demo_turns())
    print_improvement_report(turns, assistant)


if __name__ == "__main__":
    main()
