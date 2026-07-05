import asyncio
import os
from typing import Annotated, Literal
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# --- 1. Environment & Setup ---
def load_environment_variables() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dotenv_path = project_root / ".env"
    load_dotenv(dotenv_path=dotenv_path)

load_environment_variables()

def create_llm(model: str = "gpt-4o-mini", temperature: float = 0.2) -> ChatOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found. Set it before running this example.")
    return ChatOpenAI(model=model, temperature=temperature)


# --- 2. Define Structured Output Schemas ---
class CritiqueDecision(BaseModel):
    """The structured evaluation response from the critique node."""
    status: Literal["ACCEPTED", "NEEDS_REVISION"] = Field(
        description="ACCEPTED if the description is perfect, NEEDS_REVISION if changes are required."
    )
    required_changes: list[str] = Field(
        default=[],
        description="A list of specific, actionable changes required to improve the description."
    )


# --- 3. Define LangGraph State ---
class ReflectionState(BaseModel):
    """The global graph state shared across all nodes."""
    product_details: str
    current_description: str = ""
    description_history: list[str] = Field(default_factory=list)
    critique_history: list[str] = Field(default_factory=list)
    latest_critique: CritiqueDecision = Field(default_factory=lambda: CritiqueDecision(status="NEEDS_REVISION", required_changes=[]))
    iterations_run: int = 0
    max_iterations: int = 3


# --- 4. Define Graph Nodes (The Agents) ---

async def generation_node(state: ReflectionState) -> dict:
    """The Producer: Creates the initial product description."""
    llm = create_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You write concise product descriptions. Create one strong first draft from the product details. Return only the description."),
        ("user", "Product details:\n{product_details}")
    ])
    
    chain = prompt | llm
    response = await chain.ainvoke({"product_details": state.product_details})
    
    print(f"\n[Producer] Initial Draft Generated.")
    return {
        "current_description": response.content,
        "iterations_run": state.iterations_run + 1
    }


async def critique_node(state: ReflectionState) -> dict:
    """The Critique: Evaluates the current description against historical context."""
    # Bind the structured output schema directly to the model
    llm = create_llm().with_structured_output(CritiqueDecision)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are the reflection reviewer for a product-description writer. "
            "Only list changes that are required to improve clarity, concision, specificity, or customer appeal. "
            "Compare the current description against previous critiques and history. "
            "Do not repeat items already addressed. Only surface unresolved issues."
        )),
        ("user", (
            "Product details:\n{product_details}\n\n"
            "Current description:\n{current_description}\n\n"
            "Previous description versions:\n{description_history}\n\n"
            "Previous critiques:\n{critique_history}"
        ))
    ])
    
    chain = prompt | llm
    decision: CritiqueDecision = await chain.ainvoke({
        "product_details": state.product_details,
        "current_description": state.current_description,
        "description_history": "\n".join(state.description_history),
        "critique_history": "\n".join(state.critique_history)
    })
    
    # Format the critique text for history tracking
    critique_summary = f"Iteration {state.iterations_run}: Status={decision.status}, Changes={decision.required_changes}"
    print(f"[Critique] Analysis: {decision.status} | Changes Required: {len(decision.required_changes)}")
    
    return {
        "latest_critique": decision,
        "critique_history": state.critique_history + [critique_summary],
        "description_history": state.description_history + [state.current_description]
    }


async def revision_node(state: ReflectionState) -> dict:
    """The Reviser: Updates the description based on the critique feedback."""
    llm = create_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You revise product descriptions. Apply every required change from the latest critique. "
            "Do not ignore or reinterpret critique items. Preserve accurate original details. "
            "Return only the revised product description."
        )),
        ("user", (
            "Product details:\n{product_details}\n\n"
            "Current description:\n{current_description}\n\n"
            "Latest critique changes:\n{latest_changes}\n\n"
            "Previous description versions:\n{description_history}\n\n"
            "Previous critiques:\n{critique_history}"
        ))
    ])
    
    chain = prompt | llm
    response = await chain.ainvoke({
        "product_details": state.product_details,
        "current_description": state.current_description,
        "latest_changes": "\n".join(state.latest_critique.required_changes),
        "description_history": "\n".join(state.description_history),
        "critique_history": "\n".join(state.critique_history)
    })
    
    print(f"[Producer] Revised Draft Generated for Iteration {state.iterations_run + 1}.")
    return {
        "current_description": response.content,
        "iterations_run": state.iterations_run + 1
    }


# --- 5. Define Routing Logic (Conditional Edges) ---

def should_continue(state: ReflectionState) -> Literal["revision", "__end__"]:
    """Determines whether to loop back for another revision or stop execution."""
    if state.latest_critique.status == "ACCEPTED":
        print("\n[Router] Critique ACCEPTED the description. Ending graph workflow.")
        return END
        
    if state.iterations_run >= state.max_iterations:
        print(f"\n[Router] Hit maximum iteration ceiling ({state.max_iterations}). Ending graph workflow.")
        return END
        
    print(f"[Router] Routing to Revision Node.")
    return "revision"


# --- 6. Build and Compile the Graph ---

def build_reflection_graph() -> StateGraph:
    # Initialize the graph framework with our explicitly typed State schema
    workflow = StateGraph(ReflectionState)
    
    # Register the architectural nodes
    workflow.add_node("generation", generation_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("revision", revision_node)
    
    # Establish structural flow edges
    workflow.add_edge(START, "generation")
    workflow.add_edge("generation", "critique")
    
    # Establish conditional routing loop out of the critique stage
    workflow.add_conditional_edges(
        "critique",
        should_continue,
        {
            "revision": "revision",
            END: END
        }
    )
    
    # Link revision back to critique to complete the loop
    workflow.add_edge("revision", "critique")
    
    return workflow.compile()


# --- 7. Execution Entrypoint ---

async def main():
    test_details = "Ergonomic Mechanical Keyboard, RGB backlighting, Cherry MX Brown switches, wireless 2.4Ghz & Bluetooth, aluminum frame, 4000mAh battery."
    
    # Build graph object
    app = build_reflection_graph()
    
    # Initialize state inputs
    initial_state = {
        "product_details": test_details,
        "max_iterations": 3
    }
    
    print("--- Launching LangGraph Agentic Reflection Pipeline ---")
    final_output = await app.ainvoke(initial_state)
    
    print("\n================ FINAL RESPONSE ================")
    print(final_output["current_description"])
    print("================================================")

if __name__ == "__main__":
    asyncio.run(main())
