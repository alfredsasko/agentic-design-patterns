import asyncio
import inspect
import uuid
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types


def load_environment_variables() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dotenv_path = project_root / ".env"
    load_dotenv(dotenv_path=dotenv_path)


load_environment_variables()

GEMINI_MODEL = "gemini-2.5-flash"


researcher_agent_1 = LlmAgent(
    name="RenewableEnergyResearcher",
    model=GEMINI_MODEL,
    instruction=(
        "You are an AI Research Assistant specializing in energy.\n"
        "Research the latest advancements in 'renewable energy sources'.\n"
        "Use the Google Search tool provided.\n"
        "Summarize your key findings concisely (1-2 sentences).\n"
        "Output only the summary."
    ),
    description="Researches renewable energy sources.",
    tools=[google_search],
    output_key="renewable_energy_result",
)

researcher_agent_2 = LlmAgent(
    name="EVResearcher",
    model=GEMINI_MODEL,
    instruction=(
        "You are an AI Research Assistant specializing in transportation.\n"
        "Research the latest developments in 'electric vehicle technology'.\n"
        "Use the Google Search tool provided.\n"
        "Summarize your key findings concisely (1-2 sentences).\n"
        "Output only the summary."
    ),
    description="Researches electric vehicle technology.",
    tools=[google_search],
    output_key="ev_technology_result",
)

researcher_agent_3 = LlmAgent(
    name="CarbonCaptureResearcher",
    model=GEMINI_MODEL,
    instruction=(
        "You are an AI Research Assistant specializing in climate solutions.\n"
        "Research the current state of 'carbon capture methods'.\n"
        "Use the Google Search tool provided.\n"
        "Summarize your key findings concisely (1-2 sentences).\n"
        "Output only the summary."
    ),
    description="Researches carbon capture methods.",
    tools=[google_search],
    output_key="carbon_capture_result",
)

parallel_research_agent = ParallelAgent(
    name="ParallelWebResearchAgent",
    sub_agents=[researcher_agent_1, researcher_agent_2, researcher_agent_3],
    description="Runs multiple research agents in parallel to gather information.",
)

merger_agent = LlmAgent(
    name="SynthesisAgent",
    model=GEMINI_MODEL,
    instruction=(
        "You are an AI Assistant responsible for combining research findings into a structured report.\n"
        "Use only the summaries provided below and do not add external facts.\n\n"
        "Renewable Energy:\n{renewable_energy_result}\n\n"
        "Electric Vehicles:\n{ev_technology_result}\n\n"
        "Carbon Capture:\n{carbon_capture_result}\n\n"
        "Format your response with these headings exactly:\n"
        "## Summary of Recent Sustainable Technology Advancements\n"
        "### Renewable Energy Findings\n"
        "### Electric Vehicle Findings\n"
        "### Carbon Capture Findings\n"
        "### Overall Conclusion"
    ),
    description="Combines findings from parallel agents into a structured report.",
)

sequential_pipeline_agent = SequentialAgent(
    name="ResearchAndSynthesisPipeline",
    sub_agents=[parallel_research_agent, merger_agent],
    description="Coordinates parallel research and synthesis.",
)

root_agent = sequential_pipeline_agent


def run_pipeline(runner: InMemoryRunner, request: str) -> str:
    """Run the ADK sequential pipeline for one user request."""
    print(f"\n--- Running ADK Parallelization Pipeline with request: '{request}' ---")
    final_result = ""
    try:
        user_id = "user_123"
        session_id = str(uuid.uuid4())

        create_session_result = runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        if inspect.isawaitable(create_session_result):
            asyncio.run(create_session_result)

        for event in runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=request)],
            ),
        ):
            if event.is_final_response() and event.content:
                if hasattr(event.content, "text") and event.content.text:
                    final_result = event.content.text
                elif event.content.parts:
                    text_parts = [part.text for part in event.content.parts if part.text]
                    final_result = "".join(text_parts)

        return final_result
    except Exception as exc:
        message = f"An error occurred while processing your request: {exc}"
        print(message)
        return message


def main() -> None:
    print("--- Google ADK Parallelization Example ---")
    print("Note: This requires Google ADK installed and authenticated.")

    runner = InMemoryRunner(root_agent)
    request = "Summarize recent sustainability technology advancements."
    result = run_pipeline(runner, request)
    print(f"Final Output: {result}")


if __name__ == "__main__":
    main()