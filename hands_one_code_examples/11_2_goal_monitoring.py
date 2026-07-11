from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

try:
    from crewai import Agent, Crew, LLM, Process, Task
except ImportError:
    Agent = Crew = LLM = Process = Task = None  # type: ignore[assignment]


DEFAULT_MODEL = "gemini/gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_GOAL = (
    "Design a Python function that solves BinaryGap, handles invalid inputs, "
    "and is easy to test."
)

CRITERION_IMPLEMENTATION = "implementation"
CRITERION_REVIEW = "review"
CRITERION_TESTS = "tests"
CRITERION_DOCUMENTATION = "documentation"
CRITERION_PROMPT = "prompt_refinement"


@dataclass(frozen=True)
class RoleDefinition:
    name: str
    goal: str
    backstory: str
    responsibility: str


@dataclass(frozen=True)
class SuccessCriterion:
    key: str
    description: str


@dataclass(frozen=True)
class GoalContract:
    goal: str
    criteria: tuple[SuccessCriterion, ...]

    def __post_init__(self) -> None:
        if not self.goal.strip():
            raise ValueError("goal must not be empty")
        if not self.criteria:
            raise ValueError("at least one success criterion is required")
        duplicate_keys = {
            criterion.key
            for criterion in self.criteria
            if [item.key for item in self.criteria].count(criterion.key) > 1
        }
        if duplicate_keys:
            raise ValueError(f"duplicate success criteria: {sorted(duplicate_keys)}")


@dataclass(frozen=True)
class ConversationMessage:
    speaker: str
    content: str
    stage: str


@dataclass(frozen=True)
class GoalEvaluation:
    status: str
    score: float
    satisfied_criteria: tuple[str, ...]
    missing_criteria: tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        return self.status == "complete"


@dataclass(frozen=True)
class RoleContribution:
    message: ConversationMessage
    completed_criteria: tuple[str, ...] = ()
    artifact_updates: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class GoalMonitoringState:
    contract: GoalContract
    conversation: tuple[ConversationMessage, ...] = ()
    completed_criteria: tuple[str, ...] = ()
    artifacts: dict[str, str] = field(default_factory=dict)

    def with_contribution(self, contribution: RoleContribution) -> "GoalMonitoringState":
        completed = tuple(
            dict.fromkeys(self.completed_criteria + contribution.completed_criteria)
        )
        artifacts = dict(self.artifacts)
        artifacts.update(contribution.artifact_updates)
        return GoalMonitoringState(
            contract=self.contract,
            conversation=self.conversation + (contribution.message,),
            completed_criteria=completed,
            artifacts=artifacts,
        )


@dataclass(frozen=True)
class GoalMonitoringResult:
    contract: GoalContract
    conversation: tuple[ConversationMessage, ...]
    evaluation: GoalEvaluation
    artifacts: dict[str, str]


class GoalMonitor:
    """Deterministic goal evaluator used instead of an LLM judge."""

    def __init__(self, contract: GoalContract) -> None:
        self.contract = contract

    def evaluate(self, completed_criteria: Iterable[str]) -> GoalEvaluation:
        completed = set(completed_criteria)
        expected_keys = tuple(criterion.key for criterion in self.contract.criteria)
        satisfied = tuple(key for key in expected_keys if key in completed)
        missing = tuple(key for key in expected_keys if key not in completed)
        score = round(len(satisfied) / len(expected_keys), 2)
        status = "complete" if not missing else "needs_revision"
        return GoalEvaluation(
            status=status,
            score=score,
            satisfied_criteria=satisfied,
            missing_criteria=missing,
        )


class GoalMonitoringRole:
    def __init__(self, definition: RoleDefinition) -> None:
        self.definition = definition

    @property
    def name(self) -> str:
        return self.definition.name

    def _message(self, stage: str, content: str) -> ConversationMessage:
        return ConversationMessage(speaker=self.name, stage=stage, content=content)


class PeerProgrammer(GoalMonitoringRole):
    def draft_solution(self, state: GoalMonitoringState) -> RoleContribution:
        code = (
            "def binary_gap(n: int) -> int:\n"
            "    if not isinstance(n, int) or n <= 0:\n"
            "        raise ValueError('n must be a positive integer')\n"
            "    bits = bin(n)[2:].strip('0')\n"
            "    return max((len(chunk) for chunk in bits.split('1')), default=0)"
        )
        content = (
            "I will solve the goal with a small pure function, explicit input "
            "validation, and no hidden I/O so review and tests can inspect behavior."
        )
        return RoleContribution(
            message=self._message("initial implementation", content),
            completed_criteria=(CRITERION_IMPLEMENTATION,),
            artifact_updates={"implementation": code},
        )

    def revise_solution(self, state: GoalMonitoringState) -> RoleContribution:
        content = (
            "I accept the review constraint: keep the function pure, raise a stable "
            "ValueError for invalid input, and document edge cases through tests "
            "instead of adding print statements."
        )
        return RoleContribution(
            message=self._message("revision", content),
            completed_criteria=(CRITERION_IMPLEMENTATION,),
            artifact_updates={
                "revision_notes": (
                    "Pure function retained; invalid inputs and no-gap inputs are "
                    "covered by tests."
                )
            },
        )


class CodeReviewer(GoalMonitoringRole):
    def review(self, state: GoalMonitoringState) -> RoleContribution:
        content = (
            "The draft is compact and testable. Required checks: reject non-positive "
            "and non-integer input, verify numbers with no enclosed zero gap, and "
            "avoid mixing console output into the core function."
        )
        return RoleContribution(
            message=self._message("review", content),
            completed_criteria=(CRITERION_REVIEW,),
            artifact_updates={
                "review": (
                    "Validate invalid inputs, no-gap cases, and separation between "
                    "business logic and I/O."
                )
            },
        )

    def final_check(self, state: GoalMonitoringState) -> RoleContribution:
        content = (
            "Final check passes conceptually: the implementation, review notes, "
            "tests, documentation, and refined prompt all map to the goal contract."
        )
        return RoleContribution(
            message=self._message("final monitoring check", content),
            completed_criteria=(CRITERION_REVIEW,),
            artifact_updates={"final_review": "All success criteria are represented."},
        )


class TestWriter(GoalMonitoringRole):
    def write_tests(self, state: GoalMonitoringState) -> RoleContribution:
        content = (
            "I would test known BinaryGap examples, invalid input handling, a value "
            "with no enclosed zero gap, and the conversation monitor so regressions "
            "are caught at both behavior and workflow levels."
        )
        return RoleContribution(
            message=self._message("test plan", content),
            completed_criteria=(CRITERION_TESTS,),
            artifact_updates={
                "test_strategy": (
                    "Unit tests: binary_gap(9)==2, binary_gap(529)==4, "
                    "binary_gap(15)==0, invalid inputs raise ValueError, and "
                    "monitoring reports all criteria complete."
                )
            },
        )


class Documenter(GoalMonitoringRole):
    def document(self, state: GoalMonitoringState) -> RoleContribution:
        content = (
            "Documentation should state the goal, explain the pure-function API, "
            "show one call example, and tell operators to run tests before using "
            "the function in a larger agent workflow."
        )
        return RoleContribution(
            message=self._message("documentation", content),
            completed_criteria=(CRITERION_DOCUMENTATION,),
            artifact_updates={
                "documentation": (
                    "Run the demo with `uv run python -m "
                    "hands_one_code_examples.11_2_goal_monitoring`. The console "
                    "prints each role contribution and the goal-monitoring summary."
                )
            },
        )


class PromptRefiner(GoalMonitoringRole):
    def refine(self, state: GoalMonitoringState) -> RoleContribution:
        content = (
            "Refined prompt: Write a pure, typed Python BinaryGap function for "
            "positive integers; reject invalid inputs; include edge-case tests; "
            "keep business logic separate from console output; explain decisions "
            "briefly."
        )
        return RoleContribution(
            message=self._message("prompt refinement", content),
            completed_criteria=(CRITERION_PROMPT,),
            artifact_updates={"refined_prompt": content.removeprefix("Refined prompt: ")},
        )


class GoalMonitoringWorkflow:
    """Coordinates role handoffs using generate-review-revise reflection."""

    def __init__(
        self,
        *,
        contract: GoalContract,
        peer_programmer: PeerProgrammer,
        code_reviewer: CodeReviewer,
        test_writer: TestWriter,
        documenter: Documenter,
        prompt_refiner: PromptRefiner,
    ) -> None:
        self.contract = contract
        self.peer_programmer = peer_programmer
        self.code_reviewer = code_reviewer
        self.test_writer = test_writer
        self.documenter = documenter
        self.prompt_refiner = prompt_refiner
        self.monitor = GoalMonitor(contract)

    def run(self) -> GoalMonitoringResult:
        state = GoalMonitoringState(contract=self.contract)
        for contribution in (
            self.peer_programmer.draft_solution(state),
            self.code_reviewer.review(state),
            self.peer_programmer.revise_solution(state),
            self.test_writer.write_tests(state),
            self.documenter.document(state),
            self.prompt_refiner.refine(state),
            self.code_reviewer.final_check(state),
        ):
            state = state.with_contribution(contribution)

        return GoalMonitoringResult(
            contract=self.contract,
            conversation=state.conversation,
            evaluation=self.monitor.evaluate(state.completed_criteria),
            artifacts=state.artifacts,
        )


def build_default_success_criteria() -> tuple[SuccessCriterion, ...]:
    return (
        SuccessCriterion(
            CRITERION_IMPLEMENTATION,
            "A runnable implementation strategy exists for the requested goal.",
        ),
        SuccessCriterion(
            CRITERION_REVIEW,
            "A reviewer checks correctness, edge cases, and maintainability.",
        ),
        SuccessCriterion(
            CRITERION_TESTS,
            "A test strategy covers normal cases, edge cases, and failure cases.",
        ),
        SuccessCriterion(
            CRITERION_DOCUMENTATION,
            "Usage documentation explains how to run and interpret the example.",
        ),
        SuccessCriterion(
            CRITERION_PROMPT,
            "A refined prompt captures the improved instructions for future runs.",
        ),
    )


def build_goal_contract(
    goal: str = DEFAULT_GOAL,
    *,
    criteria: tuple[SuccessCriterion, ...] | None = None,
) -> GoalContract:
    return GoalContract(goal=goal, criteria=criteria or build_default_success_criteria())


def build_role_definitions() -> tuple[RoleDefinition, ...]:
    return (
        RoleDefinition(
            name="Peer Programmer",
            goal="Create a clear implementation approach for the requested coding goal.",
            backstory=(
                "A practical developer who favors small APIs, direct control flow, "
                "and easy-to-test behavior."
            ),
            responsibility="Draft and revise the implementation.",
        ),
        RoleDefinition(
            name="Code Reviewer",
            goal="Find correctness, reliability, and maintainability issues.",
            backstory=(
                "A senior reviewer who turns vague quality concerns into concrete "
                "acceptance checks."
            ),
            responsibility="Review the implementation and perform the final goal check.",
        ),
        RoleDefinition(
            name="Documenter",
            goal="Make the outcome understandable and runnable.",
            backstory="A concise technical writer focused on operator instructions.",
            responsibility="Document usage and interpretation.",
        ),
        RoleDefinition(
            name="Test Writer",
            goal="Design tests that prove the goal contract is satisfied.",
            backstory=(
                "A test engineer who covers expected behavior, edge cases, and "
                "workflow regressions."
            ),
            responsibility="Create a test strategy.",
        ),
        RoleDefinition(
            name="Prompt Refiner",
            goal="Improve the future prompt from what the team learned.",
            backstory=(
                "A prompt engineer who converts review findings into precise future "
                "instructions."
            ),
            responsibility="Produce the refined prompt.",
        ),
    )


def _role_map() -> dict[str, RoleDefinition]:
    return {definition.name: definition for definition in build_role_definitions()}


def build_goal_monitoring_workflow(
    goal: str = DEFAULT_GOAL,
) -> GoalMonitoringWorkflow:
    contract = build_goal_contract(goal)
    roles = _role_map()
    return GoalMonitoringWorkflow(
        contract=contract,
        peer_programmer=PeerProgrammer(roles["Peer Programmer"]),
        code_reviewer=CodeReviewer(roles["Code Reviewer"]),
        test_writer=TestWriter(roles["Test Writer"]),
        documenter=Documenter(roles["Documenter"]),
        prompt_refiner=PromptRefiner(roles["Prompt Refiner"]),
    )


def run_goal_monitoring_demo(goal: str = DEFAULT_GOAL) -> GoalMonitoringResult:
    return build_goal_monitoring_workflow(goal).run()


def format_goal_monitoring_result(result: GoalMonitoringResult) -> str:
    lines = [
        "# Goal",
        result.contract.goal,
        "",
        "# Role Conversation",
    ]
    for index, message in enumerate(result.conversation, start=1):
        lines.append(
            f"{index}. {message.speaker} [{message.stage}]: {message.content}"
        )

    lines.extend(
        [
            "",
            "# Goal Monitoring Summary",
            f"Status: {result.evaluation.status}",
            f"Score: {result.evaluation.score:.2f}",
            "Satisfied: " + ", ".join(result.evaluation.satisfied_criteria),
            "Missing: "
            + (", ".join(result.evaluation.missing_criteria) or "none"),
            "",
            "# Final Artifacts",
        ]
    )
    for key in sorted(result.artifacts):
        lines.append(f"- {key}: {result.artifacts[key]}")
    return "\n".join(lines)


def print_goal_monitoring_result(result: GoalMonitoringResult) -> None:
    print(format_goal_monitoring_result(result))


def load_environment_variables() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")


def normalize_model_provider_environment() -> None:
    if os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]


def require_crewai() -> None:
    if any(item is None for item in (Agent, Crew, LLM, Process, Task)):
        raise ImportError(
            "crewai is not installed. Install it with `uv add crewai==0.130.0` "
            "before running the optional live CrewAI blueprint."
        )


def _model_requires_google_api_key(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.startswith("gemini/") or normalized.startswith("google/")


def validate_runtime_environment(model: str = DEFAULT_MODEL) -> None:
    if _model_requires_google_api_key(model) and not (
        os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    ):
        raise ValueError(
            "GEMINI_API_KEY or GOOGLE_API_KEY not found. Set one of them before "
            "running the optional live CrewAI blueprint."
        )


def create_llm(
    model: str = DEFAULT_MODEL,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Any:
    require_crewai()
    return LLM(model=model, temperature=temperature)


def build_crewai_agents(
    *,
    llm: Any | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    require_crewai()
    active_llm = llm or create_llm()
    agents: dict[str, Any] = {}
    for definition in build_role_definitions():
        agents[definition.name] = Agent(
            role=definition.name,
            goal=definition.goal,
            backstory=definition.backstory,
            llm=active_llm,
            verbose=verbose,
            allow_delegation=False,
        )
    return agents


def build_crewai_tasks(
    *,
    contract: GoalContract | None = None,
    agents: dict[str, Any] | None = None,
) -> list[Any]:
    require_crewai()
    active_contract = contract or build_goal_contract()
    active_agents = agents or build_crewai_agents()

    peer_task = Task(
        name="peer_programming_task",
        description=(
            "Draft an implementation plan and compact Python solution for this goal: "
            "{goal}. Keep the logic pure and easy to test."
        ),
        expected_output="A short implementation plan and Python code block.",
        agent=active_agents["Peer Programmer"],
        markdown=True,
    )
    review_task = Task(
        name="code_review_task",
        description=(
            "Review the implementation against the goal and success criteria. "
            "List only required changes."
        ),
        expected_output="Concise review notes with required changes or approval.",
        agent=active_agents["Code Reviewer"],
        context=[peer_task],
        markdown=True,
    )
    test_task = Task(
        name="test_writing_task",
        description="Create unit tests that prove the goal contract is satisfied.",
        expected_output="A concise pytest-oriented test plan.",
        agent=active_agents["Test Writer"],
        context=[peer_task, review_task],
        markdown=True,
    )
    documentation_task = Task(
        name="documentation_task",
        description="Document how to run the solution and interpret the output.",
        expected_output="Concise usage documentation.",
        agent=active_agents["Documenter"],
        context=[peer_task, review_task, test_task],
        markdown=True,
    )
    prompt_task = Task(
        name="prompt_refinement_task",
        description=(
            "Refine the original goal prompt using the implementation, review, "
            "tests, and documentation."
        ),
        expected_output="A reusable refined prompt for a future coding agent.",
        agent=active_agents["Prompt Refiner"],
        context=[peer_task, review_task, test_task, documentation_task],
        markdown=True,
    )
    _ = active_contract
    return [peer_task, review_task, test_task, documentation_task, prompt_task]


def build_goal_monitoring_crew(
    *,
    goal: str = DEFAULT_GOAL,
    llm: Any | None = None,
    verbose: bool = False,
) -> Any:
    require_crewai()
    contract = build_goal_contract(goal)
    agents = build_crewai_agents(llm=llm, verbose=verbose)
    tasks = build_crewai_tasks(contract=contract, agents=agents)
    return Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=verbose,
    )


@dataclass(frozen=True)
class GoalMonitoringCrewApp:
    """Facade for deterministic demo output and optional CrewAI blueprint execution."""

    goal: str = DEFAULT_GOAL
    model: str = DEFAULT_MODEL
    verbose: bool = False

    def run_demo(self) -> GoalMonitoringResult:
        return run_goal_monitoring_demo(self.goal)

    def print_demo(self) -> GoalMonitoringResult:
        result = self.run_demo()
        print_goal_monitoring_result(result)
        return result

    def build_live_crew(self) -> Any:
        load_environment_variables()
        normalize_model_provider_environment()
        validate_runtime_environment(self.model)
        return build_goal_monitoring_crew(
            goal=self.goal,
            llm=create_llm(self.model),
            verbose=self.verbose,
        )

    def run_live_crew(self) -> Any:
        crew = self.build_live_crew()
        return crew.kickoff(inputs={"goal": self.goal})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demonstrate a CrewAI-style goal setting and monitoring pattern."
    )
    parser.add_argument("--goal", default=DEFAULT_GOAL, help="Goal to monitor.")
    parser.add_argument(
        "--live-crewai",
        action="store_true",
        help="Run the optional live CrewAI crew instead of the deterministic demo.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="CrewAI LLM model name.")
    parser.add_argument("--verbose", action="store_true", help="Enable CrewAI verbosity.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    app = GoalMonitoringCrewApp(
        goal=args.goal,
        model=args.model,
        verbose=args.verbose,
    )
    if args.live_crewai:
        result = app.run_live_crew()
        print(result)
        return
    app.print_demo()


if __name__ == "__main__":
    main()
