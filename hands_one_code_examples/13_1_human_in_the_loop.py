from __future__ import annotations

import asyncio
import re
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, Sequence

try:
    from google.adk.events import Event, EventActions, RequestInput
    from google.adk.runners import InMemoryRunner
    from google.adk.workflow import START, Workflow, node
    from google.genai import types
except ImportError:
    Event = EventActions = RequestInput = InMemoryRunner = None  # type: ignore[assignment]
    START = Workflow = node = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]

from hands_one_code_examples._shared.adk_runtime import (
    build_user_message,
    content_to_text,
    derive_session_id,
    event_output_to_text,
    format_structured_value,
    load_environment_variables,
    require_google_adk,
)
from hands_one_code_examples._shared.hitl_runtime import (
    get_mutable_state,
    parse_json_object,
    to_plain_data,
)


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_APP_NAME = "refund_human_in_the_loop_demo"
DEFAULT_USER_ID = "demo-user"
DEFAULT_SESSION_ID = "refund-review"
DEFAULT_REQUEST = (
    "Refund order ORD-2049 for $149.99 because the customer was double charged "
    "10 days ago. Customer tier is gold."
)
DEFAULT_SUPERVISOR_REVIEW = {
    "decision": "approve",
    "approved_amount": 149.99,
    "notes": "Approved after validating the duplicate charge evidence.",
}

STATE_REQUEST = "refund_request_text"
STATE_REVIEW_PACKET = "review_packet"
STATE_SUPERVISOR_REVIEW = "supervisor_review"
STATE_EXECUTION_RESULT = "execution_result"
STATE_EXECUTION_STATUS = "execution_status"
STATE_FINAL_SUMMARY = "final_summary"
STATE_WORKFLOW_PATH = "workflow_path"

DECISION_APPROVE = "approve"
DECISION_REJECT = "reject"
STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_EXECUTED = "executed"
STATUS_SKIPPED = "skipped"

SUPERVISOR_REVIEW_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": [DECISION_APPROVE, DECISION_REJECT],
        },
        "approved_amount": {
            "type": "number",
            "minimum": 0,
        },
        "notes": {
            "type": "string",
        },
    },
    "required": ["decision", "notes"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class RefundRequest:
    order_id: str
    amount: float
    reason: str
    customer_tier: str
    days_since_purchase: int | None


@dataclass(frozen=True)
class RefundAssessment:
    recommended_action: str
    risk_flags: list[str]
    customer_message: str


@dataclass(frozen=True)
class SupervisorReview:
    decision: str
    approved_amount: float
    notes: str


@dataclass(frozen=True)
class WorkflowStep:
    actor: str
    kind: str
    text: str
    is_final: bool


@dataclass(frozen=True)
class ScenarioRunResult:
    scenario_name: str
    request: str
    steps: list[WorkflowStep]
    final_response: str
    review_decision: str
    refund_executed: bool
    session_id: str


class RefundPolicyEngine:
    """Deterministic policy service kept separate from ADK orchestration code."""

    def __init__(
        self,
        *,
        manual_review_threshold: float = 100.0,
        stale_refund_days: int = 30,
    ) -> None:
        self.manual_review_threshold = manual_review_threshold
        self.stale_refund_days = stale_refund_days

    def parse_request(self, request: str) -> RefundRequest:
        if not request.strip():
            raise ValueError("request must not be empty")

        order_match = re.search(r"\bORD-\d+\b", request, flags=re.IGNORECASE)
        amount_match = re.search(r"\$(\d+(?:\.\d{1,2})?)", request)
        if amount_match is None:
            amount_match = re.search(
                r"\brefund(?:\s+order\s+ORD-\d+)?\s+(?:for\s+)?(\d+(?:\.\d{1,2})?)\b",
                request,
                flags=re.IGNORECASE,
            )
        days_match = re.search(r"\b(\d+)\s+days?\b", request, flags=re.IGNORECASE)
        reason_match = re.search(r"\bbecause\b(.+)", request, flags=re.IGNORECASE)

        missing_fields: list[str] = []
        if order_match is None:
            missing_fields.append("order id in the form ORD-1234")
        if amount_match is None:
            missing_fields.append("refund amount")
        if missing_fields:
            raise ValueError(
                "Unable to parse the refund request. Include "
                + " and ".join(missing_fields)
                + "."
            )

        customer_tier = "standard"
        for tier in ("platinum", "gold", "silver", "standard"):
            if re.search(rf"\b{tier}\b", request, flags=re.IGNORECASE):
                customer_tier = tier
                break

        reason = (
            reason_match.group(1).strip().rstrip(".")
            if reason_match is not None
            else request.strip()
        )
        days_since_purchase = (
            int(days_match.group(1)) if days_match is not None else None
        )

        return RefundRequest(
            order_id=order_match.group(0).upper(),
            amount=float(amount_match.group(1)),
            reason=reason,
            customer_tier=customer_tier,
            days_since_purchase=days_since_purchase,
        )

    def assess(self, refund_request: RefundRequest) -> RefundAssessment:
        risk_flags: list[str] = []

        if refund_request.amount > self.manual_review_threshold:
            risk_flags.append(
                f"Amount exceeds the auto-approval threshold of ${self.manual_review_threshold:.2f}."
            )
        if (
            refund_request.days_since_purchase is not None
            and refund_request.days_since_purchase > self.stale_refund_days
        ):
            risk_flags.append(
                f"Purchase is older than {self.stale_refund_days} days."
            )
        if re.search(
            r"\b(chargeback|fraud|bank dispute|compliance)\b",
            refund_request.reason,
            flags=re.IGNORECASE,
        ):
            risk_flags.append(
                "Case contains a fraud or dispute keyword and requires human review."
            )

        recommended_action = "manual-review" if risk_flags else "approve-and-execute"
        customer_message = (
            "We validated the request and are preparing the refund."
            if not risk_flags
            else "A supervisor needs to review this refund before it can be issued."
        )

        return RefundAssessment(
            recommended_action=recommended_action,
            risk_flags=risk_flags,
            customer_message=customer_message,
        )

    def build_review_packet(self, request_text: str) -> dict[str, Any]:
        refund_request = self.parse_request(request_text)
        assessment = self.assess(refund_request)
        return {
            "refund_request": to_plain_data(refund_request),
            "assessment": to_plain_data(assessment),
            "review_guidance": {
                "required_decisions": [
                    "approve",
                    "reject",
                ],
                "notes_expectation": (
                    "Reference the evidence you checked or the reason for rejection."
                ),
            },
        }


POLICY_ENGINE = RefundPolicyEngine()


def append_workflow_marker(state: dict[str, Any], marker: str) -> None:
    path = list(state.get(STATE_WORKFLOW_PATH, []))
    if not path or path[-1] != marker:
        path.append(marker)
    state[STATE_WORKFLOW_PATH] = path


def build_initial_state(request: str) -> dict[str, Any]:
    if not request.strip():
        raise ValueError("request must not be empty")

    return {
        STATE_REQUEST: request.strip(),
        STATE_EXECUTION_STATUS: STATUS_PENDING,
        STATE_WORKFLOW_PATH: ["request-received"],
    }


def node_input_to_text(node_input: Any) -> str:
    """Convert ADK workflow input into the original user-request text."""
    if isinstance(node_input, str):
        return node_input.strip()

    text = content_to_text(node_input).strip()
    if text:
        return text

    if node_input is None:
        return ""

    return str(node_input).strip()


def extract_request_text_from_state(state: dict[str, Any]) -> str:
    request_text = str(state.get(STATE_REQUEST, "")).strip()
    if request_text:
        return request_text

    review_packet = state.get(STATE_REVIEW_PACKET)
    if isinstance(review_packet, dict):
        refund_request = review_packet.get("refund_request", {})
        if isinstance(refund_request, dict):
            order_id = str(refund_request.get("order_id", "")).strip()
            amount = refund_request.get("amount")
            reason = str(refund_request.get("reason", "")).strip()
            customer_tier = str(refund_request.get("customer_tier", "")).strip()
            if order_id and amount is not None:
                return (
                    f"Refund order {order_id} for ${float(amount):.2f} "
                    f"because {reason or 'supervisor review is required'}. "
                    f"Customer tier is {customer_tier or 'standard'}."
                )

    return ""


def get_or_build_review_packet(
    state: dict[str, Any],
    *,
    request_text: str = "",
) -> dict[str, Any]:
    review_packet = state.get(STATE_REVIEW_PACKET)
    if isinstance(review_packet, dict):
        return review_packet

    effective_request_text = request_text.strip() or extract_request_text_from_state(
        state
    )
    if not effective_request_text:
        raise ValueError(
            "Refund review context is missing. Start a new ADK Web session and "
            f"send a request such as: {DEFAULT_REQUEST}"
        )

    review_packet = POLICY_ENGINE.build_review_packet(effective_request_text)
    state[STATE_REQUEST] = effective_request_text
    state[STATE_REVIEW_PACKET] = review_packet
    return review_packet


@node(name="InitializeRefundRequest")
def initialize_refund_request(node_input: Any) -> Event:
    """Persist the ADK Web user message into the workflow session state.

    When this module is run through ``RefundHumanInTheLoopDemo``, the session is
    created with ``build_initial_state`` before the workflow starts. ADK Web,
    however, creates an empty session and passes the typed message from ``START``
    as ``node_input``. This node supports both paths and records the normalized
    request as an event state delta before downstream nodes execute.
    """
    request_text = node_input_to_text(node_input)
    if not request_text:
        raise ValueError(
            "No refund request was provided. Enter a request such as: "
            f"{DEFAULT_REQUEST}"
        )

    return Event(
        author="InitializeRefundRequest",
        output=request_text,
        actions=EventActions(state_delta=build_initial_state(request_text)),
    )


def build_supervisor_review_message(review_packet: dict[str, Any]) -> str:
    refund_request = review_packet["refund_request"]
    assessment = review_packet["assessment"]
    risk_flags = assessment["risk_flags"] or ["No policy risk flags detected."]
    risk_summary = "\n".join(f"- {flag}" for flag in risk_flags)
    return (
        "Review the prepared refund packet and respond with JSON matching the "
        "requested schema.\n"
        f"Order: {refund_request['order_id']}\n"
        f"Amount requested: ${refund_request['amount']:.2f}\n"
        f"Customer tier: {refund_request['customer_tier']}\n"
        f"Recommended action: {assessment['recommended_action']}\n"
        "Risk flags:\n"
        f"{risk_summary}\n"
        "Use `approve` or `reject`, include `notes`, and set `approved_amount` "
        "when approving."
    )


def parse_supervisor_review(
    review_input: Any,
    *,
    requested_amount: float,
) -> SupervisorReview:
    payload = parse_json_object(review_input)
    if set(payload.keys()) == {"result"}:
        payload = parse_json_object(payload["result"])
    decision = str(payload.get("decision", "")).strip().lower()
    notes = str(payload.get("notes", "")).strip()

    if decision not in {DECISION_APPROVE, DECISION_REJECT}:
        raise ValueError("decision must be either 'approve' or 'reject'")
    if not notes:
        raise ValueError("notes must not be empty")

    if decision == DECISION_REJECT:
        return SupervisorReview(
            decision=DECISION_REJECT,
            approved_amount=0.0,
            notes=notes,
        )

    approved_amount_raw = payload.get("approved_amount", requested_amount)
    try:
        approved_amount = float(approved_amount_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("approved_amount must be numeric when approving") from exc

    if approved_amount <= 0:
        raise ValueError("approved_amount must be greater than zero")
    if approved_amount > requested_amount:
        raise ValueError("approved_amount must not exceed the requested amount")

    return SupervisorReview(
        decision=DECISION_APPROVE,
        approved_amount=approved_amount,
        notes=notes,
    )


@node(name="SupervisorReviewNode")
def request_supervisor_review(node_input: Any, ctx: Any) -> Any:
    state = get_mutable_state(ctx)

    # The predecessor output is the normalized request text. The state fallback
    # keeps this node robust when a custom runner pre-initializes the session or
    # when ADK reconstructs the workflow around a HITL interruption.
    request_text = node_input_to_text(node_input)
    if not request_text:
        request_text = str(state.get(STATE_REQUEST, "")).strip()

    if not request_text:
        raise ValueError(
            "Refund request is missing from both workflow input and session state."
        )

    # Keep state synchronized even if this node is invoked outside the standard
    # START -> InitializeRefundRequest path.
    state[STATE_REQUEST] = request_text

    review_packet = get_or_build_review_packet(state, request_text=request_text)
    state[STATE_REVIEW_PACKET] = review_packet
    append_workflow_marker(state, "review-packet-prepared")
    yield Event(
        author="SupervisorReviewNode",
        actions=EventActions(
            state_delta={
                STATE_REQUEST: state[STATE_REQUEST],
                STATE_REVIEW_PACKET: review_packet,
                STATE_WORKFLOW_PATH: state[STATE_WORKFLOW_PATH],
            }
        ),
    )

    refund_request = review_packet["refund_request"]
    yield RequestInput(
        interrupt_id=f"refund-review-{refund_request['order_id'].lower()}",
        message=build_supervisor_review_message(review_packet),
        payload=review_packet,
        response_schema=SUPERVISOR_REVIEW_RESPONSE_SCHEMA,
    )


@node(name="RecordSupervisorReview")
def record_supervisor_review(node_input: Any, ctx: Any) -> str:
    state = get_mutable_state(ctx)
    review_packet = get_or_build_review_packet(state)
    refund_request = review_packet["refund_request"]
    supervisor_review = parse_supervisor_review(
        node_input,
        requested_amount=float(refund_request["amount"]),
    )

    state[STATE_SUPERVISOR_REVIEW] = to_plain_data(supervisor_review)
    state[STATE_EXECUTION_STATUS] = (
        STATUS_APPROVED
        if supervisor_review.decision == DECISION_APPROVE
        else STATUS_REJECTED
    )
    append_workflow_marker(state, f"supervisor-{supervisor_review.decision}")

    return (
        f"Supervisor decision recorded: {supervisor_review.decision}. "
        f"Notes: {supervisor_review.notes}"
    )


@node(name="RefundDecisionRouter")
def refund_decision_router(ctx: Any) -> Event:
    state = get_mutable_state(ctx)
    review = state.get(STATE_SUPERVISOR_REVIEW, {})
    approved = review.get("decision") == DECISION_APPROVE
    if approved:
        message = (
            "Supervisor approved the refund. Route to the execution agent."
        )
        append_workflow_marker(state, "route-to-execution")
    else:
        message = "Supervisor rejected the refund. Route directly to the summary node."
        append_workflow_marker(state, "route-to-summary")

    return Event(
        author="RefundDecisionRouter",
        content=types.Content(role="model", parts=[types.Part(text=message)]),
        actions=EventActions(route=approved),
    )


def issue_refund(order_id: str, amount: float, notes: str = "") -> dict[str, Any]:
    return {
        "status": STATUS_EXECUTED,
        "order_id": order_id,
        "amount": round(float(amount), 2),
        "notes": notes.strip(),
        "transaction_id": f"refund-{order_id.lower()}",
    }


@node(name="RefundExecutionNode")
def execute_refund(ctx: Any) -> Event:
    state = get_mutable_state(ctx)
    review_packet = get_or_build_review_packet(state)
    refund_request = review_packet["refund_request"]
    supervisor_review = state.get(STATE_SUPERVISOR_REVIEW, {})
    order_id = refund_request.get("order_id", "unknown-order")

    if supervisor_review.get("decision") != DECISION_APPROVE:
        state[STATE_EXECUTION_STATUS] = STATUS_SKIPPED
        append_workflow_marker(state, "refund-skipped")
        message = f"Refund for {order_id} was skipped because the supervisor rejected it."
        return Event(
            author="RefundExecutionNode",
            content=types.Content(role="model", parts=[types.Part(text=message)]),
        )

    execution_args = {
        "order_id": order_id,
        "amount": float(supervisor_review.get("approved_amount", refund_request["amount"])),
        "notes": str(supervisor_review.get("notes", "")).strip(),
    }
    execution_result = issue_refund(**execution_args)
    state[STATE_EXECUTION_RESULT] = execution_result
    state[STATE_EXECUTION_STATUS] = STATUS_EXECUTED
    append_workflow_marker(state, "refund-issued")

    message = (
        f"Refund executed for {order_id} in the amount of "
        f"${float(execution_result['amount']):.2f}."
    )
    return Event(
        author="RefundExecutionNode",
        output=execution_result,
        content=types.Content(role="model", parts=[types.Part(text=message)]),
    )


def build_summary_text(state: dict[str, Any]) -> str:
    review_packet = state.get(STATE_REVIEW_PACKET, {})
    refund_request = review_packet.get("refund_request", {})
    review = state.get(STATE_SUPERVISOR_REVIEW, {})
    execution_result = state.get(STATE_EXECUTION_RESULT)
    order_id = refund_request.get("order_id", "unknown-order")

    if review.get("decision") == DECISION_REJECT:
        return (
            f"Refund for {order_id} was rejected by the supervisor.\n"
            f"Reason: {review.get('notes', 'No reason provided.')}"
        )

    if execution_result is not None:
        return (
            f"Refund issued for {order_id} in the amount of "
            f"${float(execution_result['amount']):.2f}.\n"
            f"Transaction id: {execution_result['transaction_id']}."
        )

    return (
        f"Refund for {order_id} was approved by the supervisor but not executed.\n"
        "Check the execution step in ADK Web or rerun the workflow."
    )


@node(name="RefundSummaryNode")
def refund_summary_node(ctx: Any) -> Event:
    state = get_mutable_state(ctx)
    summary_text = build_summary_text(state)
    state[STATE_FINAL_SUMMARY] = summary_text
    append_workflow_marker(state, "summary-generated")

    return Event(
        author="RefundSummaryNode",
        content=types.Content(role="model", parts=[types.Part(text=summary_text)]),
    )


@node(name="WorkflowTerminator")
def workflow_terminator(ctx: Any) -> None:
    del ctx
    return None


def build_refund_workflow() -> Any:
    require_google_adk(Workflow, START, node)
    return Workflow(
        name="RefundHumanInTheLoopWorkflow",
        description=(
            "Prepares a deterministic refund review packet, pauses for supervisor "
            "input, then executes the refund after approval."
        ),
        edges=[
            (START, initialize_refund_request),
            (initialize_refund_request, request_supervisor_review),
            (request_supervisor_review, record_supervisor_review),
            (record_supervisor_review, refund_decision_router),
            (
                refund_decision_router,
                {
                    True: execute_refund,
                    False: refund_summary_node,
                },
            ),
            (execute_refund, refund_summary_node),
            (refund_summary_node, workflow_terminator),
        ],
    )


def build_runner(agent: Any | None = None, *, app_name: str = DEFAULT_APP_NAME) -> Any:
    require_google_adk(InMemoryRunner)
    active_agent = agent or build_refund_workflow()
    return InMemoryRunner(agent=active_agent, app_name=app_name)


async def create_run_session(
    runner: Any,
    *,
    user_id: str,
    base_session_id: str,
    session_index: int,
    state: dict[str, Any],
) -> Any:
    session_id = derive_session_id(base_session_id, session_index)
    return await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id,
        state=state,
    )


def _get_function_calls(event: Any) -> list[Any]:
    getter = getattr(event, "get_function_calls", None)
    if callable(getter):
        return list(getter())

    parts = getattr(getattr(event, "content", None), "parts", None) or []
    return [
        function_call
        for function_call in (getattr(part, "function_call", None) for part in parts)
        if function_call is not None
    ]


def _get_function_responses(event: Any) -> list[Any]:
    getter = getattr(event, "get_function_responses", None)
    if callable(getter):
        return list(getter())

    parts = getattr(getattr(event, "content", None), "parts", None) or []
    return [
        function_response
        for function_response in (
            getattr(part, "function_response", None) for part in parts
        )
        if function_response is not None
    ]


def event_to_workflow_steps(event: Any) -> list[WorkflowStep]:
    actor = getattr(event, "author", None) or "unknown"
    if actor == "user":
        return []

    steps: list[WorkflowStep] = []

    for function_call in _get_function_calls(event):
        steps.append(
            WorkflowStep(
                actor=actor,
                kind="tool-call",
                text=(
                    f"Called `{getattr(function_call, 'name', 'unknown_tool')}` with "
                    f"{format_structured_value(getattr(function_call, 'args', None))}."
                ),
                is_final=False,
            )
        )

    for function_response in _get_function_responses(event):
        steps.append(
            WorkflowStep(
                actor=actor,
                kind="tool-result",
                text=(
                    f"`{getattr(function_response, 'name', 'unknown_tool')}` returned "
                    f"{format_structured_value(getattr(function_response, 'response', None))}."
                ),
                is_final=False,
            )
        )

    text = content_to_text(getattr(event, "content", None))
    if not text:
        text = event_output_to_text(event)

    if text:
        is_final_response = getattr(event, "is_final_response", None)
        is_final = bool(callable(is_final_response) and is_final_response())
        if actor == "SupervisorReviewNode":
            kind = "hitl-request"
        elif actor == "RefundDecisionRouter":
            kind = "route"
        elif actor == "RefundExecutionNode":
            kind = "execution"
        else:
            kind = "final" if is_final else "message"
        steps.append(
            WorkflowStep(
                actor=actor,
                kind=kind,
                text=text,
                is_final=is_final,
            )
        )

    return steps


def extract_workflow_steps(events: Sequence[Any]) -> list[WorkflowStep]:
    steps: list[WorkflowStep] = []
    for event in events:
        steps.extend(event_to_workflow_steps(event))
    return steps


def extract_final_response_text(events: Sequence[Any]) -> str:
    steps = extract_workflow_steps(events)
    for step in reversed(steps):
        if step.is_final:
            return step.text
    for step in reversed(steps):
        if step.kind in {"final", "message"}:
            return step.text
    return ""


def detect_review_decision(events: Sequence[Any]) -> str:
    for event in reversed(events):
        for function_response in _get_function_responses(event):
            response_payload = getattr(function_response, "response", None)
            if isinstance(response_payload, dict) and response_payload.get("status") == STATUS_EXECUTED:
                return DECISION_APPROVE
        text = content_to_text(getattr(event, "content", None))
        if "rejected" in text.lower():
            return DECISION_REJECT
        if "approved" in text.lower():
            return DECISION_APPROVE
    return ""


def detect_refund_execution(events: Sequence[Any]) -> bool:
    for event in events:
        if (getattr(event, "author", None) or "") == "RefundExecutionNode":
            return True
        for function_response in _get_function_responses(event):
            response_payload = getattr(function_response, "response", None)
            if isinstance(response_payload, dict) and response_payload.get("status") == STATUS_EXECUTED:
                return True
    return False


@dataclass(frozen=True)
class RefundHumanInTheLoopDemo:
    execution_node: Any
    workflow: Any
    runner: Any
    app_name: str
    user_id: str
    session_id: str

    @classmethod
    def build(
        cls,
        *,
        model: str = DEFAULT_MODEL,
        app_name: str = DEFAULT_APP_NAME,
        user_id: str = DEFAULT_USER_ID,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> "RefundHumanInTheLoopDemo":
        del model
        execution_node = execute_refund
        workflow = build_refund_workflow()
        runner = build_runner(workflow, app_name=app_name)
        return cls(
            execution_node=execution_node,
            workflow=workflow,
            runner=runner,
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )

    async def run_request(
        self,
        request: str,
        *,
        scenario_name: str,
        session_index: int,
    ) -> ScenarioRunResult:
        state = build_initial_state(request)
        session = await create_run_session(
            self.runner,
            user_id=self.user_id,
            base_session_id=self.session_id,
            session_index=session_index,
            state=state,
        )
        async with aclosing(
            self.runner.run_async(
                user_id=self.user_id,
                session_id=session.id,
                new_message=build_user_message(request),
                yield_user_message=True,
            )
        ) as event_stream:
            events = [event async for event in event_stream]

        return ScenarioRunResult(
            scenario_name=scenario_name,
            request=request,
            steps=extract_workflow_steps(events),
            final_response=extract_final_response_text(events),
            review_decision=detect_review_decision(events),
            refund_executed=detect_refund_execution(events),
            session_id=session.id,
        )

    async def close(self) -> None:
        await self.runner.close()


def simulate_console_scenario(
    request: str,
    supervisor_review: dict[str, Any],
) -> ScenarioRunResult:
    state = build_initial_state(request)
    review_packet = POLICY_ENGINE.build_review_packet(request)
    state[STATE_REVIEW_PACKET] = review_packet
    steps = [
        WorkflowStep(
            actor="SupervisorReviewNode",
            kind="hitl-request",
            text=build_supervisor_review_message(review_packet),
            is_final=False,
        )
    ]

    parsed_review = parse_supervisor_review(
        supervisor_review,
        requested_amount=float(review_packet["refund_request"]["amount"]),
    )
    state[STATE_SUPERVISOR_REVIEW] = to_plain_data(parsed_review)
    steps.append(
        WorkflowStep(
            actor="RecordSupervisorReview",
            kind="message",
            text=f"Supervisor decision recorded: {parsed_review.decision}.",
            is_final=False,
        )
    )

    if parsed_review.decision == DECISION_APPROVE:
        execution_args = {
            "order_id": review_packet["refund_request"]["order_id"],
            "amount": parsed_review.approved_amount,
            "notes": parsed_review.notes,
        }
        execution_result = issue_refund(**execution_args)
        state[STATE_EXECUTION_RESULT] = execution_result
        state[STATE_EXECUTION_STATUS] = STATUS_EXECUTED
        steps.extend(
            [
                WorkflowStep(
                    actor="RefundExecutionNode",
                    kind="execution",
                    text=(
                        f"Refund executed for {execution_args['order_id']} in the "
                        f"amount of ${float(execution_result['amount']):.2f}."
                    ),
                    is_final=False,
                ),
            ]
        )
    else:
        state[STATE_EXECUTION_STATUS] = STATUS_SKIPPED

    final_response = build_summary_text(state)
    steps.append(
        WorkflowStep(
            actor="RefundSummaryNode",
            kind="final",
            text=final_response,
            is_final=True,
        )
    )

    return ScenarioRunResult(
        scenario_name="Console Preview",
        request=request,
        steps=steps,
        final_response=final_response,
        review_decision=parsed_review.decision,
        refund_executed=parsed_review.decision == DECISION_APPROVE,
        session_id="console-preview",
    )


def print_run_result(result: ScenarioRunResult) -> None:
    print(f"## {result.scenario_name}")
    print(f"Session: {result.session_id}")
    print(f"Review decision: {result.review_decision or 'unknown'}")
    print(f"Refund executed: {'yes' if result.refund_executed else 'no'}")
    print("User:")
    print(f"  {result.request}")
    print("Agent Workflow:")
    for index, step in enumerate(result.steps, start=1):
        print(f"  {index}. [{step.actor}] [{step.kind}] {step.text}")
    print("Final Response:")
    print(f"  {result.final_response}")
    print()


def print_run_instructions() -> None:
    print("# ADK Web")
    print(
        "Run `uv run adk web hands_one_code_examples/human_in_the_loop_agent` "
        "from the repository root, then open http://127.0.0.1:8000."
    )
    print("Set `GOOGLE_API_KEY` before using the interactive ADK Web workflow.")
    print(
        "Use the sample request from DEFAULT_REQUEST and reply to the HITL prompt with "
        "JSON such as "
        "`{\"decision\":\"approve\",\"approved_amount\":149.99,"
        "\"notes\":\"Approved after validating duplicate charge evidence.\"}`."
    )
    print()


def main() -> None:
    load_environment_variables()
    preview = simulate_console_scenario(DEFAULT_REQUEST, DEFAULT_SUPERVISOR_REVIEW)
    print_run_result(preview)
    print_run_instructions()


root_agent = (
    build_refund_workflow()
    if all(
        dependency is not None
        for dependency in (Event, EventActions, RequestInput, InMemoryRunner, START, Workflow, node, types)
    )
    else None
)


if __name__ == "__main__":
    main()
