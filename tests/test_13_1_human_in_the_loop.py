from __future__ import annotations

import pathlib
import sys
import types as builtin_types
import uuid
from contextlib import contextmanager

import pytest

from tests._support.fake_adk import load_module_with_fake_adk, make_event, run


MODULE_PATH = pathlib.Path("hands_one_code_examples/13_1_human_in_the_loop.py")
FAKE_MODULE_NAMES = (
    "google",
    "google.adk",
    "google.adk.agents",
    "google.adk.agents.invocation_context",
    "google.adk.events",
    "google.adk.runners",
    "google.adk.tools",
    "google.adk.workflow",
    "google.genai",
)


@contextmanager
def load_example():
    module_name = f"human_in_the_loop_example_{uuid.uuid4().hex}"
    saved_modules = {name: sys.modules.get(name) for name in FAKE_MODULE_NAMES}
    saved_module = sys.modules.get(module_name)
    try:
        module = load_module_with_fake_adk(
            module_path=MODULE_PATH,
            module_name=module_name,
            include_workflow=True,
        )
        yield module
    finally:
        sys.modules.pop(module_name, None)
        if saved_module is not None:
            sys.modules[module_name] = saved_module
        for name, original in saved_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_policy_engine_builds_manual_review_packet_for_high_value_refund():
    with load_example() as module:
        packet = module.POLICY_ENGINE.build_review_packet(module.DEFAULT_REQUEST)

        assert packet["refund_request"]["order_id"] == "ORD-2049"
        assert packet["refund_request"]["amount"] == pytest.approx(149.99)
        assert packet["assessment"]["recommended_action"] == "manual-review"
        assert packet["assessment"]["risk_flags"] == [
            "Amount exceeds the auto-approval threshold of $100.00."
        ]


def test_request_supervisor_review_yields_request_input_and_updates_state():
    with load_example() as module:
        context = builtin_types.SimpleNamespace(
            state=module.build_initial_state(module.DEFAULT_REQUEST)
        )

        generator = module.request_supervisor_review(module.DEFAULT_REQUEST, context)
        state_event = next(generator)
        request_input = next(generator)

        assert state_event.actions.state_delta[module.STATE_REVIEW_PACKET][
            "refund_request"
        ]["order_id"] == "ORD-2049"
        assert request_input.interrupt_id == "refund-review-ord-2049"
        assert "respond with JSON" in request_input.message
        assert request_input.payload["refund_request"]["order_id"] == "ORD-2049"
        assert (
            request_input.response_schema["properties"]["decision"]["enum"]
            == [module.DECISION_APPROVE, module.DECISION_REJECT]
        )
        assert context.state[module.STATE_REVIEW_PACKET]["assessment"][
            "recommended_action"
        ] == "manual-review"


def test_record_supervisor_review_rebuilds_missing_packet_after_adk_web_resume():
    with load_example() as module:
        context = builtin_types.SimpleNamespace(
            state=module.build_initial_state(module.DEFAULT_REQUEST)
        )
        context.state.pop(module.STATE_REVIEW_PACKET, None)

        message = module.record_supervisor_review(
            {
                "decision": "approve",
                "approved_amount": 149.99,
                "notes": "Approved after checking duplicate charge evidence.",
            },
            context,
        )

        assert "Supervisor decision recorded: approve" in message
        assert context.state[module.STATE_REVIEW_PACKET]["refund_request"][
            "order_id"
        ] == "ORD-2049"
        assert context.state[module.STATE_SUPERVISOR_REVIEW]["decision"] == "approve"


def test_parse_supervisor_review_rejects_amounts_above_requested_total():
    with load_example() as module:
        with pytest.raises(ValueError, match="must not exceed the requested amount"):
            module.parse_supervisor_review(
                {
                    "decision": "approve",
                    "approved_amount": 200,
                    "notes": "Too much",
                },
                requested_amount=149.99,
            )


def test_build_workflow_wires_hitl_router_execution_and_summary_nodes():
    with load_example() as module:
        demo = module.RefundHumanInTheLoopDemo.build(
            app_name="refund-hitl-app",
            user_id="alice",
            session_id="refund-session",
        )

        assert demo.execution_node.name == "RefundExecutionNode"
        assert demo.workflow.edges[0][0].name == "START"
        assert demo.workflow.edges[0][1] is module.initialize_refund_request
        assert demo.workflow.edges[1][1] is module.request_supervisor_review
        assert demo.workflow.edges[2][1] is module.record_supervisor_review
        assert demo.workflow.edges[3][1] is module.refund_decision_router
        assert demo.workflow.edges[4][1][True] is module.execute_refund
        assert demo.workflow.edges[4][1][False] is module.refund_summary_node


def test_run_request_approved_path_collects_hitl_execution_and_final_steps():
    with load_example() as module:
        demo = module.RefundHumanInTheLoopDemo.build(
            app_name="refund-hitl-app",
            user_id="alice",
            session_id="refund-session",
        )
        demo.runner.queue_events(
            [
                make_event(module, author="user", text=module.DEFAULT_REQUEST),
                make_event(
                    module,
                    author="SupervisorReviewNode",
                    text="Review the prepared refund packet and respond with JSON.",
                ),
                make_event(
                    module,
                    author="RefundDecisionRouter",
                    text=(
                        "Supervisor approved the refund. Route to the execution agent."
                    ),
                    route=True,
                ),
                make_event(
                    module,
                    author="RefundExecutionNode",
                    text="Refund executed for ORD-2049 in the amount of $149.99.",
                ),
                make_event(
                    module,
                    author="RefundSummaryNode",
                    text=(
                        "Refund issued for ORD-2049 in the amount of $149.99.\n"
                        "Transaction id: refund-ord-2049."
                    ),
                    is_final=True,
                ),
            ]
        )

        result = run(
            demo.run_request(
                module.DEFAULT_REQUEST,
                scenario_name="Approved Refund",
                session_index=0,
            )
        )

        assert result.review_decision == module.DECISION_APPROVE
        assert result.refund_executed is True
        assert result.session_id == "refund-session-1"
        assert [step.kind for step in result.steps] == [
            "hitl-request",
            "route",
            "execution",
            "final",
        ]
        assert result.final_response.startswith("Refund issued for ORD-2049")
        assert demo.runner.session_service.create_calls == [
            {
                "app_name": "refund-hitl-app",
                "user_id": "alice",
                "session_id": "refund-session-1",
                "state": module.build_initial_state(module.DEFAULT_REQUEST),
            }
        ]


def test_run_request_rejected_path_skips_refund_execution():
    with load_example() as module:
        demo = module.RefundHumanInTheLoopDemo.build(
            app_name="refund-hitl-app",
            user_id="alice",
            session_id="refund-session",
        )
        demo.runner.queue_events(
            [
                make_event(module, author="user", text=module.DEFAULT_REQUEST),
                make_event(
                    module,
                    author="SupervisorReviewNode",
                    text="Review the prepared refund packet and respond with JSON.",
                ),
                make_event(
                    module,
                    author="RefundDecisionRouter",
                    text="Supervisor rejected the refund. Route directly to the summary node.",
                    route=False,
                ),
                make_event(
                    module,
                    author="RefundSummaryNode",
                    text=(
                        "Refund for ORD-2049 was rejected by the supervisor.\n"
                        "Reason: Duplicate charge evidence was insufficient."
                    ),
                    is_final=True,
                ),
            ]
        )

        result = run(
            demo.run_request(
                module.DEFAULT_REQUEST,
                scenario_name="Rejected Refund",
                session_index=1,
            )
        )

        assert result.review_decision == module.DECISION_REJECT
        assert result.refund_executed is False
        assert [step.kind for step in result.steps] == [
            "hitl-request",
            "route",
            "final",
        ]
        assert "rejected by the supervisor" in result.final_response
