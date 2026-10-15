"""Human-in-the-loop gate for sensitive tool calls (M11).

route_billing  : pure router, replaces tools_condition for billing.
approval_node  : pauses with interrupt(), then routes by the human's decision.

Light imports on purpose (no LLM, no retriever), so both are easy to test.
"""

from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command, interrupt

from app.tools import SENSITIVE_TOOLS


def needs_approval(message) -> bool:
    """True if the AI message asks for any sensitive tool (policy: SENSITIVE_TOOLS)."""
    return any(call["name"] in SENSITIVE_TOOLS for call in getattr(message, "tool_calls", None) or [])


def route_billing(state) -> Literal["approval", "billing_tools", "remember"]:
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return "remember"  # plain answer: done
    return "approval" if needs_approval(last) else "billing_tools"


def approval_node(state) -> Command[Literal["billing_tools", "billing"]]:
    ai_message = state["messages"][-1]

    # interrupt() FIRST: on resume this whole node re-runs from line 1, so nothing with a
    # side effect may come before it. The payload is what the reviewer sees.
    answer = interrupt(
        {
            "action": "approve_tool_calls",
            "calls": [{"name": c["name"], "args": c["args"]} for c in ai_message.tool_calls],
        }
    )
    decision = answer.get("decision")

    if decision == "approve":
        return Command(goto="billing_tools")

    if decision == "edit":
        # Overwrite the AI message (same id => add_messages replaces it) with the reviewer's
        # args for the sensitive call. The tool itself still validates them (bounds, idempotency).
        edits = answer.get("args", {})
        calls = [
            {**c, "args": {**c["args"], **edits}} if c["name"] in SENSITIVE_TOOLS else c
            for c in ai_message.tool_calls
        ]
        edited = ai_message.model_copy(update={"tool_calls": calls})
        return Command(goto="billing_tools", update={"messages": [edited]})

    if decision == "reject":
        reason = answer.get("reason") or "no reason given"
        # Every tool call needs a ToolMessage, or the provider rejects the next request.
        rejected = [
            ToolMessage(
                content=f"Rejected by human reviewer ({reason}). Do not retry. "
                "Tell the customer a reviewer declined and what they can do next.",
                tool_call_id=c["id"],
                name=c["name"],
            )
            for c in ai_message.tool_calls
        ]
        return Command(goto="billing", update={"messages": rejected})

    # Never fall through to "approve" on a malformed answer.
    raise ValueError(f"Unknown approval decision: {decision!r}")
