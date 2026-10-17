"""Tests for the HITL gate: pure router + a tiny graph with a real interrupt/resume. No LLM, no MCP.

The graph runs STUB billing tools defined here. The real ones live on the MCP server; the gate
only cares about tool names (SENSITIVE_TOOLS), so stubs are enough to test pause/approve/reject/edit.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from app.approval import approval_node, route_billing

REFUNDED: set[str] = set()  # the stub's "side effect": what a real refund would have changed


@tool
def check_payment_status(transaction_id: str) -> str:
    """Stub: look up a payment."""
    return f"Payment status: completed for {transaction_id}."


@tool
def issue_refund(transaction_id: str, amount: float | None = None) -> str:
    """Stub: queue a refund. Refuses amounts above 499 like the real service."""
    if amount is not None and not 0 < amount <= 499:
        return f"Cannot refund {amount}: more than was paid."
    REFUNDED.add(transaction_id)
    return f"Refund of {499.0 if amount is None else amount} queued for {transaction_id}."


REFUND = {"name": "issue_refund", "args": {"transaction_id": "TXN002"}, "id": "c1"}
CHECK = {"name": "check_payment_status", "args": {"transaction_id": "TXN002"}, "id": "c2"}


@pytest.fixture(autouse=True)
def clean_refunds():
    REFUNDED.clear()
    yield
    REFUNDED.clear()


# ---- Step 2: router ----
def test_router_plain_answer_goes_to_remember():
    assert route_billing({"messages": [AIMessage(content="hi")]}) == "remember"


def test_router_read_only_tool_skips_approval():
    assert route_billing({"messages": [AIMessage(content="", tool_calls=[CHECK])]}) == "billing_tools"


def test_router_sensitive_tool_needs_approval():
    assert route_billing({"messages": [AIMessage(content="", tool_calls=[REFUND])]}) == "approval"


def test_router_mixed_calls_need_approval():
    assert route_billing({"messages": [AIMessage(content="", tool_calls=[CHECK, REFUND])]}) == "approval"


# ---- Step 3: approval node inside a tiny graph (stub 'billing' instead of the LLM) ----
def build_app():
    g = StateGraph(MessagesState)
    g.add_node("approval", approval_node)
    g.add_node("billing_tools", ToolNode([check_payment_status, issue_refund]))
    g.add_node("billing", lambda state: {"messages": [AIMessage(content="final reply")]})
    g.add_edge(START, "approval")
    g.add_edge("billing_tools", "billing")
    g.add_edge("billing", END)
    return g.compile(checkpointer=InMemorySaver())


def start(app, calls=(REFUND,), thread="t1"):
    config = {"configurable": {"thread_id": thread}}
    first = app.invoke(
        {"messages": [HumanMessage(content="refund TXN002"), AIMessage(content="", tool_calls=list(calls), id="ai1")]},
        config=config,
    )
    return config, first


def resume(app, config, **answer):
    return app.invoke(Command(resume=answer), config=config)


def test_pause_happens_before_any_side_effect():
    app = build_app()
    _, first = start(app)
    assert "__interrupt__" in first
    assert first["__interrupt__"][0].value["calls"][0]["name"] == "issue_refund"
    assert REFUNDED == set()  # nothing refunded while waiting


def test_approve_runs_the_refund():
    app = build_app()
    config, _ = start(app)
    result = resume(app, config, decision="approve")
    assert "TXN002" in REFUNDED
    assert any(isinstance(m, ToolMessage) and "queued" in m.content for m in result["messages"])
    assert result["messages"][-1].content == "final reply"


def test_reject_never_runs_the_refund_and_answers_every_call():
    app = build_app()
    config, _ = start(app, calls=(CHECK, REFUND))
    result = resume(app, config, decision="reject", reason="outside window")
    assert REFUNDED == set()
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert {m.tool_call_id for m in tool_msgs} == {"c1", "c2"}  # one ToolMessage per tool call
    assert all("Rejected" in m.content for m in tool_msgs)
    assert result["messages"][-1].content == "final reply"


def test_edit_overwrites_the_ai_message_and_uses_new_amount():
    app = build_app()
    config, _ = start(app)
    result = resume(app, config, decision="edit", args={"amount": 100})
    ai_with_calls = [m for m in result["messages"] if isinstance(m, AIMessage) and m.tool_calls]
    assert len(ai_with_calls) == 1  # replaced, not appended
    assert ai_with_calls[0].tool_calls[0]["args"]["amount"] == 100
    assert any(isinstance(m, ToolMessage) and "100" in m.content for m in result["messages"])


def test_edit_cannot_bypass_tool_validation():
    app = build_app()
    config, _ = start(app)
    result = resume(app, config, decision="edit", args={"amount": 100000})
    assert REFUNDED == set()  # the tool still refuses an amount above what was paid
    assert any(isinstance(m, ToolMessage) and "Cannot refund" in m.content for m in result["messages"])


def test_unknown_decision_does_not_default_to_approve():
    app = build_app()
    config, _ = start(app)
    with pytest.raises(ValueError):
        resume(app, config, decision="maybe")
    assert REFUNDED == set()
