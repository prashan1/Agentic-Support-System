"""Unit tests for the graders. Fake messages only: no LLM, no network, runs in milliseconds."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from evals.evaluators import (
    first_tool_called,
    last_turn_messages,
    precision_recall,
    rate,
    score_escalation,
    score_route,
    score_tool,
)


def tool_request(name):
    return AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": "call_1"}])


def test_first_tool_called_finds_requested_tool():
    msgs = [HumanMessage(content="hi"), tool_request("check_payment_status"),
            ToolMessage(content="ok", tool_call_id="call_1"), AIMessage(content="done")]
    assert first_tool_called(msgs) == "check_payment_status"


def test_first_tool_called_none_when_no_tool():
    assert first_tool_called([HumanMessage(content="hi"), AIMessage(content="hello")]) is None


def test_first_tool_called_ignores_previous_turns():
    # Turn 1 used a tool; turn 2 did not. Grading turn 2 must say "no tool".
    msgs = [HumanMessage(content="turn 1"), tool_request("check_account_status"),
            ToolMessage(content="ok", tool_call_id="call_1"), AIMessage(content="answer 1"),
            HumanMessage(content="turn 2"), AIMessage(content="answer 2")]
    assert first_tool_called(msgs) is None


def test_last_turn_messages_slices_after_last_human():
    msgs = [HumanMessage(content="a"), AIMessage(content="b"), HumanMessage(content="c"), AIMessage(content="d")]
    assert [m.content for m in last_turn_messages(msgs)] == ["d"]


def test_score_route():
    assert score_route("billing", "billing") is True
    assert score_route("billing", "account") is False
    assert score_route(None, "anything") is None  # not scored


def test_score_tool_treats_none_as_no_tool():
    assert score_tool(None, None) is True
    assert score_tool(None, "check_payment_status") is False
    assert score_tool("check_payment_status", None) is False


def test_score_escalation():
    assert score_escalation(True, True) is True
    assert score_escalation(False, True) is False
    assert score_escalation(None, True) is None


def test_rate_ignores_unscored():
    assert rate([True, False, None, True]) == (2, 3)
    assert rate([None, None]) == (0, 0)


def test_precision_recall():
    # expected, predicted
    pairs = [(True, True), (True, False), (False, True), (False, False), (False, False)]
    precision, recall = precision_recall(pairs)
    assert precision == 0.5  # 1 of the 2 escalations was right
    assert recall == 0.5     # caught 1 of the 2 that should escalate


def test_precision_recall_undefined_cases():
    assert precision_recall([(False, False)]) == (None, None)
    assert precision_recall([(True, False)]) == (None, 0.0)  # never escalated: precision undefined
