"""Pure grading functions. No LLM, no graph: easy to unit test (see tests/test_evaluators.py).

A buggy grader gives confidently wrong scores, so these stay small and are tested on fake inputs.
"""

from langchain_core.messages import AIMessage, HumanMessage


def last_turn_messages(messages) -> list:
    """Messages produced after the most recent HumanMessage (the current turn only).

    With a checkpointer the state holds the whole thread, so grading must not look at old turns.
    """
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], HumanMessage):
            return list(messages[idx + 1:])
    return list(messages)


def first_tool_called(messages) -> str | None:
    """Name of the first tool the agent REQUESTED this turn (its decision), or None."""
    for message in last_turn_messages(messages):
        if isinstance(message, AIMessage) and message.tool_calls:
            return message.tool_calls[0]["name"]
    return None


def score_route(expected, actual) -> bool | None:
    """None = not scored (label is None)."""
    if expected is None:
        return None
    return expected == actual


def score_tool(expected, actual) -> bool:
    """expected/actual are tool names, or None meaning 'no tool call'."""
    return expected == actual


def score_escalation(expected, actual) -> bool | None:
    """None = not scored (label is None)."""
    if expected is None:
        return None
    return expected == actual


def rate(flags) -> tuple[int, int]:
    """(passed, total) over flags, ignoring None (= not scored)."""
    scored = [f for f in flags if f is not None]
    return sum(1 for f in scored if f), len(scored)


def precision_recall(pairs) -> tuple[float | None, float | None]:
    """pairs = [(expected_bool, predicted_bool), ...]. Returns (precision, recall); None if undefined."""
    tp = sum(1 for e, p in pairs if e and p)
    predicted_pos = sum(1 for _, p in pairs if p)
    actual_pos = sum(1 for e, _ in pairs if e)
    precision = tp / predicted_pos if predicted_pos else None
    recall = tp / actual_pos if actual_pos else None
    return precision, recall
