from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.tools import check_account_status, check_payment_status
from app.nodes import (
    account_node,
    billing_node,
    escalation_node,
    general_node,
    should_escalate,
    supervisor_node,
    technical_node,
)
from app.state import CustomerSupportState
from app.checkpoints import sqlite_checkpointer


def route_from_supervisor(state: CustomerSupportState) -> str:
    """Pure routing function: policy lives in should_escalate(), not here."""
    if should_escalate(state["routing_confidence"]):
        return "escalation"
    return state["route"]


workflow = StateGraph(CustomerSupportState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("escalation", escalation_node)

workflow.add_node("general", general_node)
workflow.add_node("technical", technical_node)
workflow.add_node("billing", billing_node)
workflow.add_node("account", account_node)

# ToolNode reads tool_calls from the last AIMessage, runs the matching function,
# and returns {"messages": [ToolMessage, ...]}. A bare function can't do that.
workflow.add_node("payment_tool", ToolNode([check_payment_status]))
workflow.add_node("account_tool", ToolNode([check_account_status]))

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "general": "general",
        "technical": "technical",
        "billing": "billing",
        "account": "account",
        "escalation": "escalation",
    },
)
for specialist in ("general", "technical", "escalation"):
    workflow.add_edge(specialist, END)

workflow.add_conditional_edges(
    "billing",
    tools_condition,
    {
        "tools": "payment_tool",
        END: END,
    },
)
workflow.add_conditional_edges(
    "account",
    tools_condition,
    {
        "tools": "account_tool",
        END: END,
    },
)

# Loop back: after the tool runs, the specialist LLM reads the ToolMessage
# and writes the final user-facing answer.
workflow.add_edge("payment_tool", "billing")
workflow.add_edge("account_tool", "account")

checkpointer = sqlite_checkpointer()
graph = workflow.compile(checkpointer=checkpointer)
