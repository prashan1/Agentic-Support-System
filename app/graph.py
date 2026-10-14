from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.tools import SPECIALIST_TOOLS
from app.nodes import (
    account_node,
    billing_node,
    escalation_node,
    general_node,
    load_memory,
    remember_node,
    retrieve_node,
    should_escalate,
    supervisor_node,
    technical_node,
)
from app.state import CustomerSupportState
from app.checkpoints import sqlite_checkpointer
from app.store import sqlite_store


def route_from_supervisor(state: CustomerSupportState) -> str:
    """Pure routing function: policy lives in should_escalate(), not here."""
    if should_escalate(state["routing_confidence"]):
        return "escalation"
    return state["route"]


workflow = StateGraph(CustomerSupportState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("escalation", escalation_node)
workflow.add_node("load_memory", load_memory)


workflow.add_node("general", general_node)
workflow.add_node("technical", technical_node)
workflow.add_node("billing", billing_node)
workflow.add_node("account", account_node)
workflow.add_node("retrieve", retrieve_node)

workflow.add_edge(START, "load_memory")
workflow.add_edge("load_memory", "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "general": "general",
        "technical": "retrieve",
        "billing": "retrieve",
        "account": "account",
        "escalation": "escalation",
    },
)

workflow.add_edge("escalation", END)

workflow.add_conditional_edges("retrieve", lambda state: state["route"], {
    "technical": "technical",
    "billing": "billing",
})

# WRITE side of memory: after any specialist's final reply, extract durable facts.
workflow.add_node("remember", remember_node)
workflow.add_edge("remember", END)

# Specialists with real tools loop through a ToolNode, then come back to write the reply.
# A reply with no tool call goes to "remember". Specialists without tools skip the loop.
for name, tools in SPECIALIST_TOOLS.items():
    if tools:
        workflow.add_node(f"{name}_tools", ToolNode(tools))
        workflow.add_conditional_edges(name, tools_condition, {"tools": f"{name}_tools", END: "remember"})
        workflow.add_edge(f"{name}_tools", name)
    else:
        workflow.add_edge(name, "remember")

checkpointer = sqlite_checkpointer()
store = sqlite_store()
graph = workflow.compile(checkpointer=checkpointer, store=store)
