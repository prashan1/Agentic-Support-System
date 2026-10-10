from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.nodes import (
    account_node,
    billing_node,
    general_node,
    supervisor_node,
    technical_node,
)
from app.state import CustomerSupportState


def route_from_supervisor(state: CustomerSupportState) -> str:
    """Pure routing function: reads state, returns the next node name. No LLM calls here."""
    return state["route"]


workflow = StateGraph(CustomerSupportState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("general", general_node)
workflow.add_node("technical", technical_node)
workflow.add_node("billing", billing_node)
workflow.add_node("account", account_node)

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "general": "general",
        "technical": "technical",
        "billing": "billing",
        "account": "account",
    },
)
for specialist in ("general", "technical", "billing", "account"):
    workflow.add_edge(specialist, END)

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
