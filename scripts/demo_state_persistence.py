"""Demo (no LLM): which state values survive between invokes on a thread?

TypedDict state has NO default values. A field holds the last value any node wrote.
With a checkpointer, the saved state is loaded at the start of every invoke on that thread.
"""

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class S(TypedDict, total=False):
    mode: str
    escalated: bool


def node(state: S):
    if state["mode"] == "escalate":
        return {"escalated": True}
    return {}  # a node that does NOT write 'escalated'


g = StateGraph(S)
g.add_node("n", node)
g.add_edge(START, "n")
g.add_edge("n", END)
graph = g.compile(checkpointer=InMemorySaver())

t1 = {"configurable": {"thread_id": "t1"}}
t2 = {"configurable": {"thread_id": "t2"}}

print("t1 turn 1 (node writes escalated=True):", graph.invoke({"mode": "escalate"}, t1))
print("t1 turn 2 (node writes nothing)       :", graph.invoke({"mode": "normal"}, t1))
print("t2 turn 1 (brand-new thread, no write) :", graph.invoke({"mode": "normal"}, t2))
print("\nNotice: t1 turn 2 still has escalated=True (stale). t2 has no 'escalated' key at all.")
