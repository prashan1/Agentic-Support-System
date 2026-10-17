"""Streamlit test console for the whole support system.

Run:  streamlit run streamlit_app.py

Sidebar: pick a user (memory key), start a new conversation (thread), see and forget stored facts.
Main:    chat with the graph. A panel shows the last turn's route, memory, KB hits and escalation.
         When the graph pauses for a refund, the reviewer panel offers approve / reject / edit.

Note: local Qdrant allows one process at a time, so stop this app before running other scripts.
"""

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from app.graph import graph, store
from app.store import forget_user, memory_namespace
from app.tools import SPECIALIST_TOOLS

st.set_page_config(page_title="Support agent lab")

# Only these nodes write the customer-facing reply; the supervisor (structured output) and the
# memory extractor also call the LLM, but their tokens must not appear in the chat.
REPLY_NODES = {"general", "technical", "billing", "account"}


def config_for(user_id: str, thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id, "user_id": user_id}, "recursion_limit": 12}


def pending_reviews(config: dict) -> list[dict]:
    snapshot = graph.get_state(config)
    return [i.value for task in snapshot.tasks for i in task.interrupts]


def new_thread() -> None:
    st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:6]}"
    st.session_state.last = None


if "thread_id" not in st.session_state:
    new_thread()


def run_turn(graph_input) -> None:
    """Run the graph until it finishes or pauses, streaming the reply tokens into the chat."""

    def reply_tokens():
        for chunk, meta in graph.stream(graph_input, config=config, stream_mode="messages"):
            if meta.get("langgraph_node") in REPLY_NODES and isinstance(chunk.content, str) and chunk.content:
                yield chunk.content

    with st.chat_message("assistant"):
        st.write_stream(reply_tokens())

    # Read the final state from the checkpoint (same data invoke() used to return).
    state = graph.get_state(config)
    last = dict(state.values)
    if any(task.interrupts for task in state.tasks):
        last["__interrupt__"] = True
    st.session_state.last = last
    st.rerun()

# ---------- sidebar: who, which conversation, what is remembered ----------
with st.sidebar:
    st.header("Session")
    user_id = st.text_input("user_id (memory key)", value="demo-user")
    st.button("New conversation", on_click=new_thread)
    st.caption(f"thread_id: `{st.session_state.thread_id}`")

    st.subheader("Billing tools (from MCP server)")
    billing_tool_names = [t.name for t in SPECIALIST_TOOLS["billing"]]
    if billing_tool_names:
        st.success(", ".join(billing_tool_names))
    else:
        st.error("None discovered: billing runs in degraded mode. Start the server, set BILLING_MCP_URL, restart the app.")

    st.subheader("Long-term memory")
    facts = [i.value["fact"] for i in store.search(memory_namespace(user_id), limit=50)]
    if facts:
        for fact in facts:
            st.write(f"- {fact}")
    else:
        st.caption("No facts stored for this user yet.")
    if st.button("Forget this user"):
        st.toast(f"Deleted {forget_user(store, user_id)} memories.")
        st.rerun()

config = config_for(user_id, st.session_state.thread_id)


# ---------- main: conversation history ----------
st.title("Support agent lab")
for message in graph.get_state(config).values.get("messages", []):
    if isinstance(message, HumanMessage):
        st.chat_message("user").write(message.content)
    elif isinstance(message, AIMessage) and message.content:
        st.chat_message("assistant").write(message.content)
    elif isinstance(message, ToolMessage):
        st.caption(f"tool result: {message.content}")


# ---------- human review panel (only when the graph is paused) ----------
def send_decision(answer: dict) -> None:
    run_turn(Command(resume=answer))


reviews = pending_reviews(config)
if reviews:
    st.warning("Paused for human review: the agent wants to run a sensitive action.")
    for call in reviews[0]["calls"]:
        st.json(call)
    approve_col, reject_col, edit_col = st.columns(3)
    if approve_col.button("Approve", type="primary"):
        send_decision({"decision": "approve"})
    reason = reject_col.text_input("Reject reason", value="declined by reviewer")
    if reject_col.button("Reject"):
        send_decision({"decision": "reject", "reason": reason})
    amount = edit_col.number_input("Refund amount", min_value=0.0, value=100.0)
    if edit_col.button("Edit and approve"):
        send_decision({"decision": "edit", "args": {"amount": amount}})
else:
    prompt = st.chat_input("Ask the support agent")
    if prompt:
        st.chat_message("user").write(prompt)
        run_turn({"messages": [HumanMessage(content=prompt)]})


# ---------- debug panel: what happened on the last turn ----------
last = st.session_state.last
if last:
    with st.expander("Debug: last turn", expanded=True):
        left, right = st.columns(2)
        left.metric("Agent", last.get("current_agent", "-"))
        right.metric("Route confidence", f"{last.get('routing_confidence', 0):.2f}")
        st.write(f"**Route:** {last.get('route')}  |  **Escalated:** {last.get('escalated')}")
        if last.get("escalation_reason"):
            st.caption(last["escalation_reason"])
        st.write("**Memory used:**", last.get("retrieved_memory") or "none")
        docs = last.get("retrieved_documents") or []
        st.write("**KB hits (this thread's last retrieval):**",
                 [d.metadata["doc_id"] for d in docs] or "none")
        if "__interrupt__" in last:
            st.write("**Interrupted:** yes")
