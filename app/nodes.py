from typing import Callable, List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.llm import get_llm
from langchain_core.messages import filter_messages
from langgraph.store.base import BaseStore

from app.store import memory_namespace, save_fact
from app.tools import SPECIALIST_TOOLS
from app.prompts import (
    ACCOUNT_SYSTEM_PROMPT,
    BILLING_SYSTEM_PROMPT,
    CONTEXT_PROMPT_TEMPLATE,
    MEMORY_EXTRACTOR_PROMPT,
    MEMORY_PROMPT_TEMPLATE,
    GENERAL_SYSTEM_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
    TECHNICAL_SYSTEM_PROMPT,
)
from langchain_core.runnables import RunnableConfig
from app.state import CustomerSupportState, MemoryFacts, RouteDecision
from app.rag.retriever import get_hybrid_retriever

# Escalation policy lives in code, not in a prompt.
CONFIDENCE_THRESHOLD = 0.70
ESCALATION_MESSAGE = (
    "Thanks for reaching out. I'm connecting you with a human support agent "
    "who will follow up with you."
)


def should_escalate(confidence: float, threshold: float = CONFIDENCE_THRESHOLD) -> bool:
    """Pure policy function: easy to unit test and to sweep thresholds against."""
    return confidence < threshold


def make_specialist_node(
    name: str, prompt: str, tool_list: List[Callable] | None = None, use_context: bool = False
):
    system_message = SystemMessage(content=prompt)  # built once

    def specialist_node(state: CustomerSupportState):
        if tool_list:
            llm = get_llm(max_tokens=600).bind_tools(tool_list)
        else:
            llm = get_llm(max_tokens=600)
        prefix = [system_message]
        if use_context and state.get("retrieved_documents"):
            prefix.append(SystemMessage(content=format_context(state["retrieved_documents"])))
        if state.get("retrieved_memory"):
            prefix.append(SystemMessage(content=format_memory(state["retrieved_memory"])))
        
        response = llm.invoke(prefix + state["messages"])
        return {
            "messages": [response],
            "final_response": response.content,
            "current_agent": name,
            "escalated": False,
            "escalation_reason": None,
            "resolution_status": "resolved",
        }

    return specialist_node


def escalation_node(state: CustomerSupportState):
    """Deterministic hand-off: no LLM call, so no invented promises and a factual reason."""
    confidence = state["routing_confidence"]
    return {
        "messages": [AIMessage(content=ESCALATION_MESSAGE)],  # keep transcript complete
        "final_response": ESCALATION_MESSAGE,
        "escalated": True,
        "escalation_reason": (
            f"Low routing confidence: {confidence:.2f} "
            f"(threshold {CONFIDENCE_THRESHOLD:.2f}, suggested route: {state['route']})"
        ),
        "resolution_status": "escalated",
        "current_agent": "escalation",
    }


ROUTER_CONTEXT_MESSAGES = 6


def recent_context(messages, n: int = ROUTER_CONTEXT_MESSAGES):
    """Last n plain conversation turns (human + AI text) for routing.

    Tool plumbing is dropped on purpose: ToolMessages are removed, and so are AI
    messages that only carry tool_calls (a tool_call without its ToolMessage is
    rejected by the provider).
    """
    turns = [
        m
        for m in filter_messages(messages, include_types=[HumanMessage, AIMessage])
        if not getattr(m, "tool_calls", None)
    ]
    return turns[-n:]


def supervisor_node(state: CustomerSupportState):
    llm = get_llm().with_structured_output(RouteDecision)
    decision = llm.invoke([SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)] + recent_context(state["messages"]))
    return {
        "route": decision.route,
        "routing_confidence": decision.confidence,
        "current_agent": "supervisor",
    }


def retrieve_node(state: CustomerSupportState):
    """Search the KB category that matches the route (technical or billing only)."""
    query = state["messages"][-1].content  # latest customer message
    documents = get_hybrid_retriever(state["route"]).invoke(query)
    return {"retrieved_documents": documents, "current_agent": "retrieve"}


def format_context(documents) -> str:
    """KB entries as a system message so the specialist answers from them."""
    entries = "\n\n".join(f"[{d.metadata['doc_id']}] {d.page_content}" for d in documents)
    return CONTEXT_PROMPT_TEMPLATE.format(entries=entries)

def format_memory(facts: list[str]) -> str:
    """This user's remembered facts as a system message."""
    return MEMORY_PROMPT_TEMPLATE.format(retrieved_memories="\n".join(f"- {f}" for f in facts))


def load_memory(state: CustomerSupportState, config: RunnableConfig, *, store: BaseStore):
    """READ side of memory: top-3 facts about this user that match the message. No LLM call."""
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:  # anonymous session (evals, scripts): no memory, not a crash
        return {"retrieved_memory": [], "current_agent": "load_memory"}
    query = state["messages"][-1].content
    items = store.search(memory_namespace(user_id), query=query, limit=3)
    return {"retrieved_memory": [i.value["fact"] for i in items], "current_agent": "load_memory"}


def remember_node(state: CustomerSupportState, config: RunnableConfig, *, store: BaseStore):
    """WRITE side of memory: runs after a specialist's reply, on the customer's latest message.

    One small structured-output call. Saves nothing for anonymous sessions (no user_id),
    so evals and scripts without a user never pay for it. Returns {} so current_agent
    keeps naming the specialist that answered.
    """
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return {}
    last_human = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if last_human is None:
        return {}
    extractor = get_llm().with_structured_output(MemoryFacts)
    extracted = extractor.invoke([SystemMessage(content=MEMORY_EXTRACTOR_PROMPT), last_human])
    for fact in extracted.facts:
        save_fact(store, user_id, fact)  # card-number guard lives in save_fact
    return {}


general_node = make_specialist_node("general", GENERAL_SYSTEM_PROMPT, SPECIALIST_TOOLS["general"])
technical_node = make_specialist_node(
    "technical", TECHNICAL_SYSTEM_PROMPT, SPECIALIST_TOOLS["technical"], use_context=True
)
billing_node = make_specialist_node(
    "billing", BILLING_SYSTEM_PROMPT, SPECIALIST_TOOLS["billing"], use_context=True
)
account_node = make_specialist_node("account", ACCOUNT_SYSTEM_PROMPT, SPECIALIST_TOOLS["account"])
