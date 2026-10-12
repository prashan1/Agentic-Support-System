from typing import Callable, List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.llm import get_llm
from langchain_core.messages import filter_messages
from app.tools import check_account_status, check_payment_status
from app.prompts import (
    ACCOUNT_SYSTEM_PROMPT,
    BILLING_SYSTEM_PROMPT,
    CONTEXT_PROMPT_TEMPLATE,
    GENERAL_SYSTEM_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
    TECHNICAL_SYSTEM_PROMPT,
)
from app.state import CustomerSupportState, RouteDecision
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


general_node = make_specialist_node("general", GENERAL_SYSTEM_PROMPT)
technical_node = make_specialist_node("technical", TECHNICAL_SYSTEM_PROMPT, use_context=True)
billing_node = make_specialist_node(
    "billing", BILLING_SYSTEM_PROMPT, [check_payment_status], use_context=True
)
account_node = make_specialist_node("account", ACCOUNT_SYSTEM_PROMPT, [check_account_status])
