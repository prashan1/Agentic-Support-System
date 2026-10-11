from typing import Callable, List
from langchain_core.messages import AIMessage, SystemMessage

from app.llm import get_llm
from app.tools import check_account_status, check_payment_status
from app.prompts import (
    ACCOUNT_SYSTEM_PROMPT,
    BILLING_SYSTEM_PROMPT,
    GENERAL_SYSTEM_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
    TECHNICAL_SYSTEM_PROMPT,
)
from app.state import CustomerSupportState, RouteDecision

# Escalation policy lives in code, not in a prompt.
CONFIDENCE_THRESHOLD = 0.70
ESCALATION_MESSAGE = (
    "Thanks for reaching out. I'm connecting you with a human support agent "
    "who will follow up with you."
)


def should_escalate(confidence: float, threshold: float = CONFIDENCE_THRESHOLD) -> bool:
    """Pure policy function: easy to unit test and to sweep thresholds against."""
    return confidence < threshold


def make_specialist_node(name: str, prompt: str, tool_list: List[Callable] | None = None):
    system_message = SystemMessage(content=prompt)  # built once
    
    def specialist_node(state: CustomerSupportState):
        if tool_list:
            llm = get_llm(max_tokens=600).bind_tools(tool_list)
        else:
            llm = get_llm(max_tokens=600)
        response = llm.invoke([system_message] + state["messages"])
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


def supervisor_node(state: CustomerSupportState):
    llm = get_llm().with_structured_output(RouteDecision)
    decision = llm.invoke([SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)] + state["messages"])
    return {
        "route": decision.route,
        "routing_confidence": decision.confidence,
        "current_agent": "supervisor",
    }


general_node = make_specialist_node("general", GENERAL_SYSTEM_PROMPT)
technical_node = make_specialist_node("technical", TECHNICAL_SYSTEM_PROMPT)
billing_node = make_specialist_node("billing", BILLING_SYSTEM_PROMPT, [check_payment_status])
account_node = make_specialist_node("account", ACCOUNT_SYSTEM_PROMPT, [check_account_status])
