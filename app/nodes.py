from app.state import CustomerSupportState, RouteDecision
from app.llm import get_llm
from langchain_core.messages import SystemMessage
from app.prompts import (
    GENERAL_SYSTEM_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
    TECHNICAL_SYSTEM_PROMPT,
    BILLING_SYSTEM_PROMPT,
    ACCOUNT_SYSTEM_PROMPT,
)

def make_specialist_node(name: str, prompt: str):
    system_message = SystemMessage(content=prompt)   # built once

    def specialist_node(state: CustomerSupportState):
        llm = get_llm()
        response = llm.invoke([system_message] + state["messages"])
        return {
            "messages": [response],
            "final_response": response.content,
            "current_agent": name,
        }

    return specialist_node

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
billing_node = make_specialist_node("billing", BILLING_SYSTEM_PROMPT)
account_node = make_specialist_node("account", ACCOUNT_SYSTEM_PROMPT)