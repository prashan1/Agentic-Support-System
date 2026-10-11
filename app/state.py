from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

Route = Literal["general", "technical", "billing", "account"]

class RouteDecision(BaseModel):
    """Typed output of the supervisor. The field descriptions are sent to the model."""

    route: Route = Field(description="The specialist that should handle the customer's request.")
    confidence: float = Field(
        ge=0,
        le=1,
        description="How confident you are in the chosen route, from 0 (guessing) to 1 (certain).",
    )


class CustomerSupportState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # Conversation transcript
    route: Route  # Chosen specialist
    escalated: bool  # Whether the query requires human intervention
    escalation_reason: str | None  # Why we escalated (None when not escalated)
    resolution_status: Literal["escalated", "resolved"]  # Resolution status
    routing_confidence: float  # Supervisor's confidence in the route (0..1)
    final_response: str  # LLM's reply to the customer
    current_agent: str  # Which node last acted (debugging / evals)
