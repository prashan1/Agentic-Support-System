
from langgraph.graph.message import add_messages
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langchain_core.messages import AnyMessage

class CustomerSupportState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # Conversation transcript
    final_response: str  # LLM's reply to the customer