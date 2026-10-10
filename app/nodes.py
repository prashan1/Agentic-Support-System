from app.state import CustomerSupportState
from app.llm import get_llm
from langchain_core.messages import SystemMessage


GENERAL_SYSTEM_PROMPT = """You are a friendly and helpful customer support agent for an online learning platform.
Your job is to answer customer questions clearly, concisely, and professionally.
If you don't know the answer, admit it and suggest they contact a human agent."""


def general_node(state: CustomerSupportState):
    llm = get_llm()
    system_prompt = SystemMessage(content=GENERAL_SYSTEM_PROMPT)
    messages = [system_prompt] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [response], 'final_response': response.content}