from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from app.state import CustomerSupportState
from app.nodes import general_node

# Create the workflow
workflow = StateGraph(CustomerSupportState)
workflow.add_node('general', general_node)
workflow.set_entry_point("general")
workflow.set_finish_point("general")

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
