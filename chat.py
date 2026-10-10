from langchain_core.messages import HumanMessage
from app.graph import graph


initial_state = {
    'messages': [HumanMessage(content="My name is Prashant, what is your name?")],
    'final_response': None
}
result = graph.invoke(initial_state, config={'configurable': {'thread_id': '123'}})
print(f"Route taken: general")
print(f"Final response: {result['final_response']}")