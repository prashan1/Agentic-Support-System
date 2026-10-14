"""Interactive chat on ONE thread, so follow-ups remember earlier turns (preview of M6).

Run:  python -m scripts.chat_cli      (type 'quit' to exit)
"""

import sys

from langchain_core.messages import HumanMessage

from app.graph import graph

sys.stdout.reconfigure(encoding="utf-8")

# recursion_limit is a TOP-LEVEL config key (not inside "configurable").
USER_ID = sys.argv[1] if len(sys.argv) > 1 else "demo-user"  # long-term memory key
config = {"configurable": {"thread_id": "chat-cli-2", "user_id": USER_ID}, "recursion_limit": 10}

while True:
    query = input("User: ").strip()
    if query.lower() in {"quit", "exit", ""}:
        break
    result = graph.invoke({"messages": [HumanMessage(content=query)]}, config=config)
    print(f"[{result['current_agent']}] AI: {result['final_response']}\n")
    print("memory:", result.get("retrieved_memory"))
