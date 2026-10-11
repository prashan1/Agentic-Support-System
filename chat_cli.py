"""Interactive chat on ONE thread, so follow-ups remember earlier turns (preview of M6).

Run:  python chat_cli.py      (Ctrl+C or type 'quit' to exit)
"""

import sys

from langchain_core.messages import HumanMessage

from app.graph import graph

sys.stdout.reconfigure(encoding="utf-8")

# recursion_limit is a TOP-LEVEL config key (not inside "configurable").
config = {"configurable": {"thread_id": "chat-cli-1"}, "recursion_limit": 10}

while True:
    query = input("User: ").strip()
    if query.lower() in {"quit", "exit", ""}:
        break
    result = graph.invoke({"messages": [HumanMessage(content=query)]}, config=config)
    print(f"[{result['current_agent']}] AI: {result['final_response']}\n")
