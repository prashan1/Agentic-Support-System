"""M5 checkpoint: (1) call tools directly, (2) watch the agent-tool loop's message types."""

import sys

from langchain_core.messages import HumanMessage

from app.graph import graph
from app.tools import check_account_status, check_payment_status

sys.stdout.reconfigure(encoding="utf-8")

print("== 1. Tools called directly (no LLM) ==")
# @tool objects are invoked with a dict of arguments.
for tid in ("TXN002", "txn002 ", "TXN999"):
    print(f"  payment({tid!r}) -> {check_payment_status.invoke({'transaction_id': tid})}")
for aid in ("ACC002", "ACC999"):
    print(f"  account({aid!r}) -> {check_account_status.invoke({'account_id': aid})}")

print("\n== 2. Agent-tool loop through the full graph ==")
CASES = [
    "Payment deducted but course not activated. TXN002",  # expect tool call
    "I can't log in, account ACC002",                      # expect tool call (account)
    "My payment failed",                                   # no ID: should ASK, not invent
    "Check transaction TXN999",                            # unknown ID: graceful reply
]

for i, query in enumerate(CASES):
    config = {"configurable": {"thread_id": f"tools-{i}"}, "recursion_limit": 10}
    result = graph.invoke({"messages": [HumanMessage(content=query)]}, config=config)
    kinds = [type(m).__name__ for m in result["messages"]]
    called = [tc["args"] for m in result["messages"] for tc in getattr(m, "tool_calls", [])]
    print(f"\nQ: {query}")
    print(f"  agent:    {result['current_agent']}")
    print(f"  messages: {kinds}")
    print(f"  tool args: {called or 'none'}")
    print(f"  A: {result['final_response']}")
