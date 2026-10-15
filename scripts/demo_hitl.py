"""Human-in-the-loop demo. Each command is a SEPARATE process, so resuming proves the pause
lives in the SQLite checkpoint, not in memory.

  python -m scripts.demo_hitl ask                     # customer asks; graph pauses; prints THREAD
  python -m scripts.demo_hitl pending THREAD          # what is waiting for review
  python -m scripts.demo_hitl approve THREAD
  python -m scripts.demo_hitl reject THREAD "reason"
  python -m scripts.demo_hitl edit THREAD 100         # approve a partial refund of 100
"""

import sys
import uuid

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command

from app.graph import graph

sys.stdout.reconfigure(encoding="utf-8")


def config_for(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id, "user_id": "demo-user"}, "recursion_limit": 12}


def pending(config: dict) -> list:
    """Interrupt payloads waiting on this thread (read from the checkpoint)."""
    snapshot = graph.get_state(config)
    return [i.value for task in snapshot.tasks for i in task.interrupts]


def show(result: dict) -> None:
    for m in result["messages"]:
        if isinstance(m, ToolMessage):
            print(f"  tool result: {m.content}")
    print(f"  [{result['current_agent']}] {result['final_response']}")


def ask() -> None:
    thread_id = f"hitl-{uuid.uuid4().hex[:6]}"
    config = config_for(thread_id)
    text = "I was charged twice. Please refund transaction TXN002."
    result = graph.invoke({"messages": [HumanMessage(content=text)]}, config=config)
    print(f"THREAD {thread_id}")
    if "__interrupt__" in result:
        print("  graph paused. Customer sees: 'Your refund request is awaiting review.'")
        print("  Reviewer sees:", pending(config))
    else:
        print("  no pause (the model did not request issue_refund):")
        show(result)


def resume(thread_id: str, answer: dict) -> None:
    config = config_for(thread_id)
    if not pending(config):
        sys.exit(f"Nothing is waiting on thread {thread_id}.")
    show(graph.invoke(Command(resume=answer), config=config))


if __name__ == "__main__":
    command, *args = sys.argv[1:] or ["ask"]
    needs_thread = {"pending", "approve", "reject", "edit"}
    if command in needs_thread and not args:
        sys.exit(f"'{command}' needs the THREAD id printed by 'ask'.\n\n{__doc__}")
    if command == "edit" and len(args) < 2:
        sys.exit("'edit' needs a THREAD and a refund amount: edit THREAD 100")
    if command == "ask":
        ask()
    elif command == "pending":
        print(pending(config_for(args[0])))
    elif command == "approve":
        resume(args[0], {"decision": "approve"})
    elif command == "reject":
        resume(args[0], {"decision": "reject", "reason": args[1] if len(args) > 1 else "declined"})
    elif command == "edit":
        resume(args[0], {"decision": "edit", "args": {"amount": float(args[1])}})
    else:
        sys.exit(__doc__)
