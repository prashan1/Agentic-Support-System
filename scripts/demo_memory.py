"""Long-term memory demo. Run: python -m scripts.demo_memory

Monday chat saves facts; Thursday is a NEW thread (empty checkpointer) with the same user_id.
A different user asks the same question: they must not see Alice's facts.
"""

import sys
import uuid

from langchain_core.messages import HumanMessage

from app.graph import graph, store
from app.store import forget_user, memory_namespace

sys.stdout.reconfigure(encoding="utf-8")

TAG = uuid.uuid4().hex[:6]
ALICE, BOB = f"alice-{TAG}", f"bob-{TAG}"


def chat(user_id: str, thread: str, text: str) -> None:
    config = {
        "configurable": {"thread_id": f"{thread}-{TAG}", "user_id": user_id},
        "recursion_limit": 12,
        "tags": ["memory-demo", TAG],
    }
    result = graph.invoke({"messages": [HumanMessage(content=text)]}, config=config)
    print(f"DEMO [{user_id} / {thread}] Q: {text}")
    print(f"DEMO   agent={result['current_agent']} memory_used={result.get('retrieved_memory')}")
    print(f"DEMO   A: {result['final_response']}\n")


def stored(user_id: str) -> list[str]:
    return [i.value["fact"] for i in store.search(memory_namespace(user_id), limit=50)]


chat(ALICE, "monday",
     "I'm on the annual plan and I use the Android app. My lesson video keeps buffering, "
     "and please keep your answers very short.")
print("DEMO stored for alice:", stored(ALICE), "\n")

chat(ALICE, "thursday", "Remind me, which plan am I on and which device do I use?")
chat(BOB, "thursday", "Remind me, which plan am I on and which device do I use?")

print("DEMO forget alice ->", forget_user(store, ALICE), "memories deleted")
print("DEMO stored for alice now:", stored(ALICE))
