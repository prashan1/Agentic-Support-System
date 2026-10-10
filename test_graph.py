"""M3 checkpoint: run the full graph on labeled queries and measure routing accuracy.

Each case gets its OWN thread_id so checkpointed state cannot leak between cases.
"""

import sys

from langchain_core.messages import HumanMessage

from app.graph import graph

sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252

# (query, expected_route)
CASES = [
    ("Payment deducted but course not activated", "billing"),
    ("I want a refund for last month", "billing"),
    ("I can't log in", "account"),
    ("How do I change my email?", "account"),
    ("Video won't load", "technical"),
    ("The app crashes on startup", "technical"),
    ("What courses do you have?", "general"),
    ("Do you offer certificates?", "general"),
    ("Something is broken, please help", "general"),  # vague: watch the confidence
    ("I already completed my billing what is my account balance because app is lagging", None),  # mixed: no single right answer
]

correct = 0
scored = 0

for i, (query, expected) in enumerate(CASES):
    config = {"configurable": {"thread_id": f"m3-test-{i}"}}
    result = graph.invoke({"messages": [HumanMessage(content=query)]}, config=config)

    route = result["route"]
    conf = result["routing_confidence"]
    if expected is None:
        verdict = "n/a "
    else:
        scored += 1
        ok = route == expected
        correct += ok
        verdict = "OK  " if ok else "MISS"

    first_line = result["final_response"].strip().splitlines()[0][:100]
    print(f"[{verdict}] route={route:<9} conf={conf:.2f} expected={expected}")
    print(f"        Q: {query}")
    print(f"        A: {first_line}\n")

print(f"Routing accuracy: {correct}/{scored}")
