"""Print the supervisor's route + confidence for a set of queries (no specialist runs).

Run: python -m scripts.check_routing
A query escalates when confidence < 0.70, so this shows WHY a message never reaches an agent.
"""

import sys

from langchain_core.messages import HumanMessage

from app.nodes import CONFIDENCE_THRESHOLD, supervisor_node

sys.stdout.reconfigure(encoding="utf-8")

QUERIES = [
    # Recall questions about the customer themselves (the bug)
    "what do you have in memory for me",
    "tell me about me",
    "what do you know about me",
    "what is my name",
    "Remind me, which plan am I on and which device do I use?",
    # Controls: clear requests that should still route normally
    "Check transaction TXN999",
    "How do I cancel my subscription plan?",
    "My lesson video keeps buffering",
    # Genuinely vague: should still escalate
    "Help me",
    "It's not working",
]

for query in QUERIES:
    decision = supervisor_node({"messages": [HumanMessage(content=query)]})
    conf = decision["routing_confidence"]
    verdict = "ESCALATE" if conf < CONFIDENCE_THRESHOLD else "routes"
    print(f"{decision['route']:<10} conf={conf:.2f} {verdict:<9} <- {query}")
