"""Regression check (run: python -m scripts.check_graph): full graph on labeled queries.

Reports routing accuracy and escalation accuracy separately, then sweeps the
escalation threshold OFFLINE from the recorded confidences (no extra LLM calls).

Each case gets its OWN thread_id so checkpointed state cannot leak between cases.
"""

import sys

from langchain_core.messages import HumanMessage

from app.graph import graph
from app.nodes import CONFIDENCE_THRESHOLD, should_escalate

sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252

# (query, expected_route, expected_escalated). None = no single right answer, not scored.
CASES = [
    ("Payment deducted but course not activated. TXN002", "billing", False),
    ("I want a refund for last month", "billing", False),
    ("I can't log in, account ACC002", "account", False),
    ("How do I change my email?", "account", False),
    ("Video won't load", "technical", False),
    ("The app crashes on startup", "technical", False),
    ("What courses do you have?", "general", False),
    ("Do you offer certificates?", "general", False),
    ("Something is broken, please help", "general", True),
    ("Help me", "general", True),
    ("It's not working", "general", True),
    ("I have a problem with my thing", "general", True),
    ("I already completed my billing what is my account balance because app is lagging", None, None),
]

route_ok = route_total = 0
esc_ok = esc_total = 0
recorded = []  # (query, confidence, expected_escalated)

for i, (query, exp_route, exp_esc) in enumerate(CASES):
    config = {"configurable": {"thread_id": f"regression-{i}"}, "recursion_limit": 10}
    result = graph.invoke({"messages": [HumanMessage(content=query)]}, config=config)

    route = result["route"]
    conf = result["routing_confidence"]
    escalated = result["escalated"]
    recorded.append((query, conf, exp_esc))

    r_flag = e_flag = "n/a "
    if exp_route is not None:
        route_total += 1
        route_ok += route == exp_route
        r_flag = "OK  " if route == exp_route else "MISS"
    if exp_esc is not None:
        esc_total += 1
        esc_ok += escalated == exp_esc
        e_flag = "OK  " if escalated == exp_esc else "MISS"

    print(f"route[{r_flag}] esc[{e_flag}] route={route:<9} conf={conf:.2f} "
          f"escalated={escalated!s:<5} agent={result['current_agent']}")
    print(f"    Q: {query}")
    if escalated:
        print(f"    reason: {result['escalation_reason']}")
    print()

print(f"Routing accuracy:    {route_ok}/{route_total}")
print(f"Escalation accuracy: {esc_ok}/{esc_total}  (threshold={CONFIDENCE_THRESHOLD})\n")

print("Threshold sweep (computed from recorded confidences):")
print(f"{'threshold':>9} | {'escalated':>9} | {'should-escalate missed':>22} | {'wrongly escalated':>17}")
for t in (0.30, 0.50, 0.70, 0.85, 0.95):
    escalated_n = missed = wrong = 0
    for _, conf, exp_esc in recorded:
        esc = should_escalate(conf, threshold=t)
        escalated_n += esc
        if exp_esc is True and not esc:
            missed += 1
        if exp_esc is False and esc:
            wrong += 1
    print(f"{t:>9.2f} | {escalated_n:>9} | {missed:>22} | {wrong:>17}")
