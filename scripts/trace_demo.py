"""M7: run 4 turns with run_name / tags / metadata, then read the traces back from LangSmith.

Run:  python -m scripts.trace_demo
"""

import os
import sys
import time
import uuid

from langchain_core.messages import HumanMessage
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client

from app.graph import graph  # importing app also loads .env

sys.stdout.reconfigure(encoding="utf-8")

PROJECT = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT")
RUN_TAG = f"m7-{uuid.uuid4().hex[:6]}"  # unique per execution, so filters only match this run

QUERIES = [
    "Payment deducted but course not activated. TXN002",
    "Help me",
    "I can't log in, account ACC002",
    "What courses do you have?",
]

client = Client()
roots: list[tuple[str, uuid.UUID, dict]] = []

# ---- 1. Run the four turns, each on its own thread, with tracing config ----
for i, query in enumerate(QUERIES):
    run_id = uuid.uuid4()  # lets us look this exact trace up afterwards
    config = {
        "run_id": run_id,
        "run_name": "support-turn",
        "tags": ["m7", RUN_TAG],
        "metadata": {"customer_tier": "free", "app_version": "0.7", "case": i},
        "configurable": {"thread_id": f"{RUN_TAG}-{i}"},
        "recursion_limit": 10,
    }
    result = graph.invoke({"messages": [HumanMessage(content=query)]}, config=config)
    roots.append((query, run_id, result))
    print(f"[{result['current_agent']:<10}] {query}\n    -> {result['final_response'][:110]}")

wait_for_all_tracers()  # flush background trace uploads
print(f"\nProject: {PROJECT}   tag for this execution: {RUN_TAG}")


# ---- 2. Read each trace back and summarize ----
def fetch_trace(run_id, timeout=60):
    """Traces upload asynchronously, so poll until the root run is complete."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        runs = list(client.list_runs(project_name=PROJECT, trace_id=str(run_id)))
        root = next((r for r in runs if r.id == run_id), None)
        if root is not None and root.end_time is not None:
            time.sleep(3)  # let child runs finish landing
            return list(client.list_runs(project_name=PROJECT, trace_id=str(run_id)))
        time.sleep(3)
    return []


def secs(run):
    return (run.end_time - run.start_time).total_seconds() if run.end_time else float("nan")


print("\n" + "=" * 78)
for query, run_id, result in roots:
    runs = sorted(fetch_trace(run_id), key=lambda r: r.start_time)
    print(f"\nQ: {query}")
    if not runs:
        print("  (trace not found yet; check the LangSmith UI)")
        continue
    root = next(r for r in runs if r.id == run_id)
    llm_runs = [r for r in runs if r.run_type == "llm"]
    tool_runs = [r for r in runs if r.run_type == "tool"]
    print(f"  total: {secs(root):.2f}s, {root.total_tokens} tokens | "
          f"{len(llm_runs)} LLM call(s), {len(tool_runs)} tool call(s)")
    # node-level chain runs (supervisor / billing / payment_tool ...)
    nodes = [r for r in runs if r.run_type == "chain" and r.name in
             {"supervisor", "general", "technical", "billing", "account",
              "escalation", "payment_tool", "account_tool"}]
    print("  nodes in order: " + " -> ".join(f"{n.name}({secs(n):.2f}s)" for n in nodes))
    for r in llm_runs:
        print(f"    LLM  {r.name:<12} {secs(r):.2f}s  in={r.prompt_tokens} out={r.completion_tokens}")
    for r in tool_runs:
        print(f"    TOOL {r.name:<22} inputs={r.inputs}")

# ---- 3. Filtering: does tag / metadata search find our runs? ----
print("\n" + "=" * 78)
print("Filter checks (root runs only):")
by_tag = list(client.list_runs(project_name=PROJECT, is_root=True,
                               filter=f'has(tags, "{RUN_TAG}")'))
print(f"  tag {RUN_TAG!r}: {len(by_tag)} runs (expect {len(QUERIES)})")
try:
    by_meta = list(client.list_runs(
        project_name=PROJECT, is_root=True,
        filter='and(eq(metadata_key, "case"), eq(metadata_value, "1"), '
               f'has(tags, "{RUN_TAG}"))'))
    print(f"  metadata case=1 within this execution: {len(by_meta)} run(s) "
          f"-> {[r.inputs for r in by_meta][:1]}")
except Exception as exc:  # filter syntax is the fragile part, so report instead of crashing
    print(f"  metadata filter failed: {exc!r}")
