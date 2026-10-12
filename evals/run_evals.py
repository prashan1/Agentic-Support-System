"""Run the code-based eval over the whole dataset.

    python -m evals.run_evals

Every row runs on its OWN thread (unique per run) so the persistent checkpointer cannot leak
state between rows or between runs. A crash on one row is recorded, not fatal.
"""

import json
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage

from app.graph import graph
from evals.datasets import get_cases
from evals.evaluators import (
    first_tool_called,
    precision_recall,
    rate,
    retrieved_doc_ids,
    score_escalation,
    score_grounding,
    score_retrieval,
    score_route,
    score_tool,
)

sys.stdout.reconfigure(encoding="utf-8")

RESULTS_DIR = Path(__file__).parent / "results"


def run_case(case: dict, run_tag: str) -> dict:
    config = {
        "configurable": {"thread_id": f"eval-{run_tag}-{case['id']}"},
        "recursion_limit": 10,
        "run_name": "eval-case",
        "tags": ["eval", run_tag],  # filter these traces in LangSmith
        "metadata": {"case_id": case["id"], "difficulty": case["difficulty"]},
    }
    row = {
        "id": case["id"],
        "category": case["id"].rsplit("-", 1)[0],
        "difficulty": case["difficulty"],
        "known_gap": bool(case.get("known_gap")),
        "query": case["query"],
        "expected_route": case["expected_route"],
        "expected_tool": case["expected_tool"],
        "expected_escalation": case["expected_escalation"],
    }
    start = time.perf_counter()
    try:
        result = graph.invoke({"messages": [HumanMessage(content=case["query"])]}, config=config)
        row.update(
            route=result["route"],
            confidence=result["routing_confidence"],
            tool=first_tool_called(result["messages"]),
            escalated=result["escalated"],
            agent=result["current_agent"],
            response=result["final_response"],
            error=None,
        )
        row["route_ok"] = score_route(row["expected_route"], row["route"])
        row["esc_ok"] = score_escalation(row["expected_escalation"], row["escalated"])
        row["tool_ok"] = score_tool(row["expected_tool"], row["tool"])
        # Retrieved docs persist per thread; eval threads are unique, so only this row's
        # retrieval is here. Non-RAG routes never retrieve => empty list.
        row["retrieved"] = retrieved_doc_ids(result.get("retrieved_documents"))
        row["retrieval_ok"] = score_retrieval(case.get("expected_docs"), row["retrieved"])
        row["grounded_ok"] = score_grounding(case.get("kb_answerable"), row["response"], row["retrieved"])
    except Exception as exc:  # a crash is a RESULT, not a reason to abort the whole eval
        row.update(route=None, confidence=None, tool=None, escalated=None, agent=None,
                   response=None, error=f"{type(exc).__name__}: {str(exc)[:160]}")
        # A crashed row fails every metric that applies to it (the customer got nothing).
        row["route_ok"] = None if row["expected_route"] is None else False
        row["esc_ok"] = None if row["expected_escalation"] is None else False
        row["tool_ok"] = False
        row["retrieved"] = []
        row["retrieval_ok"] = None if not case.get("expected_docs") else False
        row["grounded_ok"] = None if case.get("kb_answerable") is None else False
    row["seconds"] = round(time.perf_counter() - start, 2)
    return row


def fmt(passed_total: tuple[int, int]) -> str:
    ok, total = passed_total
    return f"{ok}/{total}" + (f" ({ok / total:.0%})" if total else "")


def summarize(rows: list[dict], title: str) -> None:
    print(f"\n--- {title} ({len(rows)} rows) ---")
    if not rows:
        print("  (none)")
        return
    print(f"  routing    : {fmt(rate(r['route_ok'] for r in rows))}")
    print(f"  tool       : {fmt(rate(r['tool_ok'] for r in rows))}")
    print(f"  escalation : {fmt(rate(r['esc_ok'] for r in rows))}")
    print(f"  retrieval  : {fmt(rate(r['retrieval_ok'] for r in rows))}  (recall@3)")
    print(f"  grounding  : {fmt(rate(r['grounded_ok'] for r in rows))}  (cites KB / refuses)")
    print(f"  errors     : {sum(1 for r in rows if r['error'])}")
    ok_rows = [r for r in rows if not r["error"] and r["expected_escalation"] is not None]
    precision, recall = precision_recall([(r["expected_escalation"], r["escalated"]) for r in ok_rows])
    p = "n/a" if precision is None else f"{precision:.0%}"
    rc = "n/a" if recall is None else f"{recall:.0%}"
    print(f"  escalation precision: {p} | recall: {rc}")


def breakdown(rows: list[dict], key: str) -> None:
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    print(f"\n  by {key}:")
    print(f"    {'':<12} {'route':>8} {'tool':>8} {'escal.':>8}")
    for name, group in sorted(groups.items()):
        r_ok, r_n = rate(g["route_ok"] for g in group)
        t_ok, t_n = rate(g["tool_ok"] for g in group)
        e_ok, e_n = rate(g["esc_ok"] for g in group)
        print(f"    {name:<12} {r_ok}/{r_n:>6} {t_ok}/{t_n:>6} {e_ok}/{e_n:>6}")


def main() -> None:
    run_tag = uuid.uuid4().hex[:6]
    cases = get_cases(include_known_gaps=True)
    if len(sys.argv) > 1:  # e.g. `python -m evals.run_evals rag` runs only ids starting "rag"
        cases = [c for c in cases if c["id"].startswith(sys.argv[1])]
    print(f"Eval run {run_tag}: {len(cases)} cases (LangSmith tag: {run_tag})\n")

    rows = []
    for case in cases:
        row = run_case(case, run_tag)
        rows.append(row)
        flags = [row["route_ok"], row["tool_ok"], row["esc_ok"], row["retrieval_ok"], row["grounded_ok"]]
        mark = "FAIL" if any(f is False for f in flags) else " ok "
        gap = " [gap]" if row["known_gap"] else ""
        print(f"[{mark}] {row['id']:<15}{gap}")

    regression = [r for r in rows if not r["known_gap"]]
    gaps = [r for r in rows if r["known_gap"]]

    print("\n" + "=" * 70)
    summarize(regression, "REGRESSION rows (must stay green)")
    breakdown(regression, "category")
    breakdown(regression, "difficulty")
    summarize(gaps, "KNOWN-GAP rows (backlog: failures expected)")

    failures = [
        r for r in rows
        if any(r[k] is False for k in ("route_ok", "tool_ok", "esc_ok", "retrieval_ok", "grounded_ok"))
    ]
    print("\n" + "=" * 70)
    print(f"FAILURES ({len(failures)}):")
    for r in failures:
        gap = " [known gap]" if r["known_gap"] else ""
        print(f"\n  {r['id']}{gap}   Q: {r['query'][:90]!r}")
        if r["error"]:
            print(f"    ERROR: {r['error']}")
            continue
        print(f"    route     expected={r['expected_route']!s:<10} actual={r['route']} (conf {r['confidence']:.2f})")
        print(f"    tool      expected={r['expected_tool']!s:<22} actual={r['tool']}")
        print(f"    escalated expected={r['expected_escalation']!s:<10} actual={r['escalated']}")
        if r["retrieval_ok"] is not None or r["grounded_ok"] is not None:
            print(f"    retrieved={r['retrieved']}  retrieval_ok={r['retrieval_ok']}  grounded_ok={r['grounded_ok']}")
            print(f"    reply: {r['response'][:200]!r}")

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{run_tag}.json"
    path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved per-row results to {path}")


if __name__ == "__main__":
    main()
