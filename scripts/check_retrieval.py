"""Eyeball retrieval quality + collect scores for the threshold.
Run: python -m scripts.check_retrieval
"""

import sys

from app.rag.retriever import get_hybrid_retriever, get_vectorstore

sys.stdout.reconfigure(encoding="utf-8")

# (category, query, acceptable doc_ids). Empty set => out-of-KB, nothing should match.
CASES = [
    ("technical", "The video won't play on my course", {"KB-VID-01", "KB-PERF-01"}),
    ("technical", "App keeps crashing when I open it", {"KB-APP-01", "KB-ERR-05"}),
    ("technical", "I get error 403 on a lesson", {"KB-ERR-01"}),
    ("technical", "Code sandbox is not running my code", {"KB-SBX-01"}),
    ("technical", "How do I download course files?", {"KB-DL-01"}),
    ("technical", "How do I reset my password?", set()),
    ("technical", "Which courses do you offer on cooking?", set()),
    ("billing", "My payment failed, card was declined", {"KB-PAY-01"}),
    ("billing", "Money got deducted but course is locked", {"KB-PAY-02"}),
    ("billing", "I was charged twice for one course", {"KB-PAY-03"}),
    ("billing", "Can I get a refund after 10 days?", {"KB-REF-01"}),
    ("billing", "How do I cancel my subscription?", {"KB-SUB-01"}),
    ("billing", "Where can I download my invoice?", {"KB-INV-01"}),
    ("billing", "Do you accept PayPal?", {"KB-PAY-04"}),
    ("billing", "Why is my video buffering?", set()),
    ("billing", "Who is the instructor of the Python course?", set()),
]

store = get_vectorstore()

for category, query, expected in CASES:
    top3 = [d.metadata["doc_id"] for d in get_hybrid_retriever(category).invoke(query)[:3]]
    score = store.similarity_search_with_score(query, k=1)[0][1]  # cosine over whole KB
    kind = "IN " if expected else "OUT"
    hit = "hit " if expected & set(top3) else ("    " if not expected else "MISS")
    print(f"{category:9} {kind} {hit} dense_top1={score:.3f}  {top3}  <- {query}")
