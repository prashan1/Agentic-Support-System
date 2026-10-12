"""Embed the KB into local Qdrant. Run: python -m scripts.ingest_kb"""

import sys

from app.rag.retriever import COLLECTION, ingest

sys.stdout.reconfigure(encoding="utf-8")

store = ingest()
info = store.client.get_collection(COLLECTION)
print("points :", info.points_count)  # expect 23
vectors = info.config.params.vectors  # LangChain uses a named vector => dict
print("dim    :", next(iter(vectors.values())).size)  # expect 768
print("sample :", store.similarity_search("error 403 when opening a lesson", k=1)[0].metadata)
