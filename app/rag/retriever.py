"""Hybrid retrieval built from LangChain components (no hand-written search code).

dense  : OllamaEmbeddings + QdrantVectorStore (local on-disk mode)
sparse : BM25Retriever (rank_bm25 under the hood)
fusion : EnsembleRetriever = weighted Reciprocal Rank Fusion
"""

import os
import uuid
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models

from app.rag.kb_loader import load_kb_chunks

load_dotenv()

KB_DIR = Path(__file__).parent / "kb"
QDRANT_PATH = os.getenv("QDRANT_LOCAL_PATH", ".qdrant")
COLLECTION = os.getenv("KB_COLLECTION", "support-kb")


@lru_cache(maxsize=None)
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_EMBED_BASE_URL", "http://localhost:11434"),
    )


def load_documents() -> list[Document]:
    """KB entries as LangChain Documents (text to embed + metadata for citations)."""
    return [
        Document(
            page_content=c["text"],
            metadata={k: c[k] for k in ("doc_id", "title", "category", "source", "file")},
        )
        for c in load_kb_chunks(KB_DIR)
    ]


def ingest(docs: list[Document] | None = None) -> QdrantVectorStore:
    """(Re)build the Qdrant collection. Stable uuid5 ids from doc_id => no duplicates."""
    docs = docs or load_documents()
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, d.metadata["doc_id"])) for d in docs]
    return QdrantVectorStore.from_documents(
        docs,
        embedding=get_embeddings(),
        ids=ids,
        path=QDRANT_PATH,
        collection_name=COLLECTION,
        force_recreate=True,  # wipe + rebuild: simplest idempotent ingest for a small KB
    )


@lru_cache(maxsize=None)
def get_vectorstore() -> QdrantVectorStore:
    """Open the already-ingested collection (local mode allows one client per process)."""
    return QdrantVectorStore.from_existing_collection(
        embedding=get_embeddings(),
        path=QDRANT_PATH,
        collection_name=COLLECTION,
    )


@lru_cache(maxsize=None)  # build once per category (BM25 index + filter), not per request
def get_hybrid_retriever(category: str, k: int = 3, weights=(0.5, 0.5)):
    """Hybrid retriever limited to one KB category ('technical' or 'billing')."""
    only_category = models.Filter(
        must=[models.FieldCondition(key="metadata.category", match=models.MatchValue(value=category))]
    )
    dense = get_vectorstore().as_retriever(search_kwargs={"k": k * 2, "filter": only_category})
    docs = [d for d in load_documents() if d.metadata["category"] == category]
    sparse = BM25Retriever.from_documents(docs, k=k * 2)
    # EnsembleRetriever fuses by rank (RRF, c=60) but returns the WHOLE fused union
    # (up to 2*k*2 docs), so we must cut to the top k ourselves.
    ensemble = EnsembleRetriever(retrievers=[sparse, dense], weights=list(weights))
    top_k = RunnableLambda(lambda docs: docs[:k])
    return (ensemble | top_k).with_config(run_name=f"hybrid_retriever_{category}")
