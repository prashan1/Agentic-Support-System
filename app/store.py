"""Long-term memory store: per-user facts, searchable by meaning.

Checkpointer = state of ONE thread. Store = facts shared across ALL of a user's threads.
Namespace ("memories", user_id) keeps users apart and makes "forget me" one delete.
"""

from __future__ import annotations

import re
import sqlite3
import uuid
from pathlib import Path

from langgraph.store.sqlite import SqliteStore

from app.rag.retriever import get_embeddings

_MEMORY_DIR = Path(__file__).resolve().parent.parent / ".memory"
EMBED_DIMS = 768  # nomic-embed-text


def memory_namespace(user_id: str) -> tuple[str, str]:
    return ("memories", user_id)


def sqlite_store(db_name: str = "lab.db") -> SqliteStore:
    _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    # isolation_level=None = autocommit, so writes are visible immediately.
    conn = sqlite3.connect(str(_MEMORY_DIR / db_name), check_same_thread=False, isolation_level=None)
    store = SqliteStore(
        conn,
        index={"dims": EMBED_DIMS, "embed": get_embeddings(), "fields": ["fact"]},
    )
    store.setup()  # creates tables (and the vector table) if missing
    return store


_CARD_LIKE = re.compile(r"(?:\d[ -]?){13,19}")  # long digit runs look like card numbers


def save_fact(store, user_id: str, fact: str) -> bool:
    """WRITE side of memory. Returns False if the fact was rejected.

    The user_id comes from trusted config (set by the caller), never from model text.
    """
    if _CARD_LIKE.search(fact):
        return False  # never store anything that looks like a card number
    store.put(memory_namespace(user_id), uuid.uuid4().hex, {"fact": fact.strip()})
    return True


def forget_user(store, user_id: str) -> int:
    """Delete every memory of one user. Returns how many were removed."""
    items = store.search(memory_namespace(user_id), limit=1000)
    for item in items:
        store.delete(item.namespace, item.key)
    return len(items)
