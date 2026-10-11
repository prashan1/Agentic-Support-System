"""Checkpointer helpers for short-term memory (conversation state)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver

_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / ".checkpoints"


def sqlite_checkpointer(db_name: str = "lab.db") -> SqliteSaver:
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _CHECKPOINT_DIR / db_name
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)