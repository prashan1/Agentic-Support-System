"""Spec for Step A of M9: the KB loader.

Write app/rag/kb_loader.py so these pass:   python -m pytest tests/test_kb_parser.py -q
No Ollama, no network, no LLM: parsing is pure text work, so it is fast and deterministic.
"""

import re
from pathlib import Path

import pytest

from app.rag.kb_loader import load_kb_chunks

KB_DIR = Path(__file__).resolve().parent.parent / "app" / "rag" / "kb"
REQUIRED_KEYS = {"doc_id", "title", "body", "text", "category", "source", "file"}


@pytest.fixture(scope="module")
def chunks():
    return load_kb_chunks(KB_DIR)


def test_one_chunk_per_entry(chunks):
    by_category = {}
    for c in chunks:
        by_category[c["category"]] = by_category.get(c["category"], 0) + 1
    assert by_category == {"technical": 23, "billing": 13}


def test_every_chunk_has_required_fields_and_no_empty_values(chunks):
    for chunk in chunks:
        assert REQUIRED_KEYS <= chunk.keys(), chunk.get("doc_id")
        for key in REQUIRED_KEYS:
            assert str(chunk[key]).strip(), f"{chunk['doc_id']} has empty {key}"


def test_doc_ids_are_unique_and_well_formed(chunks):
    ids = [c["doc_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "duplicate doc_id"
    assert all(re.fullmatch(r"KB-[A-Z]+-\d{2}", i) for i in ids)


def test_front_matter_is_applied_to_every_entry_in_the_file(chunks):
    video = [c for c in chunks if c["doc_id"].startswith("KB-VID-")]
    assert len(video) == 4
    assert {c["category"] for c in video} == {"technical"}
    assert {c["source"] for c in video} == {"Video playback troubleshooting"}


def test_text_is_title_plus_body(chunks):
    err = next(c for c in chunks if c["doc_id"] == "KB-ERR-01")
    assert err["title"] == "Error 403 Forbidden on a lesson page or video"
    assert err["text"].startswith(err["title"])
    assert err["body"] in err["text"]
    assert "403" in err["text"]


def test_entries_do_not_leak_into_each_other(chunks):
    # The body of one entry must not contain the next entry's heading.
    for chunk in chunks:
        assert "## KB-" not in chunk["text"], chunk["doc_id"]
    vid1 = next(c for c in chunks if c["doc_id"] == "KB-VID-01")
    assert "no sound" not in vid1["body"].lower()  # that is KB-VID-02's topic


def test_order_is_stable(chunks):
    assert [c["doc_id"] for c in load_kb_chunks(KB_DIR)] == [c["doc_id"] for c in chunks]
