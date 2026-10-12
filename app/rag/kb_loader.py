"""Parse the knowledge-base markdown files into chunks: one chunk per '## KB-...' entry.

Pure text work: no network, no LLM, so it is fast and easy to test.
"""

import re
from pathlib import Path

# File header between two '---' lines, e.g. "category: technical".
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
# Entry heading, e.g. "## KB-ERR-01: Error 403 Forbidden ..."
_ENTRY = re.compile(r"^## (KB-[A-Z]+-\d+): (.+)$", re.MULTILINE)


def _split_front_matter(raw: str) -> tuple[dict, str]:
    """Return ({key: value}, rest_of_file)."""
    match = _FRONT_MATTER.match(raw)
    if not match:
        raise ValueError("KB file is missing its '---' header (category/source)")
    meta = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, raw[match.end():]


def load_kb_chunks(kb_dir) -> list[dict]:
    """One dict per entry: {doc_id, title, body, text, category, source, file}.

    `text` (title + body) is what we embed AND what the LLM later sees: the title is
    question-shaped and carries the key terms users type (e.g. 'Error 403').
    """
    chunks = []
    for path in sorted(Path(kb_dir).glob("*.md")):  # sorted => stable order
        meta, content = _split_front_matter(path.read_text(encoding="utf-8"))
        headings = list(_ENTRY.finditer(content))
        for i, heading in enumerate(headings):
            # An entry runs from its heading to the next heading (or end of file).
            end = headings[i + 1].start() if i + 1 < len(headings) else len(content)
            doc_id, title = heading.group(1), heading.group(2).strip()
            body = content[heading.end():end].strip()
            chunks.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "body": body,
                    "text": f"{title}\n{body}",
                    "category": meta["category"],
                    "source": meta["source"],
                    "file": path.name,
                }
            )
    return chunks
