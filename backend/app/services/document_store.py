from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from loguru import logger


@dataclass
class ParsedDocument:
    doc_id: str
    filename: str
    title: str
    raw_markdown: str
    sentences: list[str]
    word_count: int
    sentence_count: int
    uploaded_at: str


def _iso() -> str:
    return datetime.now(UTC).isoformat()


def _strip_markdown_formatting(text: str) -> str:
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", "", text)
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^>+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"-{3,}|={3,}|\*{3,}", "", text)
    return text


def _split_long_fragment(fragment: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", fragment).strip(" -•|\t")
    if len(normalized) <= 220:
        return [normalized] if len(normalized) > 10 else []

    split_patterns = [
        r"\s+[•|]\s+",
        r"\s+[-–—]\s+",
        r":\s+(?=[A-Z])",
        r"(?<=[a-z])\s+(?=[A-Z][a-z]{3,}\b)",
    ]

    pending = [normalized]
    for pattern in split_patterns:
        next_pending: list[str] = []
        changed = False
        for item in pending:
            if len(item) <= 220:
                next_pending.append(item)
                continue
            parts = [part.strip(" -•|\t") for part in re.split(pattern, item) if len(part.strip(" -•|\t")) > 10]
            if len(parts) > 1:
                next_pending.extend(parts)
                changed = True
            else:
                next_pending.append(item)
        pending = next_pending
        if changed:
            break

    final_parts: list[str] = []
    for item in pending:
        compact = re.sub(r"\s+", " ", item).strip()
        if len(compact) <= 240:
            if len(compact) > 10:
                final_parts.append(compact)
            continue
        words = compact.split()
        chunk: list[str] = []
        for word in words:
            chunk.append(word)
            candidate = " ".join(chunk)
            if len(candidate) >= 180 and word.endswith((",", ";", ":", ".")):
                if len(candidate) > 10:
                    final_parts.append(candidate.strip())
                chunk = []
        if chunk:
            remainder = " ".join(chunk).strip()
            if len(remainder) > 10:
                final_parts.append(remainder)

    return final_parts


def _parse_markdown(raw: str) -> tuple[str, list[str]]:
    """Return (title, speakable_sentences)."""
    title_match = re.search(r"^#{1,2}\s+(.+)$", raw, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Untitled"

    plain = _strip_markdown_formatting(raw)
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in plain.splitlines() if ln.strip()]

    sentences: list[str] = []
    for line in lines:
        raw_fragments = re.split(r"(?<=[.!?])\s+|\s+[•|]\s+", line)
        for fragment in raw_fragments:
            cleaned = fragment.strip()
            if len(cleaned) <= 10:
                continue
            sentences.extend(_split_long_fragment(cleaned))

    return title, sentences


class DocumentStore:
    def __init__(self, docs_dir: Path, annotations_dir: Path) -> None:
        self._docs_dir = docs_dir
        self._annotations_dir = annotations_dir
        self._docs_dir.mkdir(parents=True, exist_ok=True)
        self._annotations_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = docs_dir / "index.json"
        self._index: list[dict] = self._load_index()

    def _load_index(self) -> list[dict]:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _save_index(self) -> None:
        self._index_path.write_text(
            json.dumps(self._index, indent=2), encoding="utf-8"
        )

    def save_document(self, filename: str, content: bytes) -> ParsedDocument:
        doc_id = uuid.uuid4().hex[:12]
        safe_name = re.sub(r"[^\w.-]", "_", filename)
        dest = self._docs_dir / f"{doc_id}.md"
        dest.write_bytes(content)

        raw = content.decode("utf-8", errors="replace")
        title, sentences = _parse_markdown(raw)
        word_count = sum(len(s.split()) for s in sentences)
        now = _iso()

        meta = {
            "doc_id": doc_id,
            "filename": safe_name,
            "title": title,
            "word_count": word_count,
            "sentence_count": len(sentences),
            "uploaded_at": now,
        }
        self._index.append(meta)
        self._save_index()
        logger.info("event=doc_saved doc_id={} filename={} sentences={}", doc_id, safe_name, len(sentences))

        return ParsedDocument(
            doc_id=doc_id,
            filename=safe_name,
            title=title,
            raw_markdown=raw,
            sentences=sentences,
            word_count=word_count,
            sentence_count=len(sentences),
            uploaded_at=now,
        )

    def list_documents(self) -> list[dict]:
        return [
            {k: v for k, v in m.items() if k != "sentences"}
            for m in self._index
        ]

    def get_document(self, doc_id: str) -> ParsedDocument | None:
        meta = next((m for m in self._index if m["doc_id"] == doc_id), None)
        if not meta:
            return None
        path = self._docs_dir / f"{doc_id}.md"
        if not path.exists():
            return None
        raw = path.read_text(encoding="utf-8", errors="replace")
        _, sentences = _parse_markdown(raw)
        return ParsedDocument(
            doc_id=doc_id,
            filename=meta["filename"],
            title=meta["title"],
            raw_markdown=raw,
            sentences=sentences,
            word_count=meta.get("word_count", 0),
            sentence_count=len(sentences),
            uploaded_at=meta.get("uploaded_at", ""),
        )

    def delete_document(self, doc_id: str) -> bool:
        self._index = [m for m in self._index if m["doc_id"] != doc_id]
        self._save_index()
        removed = False
        for path in [self._docs_dir / f"{doc_id}.md", self._annotations_dir / f"{doc_id}_annotations.json"]:
            if path.exists():
                path.unlink()
                removed = True
        return removed

    def load_annotations(self, doc_id: str) -> dict:
        path = self._annotations_dir / f"{doc_id}_annotations.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"doc_id": doc_id, "highlights": [], "snippets": [], "reading_position": None}

    def _save_annotations(self, doc_id: str, data: dict) -> None:
        path = self._annotations_dir / f"{doc_id}_annotations.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def save_highlight(self, doc_id: str, sentence_idx: int, color: str = "yellow") -> None:
        data = self.load_annotations(doc_id)
        existing = {h["sentence_idx"] for h in data["highlights"]}
        if sentence_idx not in existing:
            data["highlights"].append({
                "id": f"hl_{uuid.uuid4().hex[:8]}",
                "sentence_idx": sentence_idx,
                "color": color,
                "created_at": _iso(),
                "source": "user",
            })
            self._save_annotations(doc_id, data)

    def save_snippet(self, doc_id: str, term: str, explanation: str,
                     sentence_idx: int = -1, word_idx: int = -1,
                     search_results: list[dict] | None = None) -> dict:
        data = self.load_annotations(doc_id)
        snippet = {
            "id": f"sn_{uuid.uuid4().hex[:8]}",
            "term": term,
            "explanation": explanation,
            "sentence_idx": sentence_idx,
            "word_idx": word_idx,
            "search_results": search_results or [],
            "created_at": _iso(),
        }
        data["snippets"].append(snippet)
        self._save_annotations(doc_id, data)
        return snippet

    def save_reading_position(self, doc_id: str, sentence_idx: int) -> None:
        data = self.load_annotations(doc_id)
        data["reading_position"] = {"last_sentence_idx": sentence_idx, "saved_at": _iso()}
        self._save_annotations(doc_id, data)


@lru_cache(maxsize=1)
def get_document_store() -> DocumentStore:
    from config.settings import get_settings
    s = get_settings()
    return DocumentStore(docs_dir=s.docs_dir, annotations_dir=s.annotations_dir)
