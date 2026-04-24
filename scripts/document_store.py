"""
document_store.py — Minimal document storage demo for learning.

What this teaches:
  - Turn markdown into a list of speakable sentences
  - Persist documents, highlights, notes, and reading position
  - Reload state later without involving an LLM

Usage:
  uv run --project backend python scripts/document_store.py
  uv run --project backend python scripts/document_store.py --base-dir /tmp/doc-store-demo
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger


def setup_logging() -> None:
    logger.remove()
    logger.add(
        sink=sys.stdout,
        level="INFO",
        colorize=True,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <7}</level> | "
            "<cyan>{extra[step]}</cyan> | "
            "<level>{message}</level>\n"
        ),
    )


def log(step: str, message: str, *args: object) -> None:
    logger.bind(step=step).info(message, *args)


def iso_now() -> str:
    return datetime.now(UTC).isoformat()


def strip_markdown(text: str) -> str:
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    return text


def parse_markdown(raw_markdown: str) -> tuple[str, list[str]]:
    """
    Convert markdown into a title plus speakable sentences.

    The production module does more edge-case handling. This version keeps only
    the core idea so the pipeline is easy to follow.
    """
    title_match = re.search(r"^#{1,2}\s+(.+)$", raw_markdown, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Untitled"

    plain_text = strip_markdown(raw_markdown)
    lines = [line.strip() for line in plain_text.splitlines() if line.strip()]

    sentences: list[str] = []
    for line in lines:
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            cleaned = re.sub(r"\s+", " ", sentence).strip()
            if len(cleaned) > 10:
                sentences.append(cleaned)

    return title, sentences


@dataclass
class ParsedDocument:
    doc_id: str
    filename: str
    title: str
    raw_markdown: str
    sentences: list[str]
    uploaded_at: str


class DocumentStoreDemo:
    """
    A minimal JSON-backed document store.

    This mirrors the real service shape:
      - documents live on disk
      - metadata lives in an index
      - annotations are stored separately
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.docs_dir = base_dir / "documents"
        self.annotations_dir = base_dir / "annotations"
        self.index_path = self.docs_dir / "index.json"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.index = self._load_index()
        log("init", "base_dir={} docs={} annotations={}", self.base_dir, self.docs_dir, self.annotations_dir)

    def _load_index(self) -> list[dict]:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        return []

    def _save_index(self) -> None:
        self.index_path.write_text(json.dumps(self.index, indent=2), encoding="utf-8")

    def save_document(self, filename: str, raw_markdown: str) -> ParsedDocument:
        doc_id = uuid.uuid4().hex[:12]
        title, sentences = parse_markdown(raw_markdown)
        doc_path = self.docs_dir / f"{doc_id}.md"
        doc_path.write_text(raw_markdown, encoding="utf-8")

        metadata = {
            "doc_id": doc_id,
            "filename": filename,
            "title": title,
            "uploaded_at": iso_now(),
        }
        self.index.append(metadata)
        self._save_index()

        log("save", "saved doc_id={} title={!r} sentence_count={}", doc_id, title, len(sentences))
        return ParsedDocument(
            doc_id=doc_id,
            filename=filename,
            title=title,
            raw_markdown=raw_markdown,
            sentences=sentences,
            uploaded_at=metadata["uploaded_at"],
        )

    def get_document(self, doc_id: str) -> ParsedDocument | None:
        metadata = next((item for item in self.index if item["doc_id"] == doc_id), None)
        if metadata is None:
            return None

        doc_path = self.docs_dir / f"{doc_id}.md"
        if not doc_path.exists():
            return None

        raw_markdown = doc_path.read_text(encoding="utf-8")
        title, sentences = parse_markdown(raw_markdown)
        log("load", "loaded doc_id={} title={!r} sentence_count={}", doc_id, title, len(sentences))
        return ParsedDocument(
            doc_id=doc_id,
            filename=str(metadata["filename"]),
            title=title,
            raw_markdown=raw_markdown,
            sentences=sentences,
            uploaded_at=str(metadata["uploaded_at"]),
        )

    def load_annotations(self, doc_id: str) -> dict:
        path = self.annotations_dir / f"{doc_id}_annotations.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {"doc_id": doc_id, "highlights": [], "notes": [], "reading_position": None}

    def _save_annotations(self, doc_id: str, payload: dict) -> None:
        path = self.annotations_dir / f"{doc_id}_annotations.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def save_highlight(self, doc_id: str, sentence_idx: int, color: str = "yellow") -> None:
        data = self.load_annotations(doc_id)
        data["highlights"].append(
            {
                "id": f"hl_{uuid.uuid4().hex[:8]}",
                "sentence_idx": sentence_idx,
                "color": color,
                "created_at": iso_now(),
            }
        )
        self._save_annotations(doc_id, data)
        log("highlight", "saved sentence_idx={} color={} for doc_id={}", sentence_idx, color, doc_id)

    def save_note(self, doc_id: str, sentence_idx: int, note_text: str) -> None:
        data = self.load_annotations(doc_id)
        data["notes"].append(
            {
                "id": f"note_{uuid.uuid4().hex[:8]}",
                "sentence_idx": sentence_idx,
                "text": note_text,
                "created_at": iso_now(),
            }
        )
        self._save_annotations(doc_id, data)
        log("note", "saved note for sentence_idx={} text={!r}", sentence_idx, note_text)

    def save_reading_position(self, doc_id: str, sentence_idx: int, word_idx: int) -> None:
        data = self.load_annotations(doc_id)
        data["reading_position"] = {
            "sentence_idx": sentence_idx,
            "word_idx": word_idx,
            "saved_at": iso_now(),
        }
        self._save_annotations(doc_id, data)
        log("resume", "saved reading_position sentence_idx={} word_idx={}", sentence_idx, word_idx)


SAMPLE_MARKDOWN = """# Attention Is All You Need

Transformers replaced recurrence with attention. This made training easier to parallelize.

- Self-attention lets each token look at other tokens.
- Positional information is added because order still matters.

The architecture is simple, but the training dynamics matter a lot.
"""


def run_demo(base_dir: Path) -> None:
    store = DocumentStoreDemo(base_dir)

    log("demo", "saving sample markdown document")
    document = store.save_document("attention.md", SAMPLE_MARKDOWN)

    for index, sentence in enumerate(document.sentences):
        log("sentence", "[{}] {}", index, sentence)

    store.save_highlight(document.doc_id, sentence_idx=1, color="yellow")
    store.save_note(
        document.doc_id,
        sentence_idx=2,
        note_text="This is a good place to explain why recurrence was slower.",
    )
    store.save_reading_position(document.doc_id, sentence_idx=2, word_idx=4)

    log("demo", "reloading the same document and annotations from disk")
    reloaded_document = store.get_document(document.doc_id)
    annotations = store.load_annotations(document.doc_id)

    if reloaded_document is None:
        raise RuntimeError("Expected saved document to exist")

    log("result", "title={!r}", reloaded_document.title)
    log("result", "sentences={}", len(reloaded_document.sentences))
    log("result", "annotations={}", json.dumps(annotations, indent=2))
    log("result", "parsed_document={}", json.dumps(asdict(reloaded_document), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal document store learning script")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/tmp/neurotalk_document_store_demo"),
        help="Directory used for demo storage",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    run_demo(args.base_dir)
