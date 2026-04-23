from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from app.prompts.system import DOCUMENT_TURN_PROMPT
from app.services.document_store import ParsedDocument, get_document_store

_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
_NON_WORD_PATTERN = re.compile(r"[^a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "continue", "did",
    "do", "document", "explain", "for", "from", "go", "i", "in", "is", "it",
    "keep", "me", "of", "on", "please", "read", "reading", "resume", "start",
    "stop", "tell", "that", "the", "this", "to", "what", "where", "which",
    "why", "with", "you",
}
_READ_KEYWORDS = ("read", "start reading", "read aloud", "begin reading")
_BEGINNING_KEYWORDS = ("beginning", "from start", "start over", "restart", "from the top")
_CONTINUE_KEYWORDS = ("continue reading", "keep reading", "resume reading", "where you left", "where we left")

DocumentAction = Literal[
    "answer",
    "read_document",
    "continue_reading",
    "pause_reading",
    "ask_document_clarification",
    "list_documents",
    "save_note",
    "highlight_sentence",
]


@dataclass
class DocumentTurnDecision:
    action: DocumentAction = "answer"
    document_name: str | None = None
    response_text: str = ""
    restart_from_beginning: bool = False
    sentence_idx: int | None = None
    note_text: str = ""
    highlight_color: str = "yellow"


def detect_direct_read_intent(user_text: str) -> DocumentTurnDecision | None:
    normalized = _normalize(user_text)
    if not normalized:
        return None

    if any(_normalize(keyword) in normalized for keyword in _CONTINUE_KEYWORDS):
        return DocumentTurnDecision(action="continue_reading")

    if any(_normalize(keyword) in normalized for keyword in _READ_KEYWORDS):
        return DocumentTurnDecision(
            action="read_document",
            restart_from_beginning=any(_normalize(keyword) in normalized for keyword in _BEGINNING_KEYWORDS),
        )

    return None


def _normalize(value: str) -> str:
    return _NON_WORD_PATTERN.sub(" ", value.lower()).strip()


def _extract_keywords(text: str) -> list[str]:
    keywords: list[str] = []
    for token in _normalize(text).split():
        if len(token) < 4 or token in _STOPWORDS or token in keywords:
            continue
        keywords.append(token)
    return keywords[:6]


def _pick_relevant_sentences(
    doc: ParsedDocument,
    user_text: str,
    last_read_sentence_idx: int,
    limit: int = 10,
) -> list[tuple[int, str]]:
    if not doc.sentences:
        return []

    picks: list[tuple[int, str]] = []
    seen: set[int] = set()

    if 0 <= last_read_sentence_idx < len(doc.sentences):
        start = max(0, last_read_sentence_idx - 2)
        end = min(len(doc.sentences), last_read_sentence_idx + 3)
        for idx in range(start, end):
            picks.append((idx, doc.sentences[idx]))
            seen.add(idx)

    keywords = _extract_keywords(user_text)
    if keywords:
        for idx, sentence in enumerate(doc.sentences):
            normalized_sentence = _normalize(sentence)
            if any(keyword in normalized_sentence for keyword in keywords):
                for neighbor in range(max(0, idx - 1), min(len(doc.sentences), idx + 2)):
                    if neighbor in seen:
                        continue
                    picks.append((neighbor, doc.sentences[neighbor]))
                    seen.add(neighbor)
            if len(picks) >= limit:
                break

    if not picks:
        for idx, sentence in enumerate(doc.sentences[: min(limit, len(doc.sentences))]):
            picks.append((idx, sentence))

    return picks[:limit]


def build_document_turn_context(
    *,
    user_text: str,
    active_document_id: str | None,
    last_read_sentence_idx: int,
) -> str | None:
    store = get_document_store()
    docs = store.list_documents()
    if not docs:
        return None

    lines = [DOCUMENT_TURN_PROMPT, "", "Available documents:"]
    for doc_meta in docs:
        lines.append(f"- {doc_meta['title']} (id: {doc_meta['doc_id']})")

    if active_document_id:
        doc = store.get_document(active_document_id)
        if doc:
            lines.extend(
                [
                    "",
                    f"Selected document: {doc.title} (id: {doc.doc_id})",
                    "A selected document is available for reading and question answering.",
                ]
            )
            excerpt = _pick_relevant_sentences(doc, user_text, last_read_sentence_idx)
            if excerpt:
                lines.append("Relevant selected-document excerpts:")
                lines.extend(f"[{idx}] {sentence}" for idx, sentence in excerpt)
            if last_read_sentence_idx >= 0:
                start = max(0, last_read_sentence_idx - 12)
                end = min(len(doc.sentences), last_read_sentence_idx + 1)
                lines.append("Document reading history up to the current point:")
                lines.extend(f"[{idx}] {doc.sentences[idx]}" for idx in range(start, end))
        else:
            lines.extend(["", "Selected document: none"])
    else:
        lines.extend(["", "Selected document: none"])

    return "\n".join(lines)


def parse_document_turn_response(raw: str) -> DocumentTurnDecision:
    text = raw.strip()
    match = _JSON_OBJECT_PATTERN.search(text)
    if not match:
        return DocumentTurnDecision(action="answer", response_text=text)

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return DocumentTurnDecision(action="answer", response_text=text)

    action = str(payload.get("action", "answer")).strip().lower()
    if action not in {
        "answer",
        "read_document",
        "continue_reading",
        "pause_reading",
        "ask_document_clarification",
        "list_documents",
        "save_note",
        "highlight_sentence",
    }:
        action = "answer"

    document_name = payload.get("document_name")
    if document_name is not None:
        document_name = str(document_name).strip() or None

    response_text = payload.get("response_text", "")
    if response_text is None:
        response_text = ""

    restart_from_beginning = bool(
        payload.get("restart_from_beginning")
        or payload.get("start_from_beginning")
        or payload.get("restart")
    )
    sentence_idx = payload.get("sentence_idx")
    try:
        parsed_sentence_idx = int(sentence_idx) if sentence_idx is not None else None
    except (TypeError, ValueError):
        parsed_sentence_idx = None

    note_text = payload.get("note_text") or payload.get("note") or ""
    highlight_color = str(payload.get("highlight_color") or payload.get("color") or "yellow").strip() or "yellow"

    return DocumentTurnDecision(
        action=action,
        document_name=document_name,
        response_text=str(response_text).strip(),
        restart_from_beginning=restart_from_beginning,
        sentence_idx=parsed_sentence_idx,
        note_text=str(note_text).strip(),
        highlight_color=highlight_color,
    )


def resolve_document_by_name(
    document_name: str | None,
    *,
    active_document_id: str | None = None,
) -> ParsedDocument | None:
    store = get_document_store()
    if active_document_id:
        active_doc = store.get_document(active_document_id)
        if active_doc and (
            not document_name
            or _normalize(document_name) in {
                _normalize(active_doc.title),
                _normalize(active_doc.doc_id),
                _normalize(active_doc.filename),
            }
        ):
            return active_doc

    if not document_name:
        return None

    normalized_target = _normalize(document_name)
    docs = store.list_documents()

    for doc_meta in docs:
        haystack = {
            _normalize(doc_meta["title"]),
            _normalize(doc_meta["doc_id"]),
            _normalize(str(doc_meta.get("filename", ""))),
        }
        if normalized_target in haystack:
            return store.get_document(str(doc_meta["doc_id"]))

    for doc_meta in docs:
        title = _normalize(doc_meta["title"])
        filename = _normalize(str(doc_meta.get("filename", "")))
        if normalized_target and (
            normalized_target in title
            or title in normalized_target
            or normalized_target in filename
        ):
            return store.get_document(str(doc_meta["doc_id"]))

    return None


def user_explicitly_named_document(user_text: str) -> bool:
    normalized_user_text = _normalize(user_text)
    if not normalized_user_text:
        return False

    for doc_meta in get_document_store().list_documents():
        candidates = [
            _normalize(doc_meta["title"]),
            _normalize(str(doc_meta["doc_id"])),
            _normalize(str(doc_meta.get("filename", ""))),
        ]
        for candidate in candidates:
            if candidate and candidate in normalized_user_text:
                return True

    return False
