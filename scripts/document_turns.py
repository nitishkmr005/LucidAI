"""
document_turns.py — Minimal document turn router for learning.

What this teaches:
  - Detect obvious reading commands without calling an LLM
  - Build document-aware context for Q&A
  - Parse a structured decision into an action the app can execute

Usage:
  uv run --project backend python scripts/document_turns.py
  uv run --project backend python scripts/document_turns.py "continue reading"
  uv run --project backend python scripts/document_turns.py "what does self attention mean?"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from typing import Literal

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


DocumentAction = Literal[
    "answer",
    "read_document",
    "continue_reading",
    "pause_reading",
    "ask_document_clarification",
    "list_documents",
    "save_note",
    "highlight_sentence",
    "web_search",
    "open_document",
]


@dataclass
class DocumentMeta:
    doc_id: str
    title: str
    filename: str
    sentences: list[str]


@dataclass
class DocumentTurnDecision:
    action: DocumentAction = "answer"
    document_name: str | None = None
    response_text: str = ""
    restart_from_beginning: bool = False
    sentence_idx: int | None = None
    note_text: str = ""
    highlight_color: str = "yellow"


STOPWORDS = {
    "a", "an", "and", "are", "can", "continue", "document", "explain", "from",
    "for", "how", "i", "in", "is", "it", "of", "please", "read", "resume",
    "tell", "the", "this", "to", "what", "where", "why", "you",
}

READ_PATTERNS = ("read", "start reading", "read aloud", "begin reading")
CONTINUE_PATTERNS = ("continue", "continue reading", "resume", "keep reading")
PAUSE_PATTERNS = ("pause", "stop reading", "hold on", "wait")


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def detect_direct_intent(user_text: str) -> DocumentTurnDecision | None:
    """
    Fast path for obvious commands.

    This is important because commands like "continue" should not wait on an LLM
    when we already know the exact action.
    """
    normalized = normalize(user_text)
    if not normalized:
        return None

    if any(pattern in normalized for pattern in PAUSE_PATTERNS):
        return DocumentTurnDecision(action="pause_reading")
    if any(pattern in normalized for pattern in CONTINUE_PATTERNS):
        return DocumentTurnDecision(action="continue_reading")
    if any(pattern in normalized for pattern in READ_PATTERNS):
        return DocumentTurnDecision(
            action="read_document",
            restart_from_beginning="beginning" in normalized or "start over" in normalized,
        )
    return None


def extract_keywords(user_text: str) -> list[str]:
    keywords: list[str] = []
    for token in normalize(user_text).split():
        if len(token) < 4 or token in STOPWORDS or token in keywords:
            continue
        keywords.append(token)
    return keywords


def pick_relevant_sentences(
    document: DocumentMeta,
    user_text: str,
    last_read_sentence_idx: int,
    limit: int = 5,
) -> list[tuple[int, str]]:
    """
    Pick nearby reading context first, then keyword matches.

    This is the core trick that makes answers feel grounded in the document
    instead of generic.
    """
    picked: list[tuple[int, str]] = []
    seen: set[int] = set()

    if 0 <= last_read_sentence_idx < len(document.sentences):
        for index in range(max(0, last_read_sentence_idx - 1), min(len(document.sentences), last_read_sentence_idx + 2)):
            picked.append((index, document.sentences[index]))
            seen.add(index)

    for keyword in extract_keywords(user_text):
        for index, sentence in enumerate(document.sentences):
            if keyword in normalize(sentence) and index not in seen:
                picked.append((index, sentence))
                seen.add(index)
            if len(picked) >= limit:
                return picked[:limit]

    if not picked:
        for index, sentence in enumerate(document.sentences[:limit]):
            picked.append((index, sentence))

    return picked[:limit]


def build_document_context(
    documents: list[DocumentMeta],
    active_document_id: str | None,
    user_text: str,
    last_read_sentence_idx: int,
) -> str:
    lines = ["Available documents:"]
    for document in documents:
        lines.append(f"- {document.title} (id: {document.doc_id})")

    active_document = next((doc for doc in documents if doc.doc_id == active_document_id), None)
    if active_document is None:
        lines.append("")
        lines.append("Selected document: none")
        return "\n".join(lines)

    lines.append("")
    lines.append(f"Selected document: {active_document.title}")
    lines.append("Relevant excerpts:")
    for index, sentence in pick_relevant_sentences(active_document, user_text, last_read_sentence_idx):
        lines.append(f"[{index}] {sentence}")

    return "\n".join(lines)


def parse_document_turn_response(raw_text: str) -> DocumentTurnDecision:
    """
    In production the LLM returns JSON. This parser turns that JSON into a
    strongly shaped decision object.
    """
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return DocumentTurnDecision(action="answer", response_text=raw_text.strip())

    return DocumentTurnDecision(
        action=payload.get("action", "answer"),
        document_name=payload.get("document_name"),
        response_text=payload.get("response_text", ""),
        restart_from_beginning=bool(payload.get("restart_from_beginning", False)),
        sentence_idx=payload.get("sentence_idx"),
        note_text=payload.get("note_text", ""),
        highlight_color=payload.get("highlight_color", "yellow"),
    )


def mock_llm_router(user_text: str, active_document: DocumentMeta | None) -> str:
    """
    Stand-in for a real LLM call.

    The goal here is to show the contract shape, not to be clever.
    """
    normalized = normalize(user_text)
    if "list" in normalized and "document" in normalized:
        return json.dumps({"action": "list_documents", "response_text": "Here are your uploaded documents."})
    if "highlight" in normalized:
        return json.dumps({"action": "highlight_sentence", "sentence_idx": 1, "highlight_color": "yellow"})
    if "note" in normalized:
        return json.dumps({"action": "save_note", "sentence_idx": 1, "note_text": "Review this section later."})
    if active_document and ("what" in normalized or "why" in normalized or "how" in normalized):
        return json.dumps(
            {
                "action": "answer",
                "response_text": (
                    "Answer from document context: self-attention lets each token use information "
                    "from other tokens in the same sequence."
                ),
            }
        )
    return json.dumps({"action": "ask_document_clarification", "response_text": "Which document should I use?"})


SAMPLE_DOCUMENTS = [
    DocumentMeta(
        doc_id="doc_001",
        title="Attention Is All You Need",
        filename="attention.md",
        sentences=[
            "Transformers replaced recurrence with attention.",
            "Self-attention lets each token look at the other tokens.",
            "Positional encoding is added because order still matters.",
            "This design allows more parallel training than recurrent models.",
        ],
    ),
    DocumentMeta(
        doc_id="doc_002",
        title="BERT Notes",
        filename="bert.md",
        sentences=[
            "BERT uses masked language modeling.",
            "Bidirectional context improves representation quality.",
        ],
    ),
]


def run_single_turn(user_text: str) -> None:
    active_document_id = "doc_001"
    last_read_sentence_idx = 1

    log("input", "user_text={!r}", user_text)

    direct_decision = detect_direct_intent(user_text)
    if direct_decision is not None:
        log("direct", "matched direct intent -> {}", json.dumps(asdict(direct_decision), indent=2))
        return

    context = build_document_context(
        documents=SAMPLE_DOCUMENTS,
        active_document_id=active_document_id,
        user_text=user_text,
        last_read_sentence_idx=last_read_sentence_idx,
    )
    log("context", "\n{}", context)

    active_document = next((doc for doc in SAMPLE_DOCUMENTS if doc.doc_id == active_document_id), None)
    mocked_llm_output = mock_llm_router(user_text, active_document)
    log("llm", "mocked_output={}", mocked_llm_output)

    parsed_decision = parse_document_turn_response(mocked_llm_output)
    log("decision", "parsed={}", json.dumps(asdict(parsed_decision), indent=2))


def run_demo_turns() -> None:
    for example in [
        "continue reading",
        "what does self attention mean?",
        "highlight that sentence",
        "save a note here",
        "list my documents",
    ]:
        run_single_turn(example)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal document turn learning script")
    parser.add_argument("user_text", nargs="*", help="Optional user utterance to route")
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    if args.user_text:
        run_single_turn(" ".join(args.user_text))
    else:
        run_demo_turns()
