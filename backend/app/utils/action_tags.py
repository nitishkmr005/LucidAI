from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

_DOC_ACTION_PATTERN = re.compile(r"\[DOC_ACTION:([^\]]+)\]")

ActionKind = Literal[
    "list_docs", "read", "highlight",
    "save_snippet", "search", "export",
    "reading_pause", "reading_resume",
]


@dataclass
class DocAction:
    kind: str
    params: list[str] = field(default_factory=list)


def extract_doc_actions(text: str) -> tuple[str, list[DocAction]]:
    """
    Remove all [DOC_ACTION:...] tags from text and return parsed actions.

    Returns:
        (cleaned_text, list_of_DocAction)
    """
    actions: list[DocAction] = []

    def _replacer(m: re.Match) -> str:
        payload = m.group(1)
        parts = payload.split(":")
        if parts:
            actions.append(DocAction(kind=parts[0], params=parts[1:]))
        return ""

    cleaned = _DOC_ACTION_PATTERN.sub(_replacer, text)
    return " ".join(cleaned.split()), actions


def actions_to_ws_messages(actions: list[DocAction]) -> list[dict]:
    """Convert extracted actions into WebSocket message dicts for dispatch."""
    messages: list[dict] = []
    for action in actions:
        if action.kind == "list_docs":
            messages.append({"type": "doc_list_requested"})
        elif action.kind == "read" and action.params:
            messages.append({"type": "doc_read_start", "doc_id": action.params[0]})
        elif action.kind == "highlight" and len(action.params) >= 2:
            try:
                messages.append({
                    "type": "doc_highlight",
                    "sentence_idx": int(action.params[0]),
                    "word_count": int(action.params[1]),
                })
            except (ValueError, IndexError):
                pass
        elif action.kind == "save_snippet" and action.params:
            messages.append({"type": "doc_save_snippet", "term": ":".join(action.params)})
        elif action.kind == "search" and action.params:
            messages.append({"type": "doc_search_start", "query": ":".join(action.params)})
        elif action.kind == "export" and action.params:
            messages.append({"type": "doc_export", "format": action.params[0]})
        elif action.kind == "reading_pause":
            messages.append({"type": "doc_reading_pause"})
        elif action.kind == "reading_resume":
            messages.append({"type": "doc_reading_resume"})
    return messages
