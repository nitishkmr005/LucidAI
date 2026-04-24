"""Shared voice-command regex patterns used by both WebSocket and WebRTC transports."""
from __future__ import annotations

import re

PAUSE_PATTERN = re.compile(
    r"^\s*(wait|hold on|hold up|one moment|one sec(?:ond)?|just a (?:moment|second|sec)|"
    r"give me a (?:second|moment|sec)|hang on|please wait|just wait|ok wait|okay wait|"
    r"stop|stop it|stop please|please stop|ok stop|okay stop)\s*[.!?,]?\s*$",
    re.IGNORECASE,
)

CONTINUE_READING_PATTERN = re.compile(
    r"^\s*("
    r"keep reading|continue reading|continue|resume reading|resume|"
    r"start reading from where (you|we) left( off)?|"
    r"(continue|carry on|go on) from where (you|we) (stopped|left|paused|were)|"
    r"pick up (where|from) (you|we) left( off)?|"
    r"go on|carry on"
    r")\s*[.!?,]?\s*$",
    re.IGNORECASE,
)

READ_FROM_BEGINNING_PATTERN = re.compile(
    r"\b(read|start reading|read aloud|resume reading|continue reading)\b.*\b(beginning|start|top)\b"
    r"|\bstart over\b|\brestart\b",
    re.IGNORECASE,
)

# Matches requests to highlight the sentence currently being read (vague "this" reference).
_CURRENT_SENTENCE_HIGHLIGHT_PATTERN = re.compile(
    r"\b(highlight|mark|emphasize)\b.{0,40}\b(this|that|current|the sentence (you('re| are)|were|just)|what you (just |were )?read)\b"
    r"|\b(highlight|mark)\s+(this|that)\b",
    re.IGNORECASE,
)


def refers_to_current_sentence(text: str) -> bool:
    """Return True when the user's highlight request refers to the actively-read sentence.

    Args:
        text: Raw user utterance.

    Returns:
        True if the utterance is a vague "highlight this" / "highlight what you just read"
        reference that should resolve to the last-read sentence index rather than a
        content-matched sentence chosen by the LLM.
    """
    return bool(_CURRENT_SENTENCE_HIGHLIGHT_PATTERN.search(text))
