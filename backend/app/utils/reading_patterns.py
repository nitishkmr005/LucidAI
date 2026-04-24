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
