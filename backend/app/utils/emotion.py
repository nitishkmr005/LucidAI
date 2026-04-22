from __future__ import annotations

import re

_TAG_PATTERN = re.compile(r"\[[^\]]+\]")

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)

_MARKDOWN_PATTERN = re.compile(
    r"#{1,6}\s?"
    r"|(\*{1,3}|_{1,3})"
    r"|`{1,3}"
    r"|^\s*[-*•]\s+"
    r"|^\s*\d+\.\s+",
    re.MULTILINE,
)


def strip_emotion_tags(text: str) -> str:
    """
    Remove inline emotion tags (e.g. [laugh], [sigh]) from a text string.

    Args:
        text: Raw LLM response string that may contain bracketed emotion tags.

    Returns:
        Cleaned string with all bracketed tags removed and whitespace normalised.

    Library:
        re (standard library) — uses _TAG_PATTERN to match and strip tags.
    """
    stripped = _TAG_PATTERN.sub("", text)
    return " ".join(stripped.split())


def clean_for_tts(text: str) -> str:
    """
    Strip markdown formatting and emojis so TTS receives plain speakable text.

    Args:
        text: Raw LLM response string potentially containing markdown and emojis.

    Returns:
        Plain text string safe for TTS — no asterisks, hashes, emojis, or
        symbols that would be read aloud verbatim.

    Library:
        re (standard library) — uses _MARKDOWN_PATTERN and _EMOJI_PATTERN.
    """
    text = _EMOJI_PATTERN.sub("", text)
    text = _MARKDOWN_PATTERN.sub("", text)
    return " ".join(text.split())
