"""Shared LLM → TTS agent pipeline for NeuroTalk.

Both the WebSocket transport (main.py) and WebRTC transport (webrtc/session.py)
instantiate AgentPipeline and pass a `send_json` callable.  All conversation
state, LLM routing, document reading, and TTS streaming live here; transport-
specific code (audio capture, WebRTC peer connection, session logging) stays
in the caller.
"""
from __future__ import annotations

import asyncio
import base64
import re
from contextlib import suppress
from time import perf_counter
from typing import Callable, Coroutine

from loguru import logger

from app.services.document_store import get_document_store
from app.services.llm import complete_llm_response
from app.services.search import web_search
from app.services.tts import get_available_tts_voices, get_default_tts_voice, get_tts_service
from app.utils.document_turns import (
    DocumentTurnDecision,
    build_document_turn_context,
    detect_direct_read_intent,
    parse_document_turn_response,
    resolve_document_by_name,
    user_explicitly_named_document,
)
from app.utils.emotion import clean_for_tts
from app.utils.reading_patterns import (
    CONTINUE_READING_PATTERN,
    PAUSE_PATTERN,
    READ_FROM_BEGINNING_PATTERN,
    refers_to_current_sentence as _refers_to_current_sentence,
)
from config.settings import get_settings


class AgentPipeline:
    """
    Shared LLM → TTS pipeline for both WebSocket and WebRTC transports.

    Encapsulates conversation state, LLM intent routing, document reading, and
    TTS sentence streaming.  The caller provides:
      - ``send_json_fn``: async callable to push JSON to the client.
      - ``on_turn_start``: called at the start of each LLM turn (use to clear the
        audio buffer before processing — WebSocket transport uses this).
      - ``on_turn_complete``: called after a successful, uninterrupted LLM turn
        (use to clear the audio buffer after TTS finishes — WebRTC transport uses this).
      - ``on_llm_done``: optional logging hook ``(trigger, llm_ms, full_prompt, response, error)``.
      - ``on_tts_synth``: optional per-utterance logging hook ``(text, sr, latency_ms, error)``.
    """

    def __init__(
        self,
        session_id: str,
        send_json_fn: Callable[[dict], Coroutine],
        initial_voice: str | None = None,
        on_turn_start: Callable[[], None] | None = None,
        on_turn_complete: Callable[[], None] | None = None,
        on_llm_done: Callable[..., None] | None = None,
        on_tts_synth: Callable[..., None] | None = None,
    ) -> None:
        self._session_id = session_id
        self._send_json = send_json_fn
        self._settings = get_settings()
        self._on_turn_start = on_turn_start
        self._on_turn_complete = on_turn_complete
        self._on_llm_done = on_llm_done
        self._on_tts_synth = on_tts_synth

        # Conversation state
        self._conversation_history: list[dict[str, str]] = []
        self._active_document_id: str | None = None
        self._last_read_sentence_idx: int = -1
        self._resume_from_sentence_idx: int | None = None
        self._last_read_words: str = ""  # last 5 words of the most recently spoken document sentence
        self._last_answer_text: str = ""
        self._tts_voice: str = self._resolve_tts_voice(initial_voice)
        self._tts_speed: float = 1.0

        # Pipeline control
        self._interrupt_event = asyncio.Event()
        self._llm_task: asyncio.Task | None = None
        self._tts_task: asyncio.Task | None = None
        self._pending_llm_call: tuple[str, str] | None = None
        self._latest_llm_input: str = ""
        self._is_agent_speaking: bool = False
        self._llm_responded: bool = False

    # ── Public read-only state ────────────────────────────────────────────────

    @property
    def is_agent_speaking(self) -> bool:
        return self._is_agent_speaking

    @property
    def interrupt_event(self) -> asyncio.Event:
        return self._interrupt_event

    @property
    def llm_task(self) -> asyncio.Task | None:
        return self._llm_task

    @property
    def llm_responded(self) -> bool:
        return self._llm_responded

    @property
    def latest_llm_input(self) -> str:
        return self._latest_llm_input

    # ── Public mutable state ──────────────────────────────────────────────────

    @property
    def active_document_id(self) -> str | None:
        return self._active_document_id

    @active_document_id.setter
    def active_document_id(self, value: str | None) -> None:
        self._active_document_id = value

    @property
    def last_read_sentence_idx(self) -> int:
        return self._last_read_sentence_idx

    @last_read_sentence_idx.setter
    def last_read_sentence_idx(self, value: int) -> None:
        self._last_read_sentence_idx = value

    @property
    def resume_from_sentence_idx(self) -> int | None:
        return self._resume_from_sentence_idx

    @resume_from_sentence_idx.setter
    def resume_from_sentence_idx(self, value: int | None) -> None:
        self._resume_from_sentence_idx = value

    @property
    def tts_voice(self) -> str:
        return self._tts_voice

    @tts_voice.setter
    def tts_voice(self, value: str) -> None:
        self._tts_voice = self._resolve_tts_voice(value)

    @property
    def tts_speed(self) -> float:
        return self._tts_speed

    @tts_speed.setter
    def tts_speed(self, value: float) -> None:
        self._tts_speed = max(0.8, min(1.3, value))

    # ── Public actions ────────────────────────────────────────────────────────

    async def interrupt(self) -> None:
        """Cancel in-flight TTS and LLM tasks and signal the interrupt event."""
        self._interrupt_event.set()
        self._pending_llm_call = None
        self._is_agent_speaking = False

        if self._tts_task and not self._tts_task.done():
            self._tts_task.cancel()
            self._tts_task = None

        if self._llm_task and not self._llm_task.done():
            self._llm_task.cancel()
            self._llm_task = None

    def schedule_llm(self, text: str, trigger: str) -> None:
        """Schedule an LLM call, cancelling any stale in-flight call first.

        Args:
            text: Normalised user utterance to process.
            trigger: Label used in session log (e.g. ``"final"``).
        """
        normalized = text.strip()
        if len(normalized) < self._settings.stream_llm_min_chars:
            return
        if normalized == self._latest_llm_input:
            return
        if PAUSE_PATTERN.match(normalized):
            return

        if self._llm_task and not self._llm_task.done():
            logger.info("session_id={} event=llm_cancel_for_newer_text", self._session_id)
            self._interrupt_event.set()
            self._pending_llm_call = None
            if self._tts_task and not self._tts_task.done():
                self._tts_task.cancel()
                self._tts_task = None
            self._llm_task.cancel()
            self._llm_task = None

        self._pending_llm_call = (normalized, trigger)
        if self._llm_task is None or self._llm_task.done():
            next_text, next_trigger = self._pending_llm_call
            self._pending_llm_call = None
            self._llm_task = asyncio.create_task(self._run_llm(next_text, next_trigger))

    def schedule_document_read(
        self,
        doc_id: str,
        start_idx: int,
        user_text: str = "",
        llm_ms: float = 0.0,
    ) -> None:
        """Start a document read as the active LLM task.

        Args:
            doc_id: Document ID to read from the store.
            start_idx: Sentence index to begin reading from.
            user_text: Originating user utterance echoed in LLM events.
            llm_ms: LLM latency to forward to the read pipeline.
        """
        self._llm_task = asyncio.ensure_future(
            self._run_document_read(
                doc_id=doc_id, user_text=user_text, start_idx=start_idx, llm_ms=llm_ms
            )
        )

    def get_read_start_idx(self, *, restart_from_beginning: bool) -> int:
        """Compute the sentence index to start the next document read.

        Args:
            restart_from_beginning: When True, returns 0 regardless of saved position.

        Returns:
            Index of the first sentence to read next.
        """
        if restart_from_beginning:
            return 0
        if self._resume_from_sentence_idx is not None:
            return max(0, self._resume_from_sentence_idx)
        if self._last_read_sentence_idx >= 0:
            return self._last_read_sentence_idx + 1
        return 0

    def find_resume_sentence_idx(self, sentences: list[str]) -> int | None:
        """Search the last-read words in the document and return the next sentence index.

        Tries an exact phrase match first, then a majority-word match. Returns the
        index AFTER the matched sentence so reading continues forward, not repeats.

        Args:
            sentences: Ordered list of document sentences to search.

        Returns:
            Index of the sentence immediately after the match, or ``None`` if
            ``_last_read_words`` is empty or no match is found.
        """
        if not self._last_read_words or not sentences:
            return None
        search = self._last_read_words.lower()
        words = search.split()
        # Exact phrase match
        for idx, sentence in enumerate(sentences):
            if search in sentence.lower():
                next_idx = idx + 1
                return next_idx if next_idx < len(sentences) else None
        # Majority-word match: at least (n-1) of n tracked words present
        threshold = max(1, len(words) - 1)
        for idx, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            if sum(1 for w in words if w in sentence_lower) >= threshold:
                next_idx = idx + 1
                return next_idx if next_idx < len(sentences) else None
        return None

    async def run_welcome(self) -> None:
        """Synthesise and stream the configured welcome message on session start.

        Barge-in is disabled for the welcome utterance.
        """
        welcome = self._settings.welcome_message
        if not welcome:
            return
        await self._send_json({"type": "llm_start", "user_text": ""})
        await self._send_json({"type": "llm_final", "text": welcome, "llm_ms": 0})

        sent_queue: asyncio.Queue[tuple[str, int | None, str | None] | None] = asyncio.Queue()
        for raw in re.split(r"(?<=[.!?])\s+", welcome.strip()):
            sentence = clean_for_tts(raw.strip())
            if sentence:
                sent_queue.put_nowait((sentence, None, None))
        sent_queue.put_nowait(None)

        await self._tts_sentence_pipeline(sent_queue, enable_barge_in=False)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_tts_voice(self, voice: str | None) -> str:
        """Return a validated TTS voice ID, falling back to the default if invalid.

        Args:
            voice: Requested voice ID to validate against the available list.

        Returns:
            A voice ID that exists in the available voices list.
        """
        valid_voices = {item["id"] for item in get_available_tts_voices()}
        if voice and voice in valid_voices:
            return voice
        return get_default_tts_voice()

    def _split_text_for_tts(self, text: str) -> list[tuple[str, int | None, str | None]]:
        """Split plain text into TTS utterance tuples at sentence boundaries.

        Args:
            text: Raw text to split.

        Returns:
            List of ``(sentence, sentence_idx, display_text)`` tuples where
            ``sentence_idx`` and ``display_text`` are both ``None`` for Q&A turns.
        """
        utterances: list[tuple[str, int | None, str | None]] = []
        for raw in re.split(r"(?<=[.!?])\s+", text.strip()):
            sentence = clean_for_tts(raw.strip())
            if sentence:
                utterances.append((sentence, None, None))
        return utterances

    def _build_prompt_dump(self, user_text: str, document_context: str | None) -> str:
        """Serialise the full LLM prompt for session logging.

        Args:
            user_text: The user's current utterance.
            document_context: Surrounding document sentences, or ``None`` when no
                document is active.

        Returns:
            Human-readable string combining system prompt, document context,
            conversation history, and the current user turn.
        """
        history = "\n".join(
            f"{entry['role']}: {entry['content']}" for entry in self._conversation_history
        )
        parts = [
            f"system: {self._settings.llm_system_prompt}",
            f"document_context: {document_context or ''}",
            f"history:\n{history}" if history else "history:",
            f"user: {user_text}",
        ]
        return "\n\n".join(parts)

    # ── TTS pipeline ──────────────────────────────────────────────────────────

    async def _tts_sentence_pipeline(
        self,
        queue: asyncio.Queue[tuple[str, int | None, str | None] | None],
        enable_barge_in: bool = True,
    ) -> None:
        """Consume sentences from *queue* and stream TTS audio to the client.

        Runs until a ``None`` sentinel is received or the interrupt event is set.
        Emits ``tts_start``, ``tts_audio``, and ``tts_done`` / ``tts_interrupted``
        JSON events; updates ``_last_read_sentence_idx`` for document reads.

        Args:
            queue: Producer-filled queue of ``(tts_text, sentence_idx, display_text)``
                tuples; a ``None`` item terminates the loop.
            enable_barge_in: When True, sets ``_is_agent_speaking`` so the VAD
                barge-in path can interrupt. Should be False during document reading.
        """
        tts_service = get_tts_service()
        tts_started = False
        tts_t0 = perf_counter()
        if enable_barge_in:
            self._is_agent_speaking = True

        while True:
            item = await queue.get()
            if item is None or self._interrupt_event.is_set():
                break
            tts_text, sentence_idx, display_text = item
            synth_t0 = perf_counter()
            try:
                wav_bytes, sr = await tts_service.synthesize(
                    tts_text, voice=self._tts_voice, speed=self._tts_speed
                )
            except Exception as err:
                logger.warning("session_id={} event=tts_error error={}", self._session_id, err)
                if self._on_tts_synth:
                    self._on_tts_synth(tts_text, 0, round((perf_counter() - synth_t0) * 1000, 2), str(err))
                continue
            latency_ms = round((perf_counter() - synth_t0) * 1000, 2)
            if self._on_tts_synth:
                self._on_tts_synth(tts_text, sr, latency_ms, None)
            if self._interrupt_event.is_set():
                break
            if not tts_started:
                tts_started = True
                tts_t0 = perf_counter()
                await self._send_json({"type": "tts_start"})
            if sentence_idx is not None:
                self._last_read_sentence_idx = sentence_idx
                spoken = display_text or tts_text
                self._last_read_words = " ".join(spoken.split()[-5:])
                # _resume_from_sentence_idx is intentionally NOT updated here.
                # It is only used to restore a cross-session saved position before
                # reading starts; _run_document_read clears it immediately so that
                # in-session resumes use _last_read_sentence_idx + 1 (next sentence).
                if self._active_document_id:
                    get_document_store().save_reading_position(self._active_document_id, sentence_idx)
                await self._send_json({
                    "type": "doc_highlight",
                    "sentence_idx": sentence_idx,
                    "word_count": len((display_text or tts_text).split()),
                })
            tts_ms = round((perf_counter() - tts_t0) * 1000, 2)
            wav_b64 = base64.b64encode(wav_bytes).decode()
            await self._send_json({
                "type": "tts_audio",
                "data": wav_b64,
                "sample_rate": sr,
                "tts_ms": tts_ms,
                "sentence_text": display_text or tts_text,
                "sentence_idx": sentence_idx,
            })

        self._is_agent_speaking = False
        if self._interrupt_event.is_set():
            await self._send_json({"type": "tts_interrupted"})
        else:
            await self._send_json({"type": "tts_done"})

    async def _play_tts_turn(
        self,
        *,
        user_text: str,
        display_text: str,
        utterances: list[tuple[str, int | None, str | None]],
        llm_ms: float,
        enable_barge_in: bool = True,
    ) -> None:
        """Emit LLM response events then pipe utterances through _tts_sentence_pipeline.

        Sends ``llm_start``, ``llm_partial``, and ``llm_final``, then delegates
        audio synthesis and streaming to ``_tts_sentence_pipeline``.

        Args:
            user_text: The original user utterance echoed in ``llm_start``.
            display_text: Rendered text to show in the UI.
            utterances: Ordered list of ``(tts_text, sentence_idx, display_text)``
                tuples to synthesise.
            llm_ms: LLM inference latency included in the ``llm_final`` event.
            enable_barge_in: Forwarded to ``_tts_sentence_pipeline``.
        """
        await self._send_json({"type": "llm_start", "user_text": user_text})
        await self._send_json({"type": "llm_partial", "text": display_text})
        await self._send_json({"type": "llm_final", "text": display_text, "llm_ms": llm_ms})
        if not utterances:
            return

        sent_queue: asyncio.Queue[tuple[str, int | None, str | None] | None] = asyncio.Queue()
        for utterance in utterances:
            await sent_queue.put(utterance)
        await sent_queue.put(None)

        tts_task = asyncio.create_task(
            self._tts_sentence_pipeline(sent_queue, enable_barge_in=enable_barge_in)
        )
        self._tts_task = tts_task
        with suppress(asyncio.CancelledError):
            await tts_task
        self._tts_task = None

    # ── Document reading ──────────────────────────────────────────────────────

    async def _run_document_read(
        self,
        *,
        doc_id: str,
        user_text: str,
        start_idx: int,
        llm_ms: float,
    ) -> str:
        """Stream the requested document sentences as TTS from *start_idx* onward.

        Clears ``_resume_from_sentence_idx`` immediately so in-session interrupts
        resume via ``_last_read_sentence_idx`` rather than a stale cross-session index.

        Args:
            doc_id: Document ID to fetch from the document store.
            user_text: Originating user utterance echoed in LLM events.
            start_idx: Sentence index to start reading from.
            llm_ms: LLM latency forwarded to ``_play_tts_turn``.

        Returns:
            Short status string describing what was read, or a "finished" message
            if start_idx is past the last sentence.
        """
        self._interrupt_event.clear()
        doc = get_document_store().get_document(doc_id)
        if not doc:
            await self._play_tts_turn(
                user_text=user_text,
                display_text="I couldn't find that document.",
                utterances=self._split_text_for_tts("I couldn't find that document."),
                llm_ms=llm_ms,
            )
            return "I couldn't find that document."

        if start_idx >= doc.sentence_count:
            logger.info(
                "session_id={} event=doc_read_finished doc_id={} start_idx={} sentence_count={}",
                self._session_id, doc_id, start_idx, doc.sentence_count,
            )
            # Reset so that a subsequent "keep reading" starts fresh from the beginning.
            self._last_read_sentence_idx = -1
            self._resume_from_sentence_idx = None
            await self._send_json({"type": "doc_reading_pause"})
            finished_msg = f"That's the end of {doc.title}. Would you like me to read it again from the beginning?"
            await self._play_tts_turn(
                user_text=user_text,
                display_text=finished_msg,
                utterances=self._split_text_for_tts(finished_msg),
                llm_ms=llm_ms,
            )
            return finished_msg

        self._active_document_id = doc.doc_id
        # Clear cross-session restore index now that start_idx is confirmed.
        # Subsequent in-session interrupts will use _last_read_sentence_idx + 1.
        self._resume_from_sentence_idx = None
        await self._send_json({
            "type": "doc_read_start",
            "doc_id": doc.doc_id,
            "sentences": doc.sentences,
            "title": doc.title,
        })

        utterances = [
            (clean_for_tts(doc.sentences[idx]), idx, doc.sentences[idx])
            for idx in range(start_idx, len(doc.sentences))
            if clean_for_tts(doc.sentences[idx])
        ]
        await self._play_tts_turn(
            user_text=user_text,
            display_text="",
            utterances=utterances,
            llm_ms=llm_ms,
            enable_barge_in=False,
        )
        if not self._interrupt_event.is_set():
            self._resume_from_sentence_idx = None
        return f"Reading {doc.title} from sentence {start_idx}."

    # ── LLM dispatch ──────────────────────────────────────────────────────────

    async def _run_llm(self, text: str, trigger: str) -> None:
        """Run one full LLM → dispatch → TTS turn for the given user utterance.

        Detects direct-read intent first (no LLM call needed), otherwise calls
        the LLM, parses the decision, and delegates to ``_dispatch_decision`` or
        a plain TTS turn. Fires all registered callbacks and updates conversation
        history on success.

        Args:
            text: Normalised user utterance to process.
            trigger: Label for session logging (e.g. ``"final"``).
        """
        self._llm_responded = False
        self._interrupt_event.clear()
        self._latest_llm_input = text

        if self._on_turn_start:
            self._on_turn_start()

        display_response = ""
        llm_ms = 0.0
        call_error: str | None = None
        full_prompt = ""
        llm_t0 = perf_counter()

        try:
            direct_read = detect_direct_read_intent(text)
            if direct_read and self._active_document_id:
                if direct_read.restart_from_beginning:
                    self._last_read_sentence_idx = -1
                    self._resume_from_sentence_idx = None
                elif (
                    direct_read.action == "continue_reading"
                    and self._resume_from_sentence_idx is None
                    and self._last_read_sentence_idx < 0
                ):
                    self._resume_from_sentence_idx = 0
                if direct_read.action == "continue_reading" and not direct_read.restart_from_beginning:
                    _doc = get_document_store().get_document(self._active_document_id)
                    _word_idx = self.find_resume_sentence_idx(_doc.sentences) if _doc else None
                    start_idx = _word_idx if _word_idx is not None else self.get_read_start_idx(restart_from_beginning=False)
                else:
                    start_idx = self.get_read_start_idx(restart_from_beginning=direct_read.restart_from_beginning)
                logger.info(
                    "session_id={} event=direct_read_resume action={} last_words={!r} start_idx={}",
                    self._session_id, direct_read.action, self._last_read_words, start_idx,
                )
                display_response = await self._run_document_read(
                    doc_id=self._active_document_id,
                    user_text=text,
                    start_idx=start_idx,
                    llm_ms=0.0,
                )
                return

            document_context = build_document_turn_context(
                user_text=text,
                active_document_id=self._active_document_id,
                last_read_sentence_idx=self._last_read_sentence_idx,
            )
            full_prompt = self._build_prompt_dump(text, document_context)
            full_response = await complete_llm_response(
                text,
                conversation_history=list(self._conversation_history),
                document_context=document_context,
            )
            llm_ms = round((perf_counter() - llm_t0) * 1000, 2)
            logger.info("session_id={} event=llm_done llm_ms={}", self._session_id, llm_ms)

            # Parse + Pydantic-validate the JSON decision with retries.
            # Raw LLM JSON is never forwarded to the UI.
            decision: DocumentTurnDecision | None = None
            json_parse_failed = False
            if document_context:
                max_attempts = 1 + self._settings.llm_json_retry_attempts
                for attempt in range(max_attempts):
                    if attempt > 0:
                        logger.info(
                            "session_id={} event=llm_json_retry attempt={}/{}",
                            self._session_id, attempt + 1, max_attempts,
                        )
                        full_response = await complete_llm_response(
                            text,
                            conversation_history=list(self._conversation_history),
                            document_context=document_context,
                        )
                    try:
                        decision = parse_document_turn_response(full_response)
                        break
                    except ValueError as exc:
                        logger.warning(
                            "session_id={} event=llm_json_invalid attempt={}/{} error={}",
                            self._session_id, attempt + 1, max_attempts, exc,
                        )
                else:
                    json_parse_failed = True
                    logger.error(
                        "session_id={} event=llm_json_retry_exhausted attempts={}",
                        self._session_id, max_attempts,
                    )

            # Pattern-based overrides take priority over the LLM decision.
            if self._active_document_id and CONTINUE_READING_PATTERN.match(text):
                active_doc = get_document_store().get_document(self._active_document_id)
                if active_doc:
                    decision = DocumentTurnDecision(
                        action="continue_reading",
                        document_name=active_doc.title,
                        response_text="",
                        restart_from_beginning=False,
                    )
            elif self._active_document_id and READ_FROM_BEGINNING_PATTERN.search(text):
                active_doc = get_document_store().get_document(self._active_document_id)
                if active_doc:
                    decision = DocumentTurnDecision(
                        action="read_document",
                        document_name=active_doc.title,
                        response_text="",
                        restart_from_beginning=True,
                    )

            if decision is not None:
                display_response = await self._dispatch_decision(text, decision, llm_ms)
            elif json_parse_failed:
                # All retries exhausted — use a safe spoken fallback, never raw JSON.
                display_response = "I'm having trouble processing that. Could you try again?"
                await self._play_tts_turn(
                    user_text=text,
                    display_text=display_response,
                    utterances=self._split_text_for_tts(display_response),
                    llm_ms=llm_ms,
                )
            else:
                # No document context — plain-text LLM response is safe to speak directly.
                display_response = full_response.strip()
                await self._play_tts_turn(
                    user_text=text,
                    display_text=display_response,
                    utterances=self._split_text_for_tts(display_response),
                    llm_ms=llm_ms,
                )

        except Exception as err:
            llm_ms = round((perf_counter() - llm_t0) * 1000, 2)
            call_error = str(err)
            logger.warning("session_id={} event=llm_error error={}", self._session_id, err)
            await self._send_json(
                {"type": "llm_error", "message": "LLM unavailable — is Ollama running?"}
            )

        if self._on_llm_done:
            self._on_llm_done(trigger, llm_ms, full_prompt, display_response, call_error)

        if not self._interrupt_event.is_set():
            if self._on_turn_complete:
                self._on_turn_complete()

        if display_response and call_error is None and not self._interrupt_event.is_set():
            self._llm_responded = True
            if not display_response.startswith("Reading "):
                self._last_answer_text = display_response
            self._conversation_history.append({"role": "user", "content": text})
            self._conversation_history.append({"role": "assistant", "content": display_response})
            max_msgs = self._settings.llm_max_history_turns * 2
            if len(self._conversation_history) > max_msgs:
                self._conversation_history[:] = self._conversation_history[-max_msgs:]

        if self._pending_llm_call:
            next_text, next_trigger = self._pending_llm_call
            self._pending_llm_call = None
            if next_text != self._latest_llm_input:
                self._llm_task = asyncio.create_task(self._run_llm(next_text, next_trigger))
                return
        self._llm_task = None

    async def _dispatch_decision(
        self, text: str, decision: DocumentTurnDecision, llm_ms: float
    ) -> str:
        """Execute the action chosen by the LLM and return the agent's spoken response.

        Handles the full set of ``DocumentTurnDecision`` actions: list_documents,
        ask_document_clarification, pause_reading, read_document, continue_reading,
        save_note, highlight_sentence, open_document, web_search, and fallback.

        Args:
            text: Original user utterance used to echo in TTS events.
            decision: Parsed LLM decision containing action and metadata.
            llm_ms: LLM latency forwarded to ``_play_tts_turn``.

        Returns:
            The text the agent spoke aloud.
        """
        display_response = ""

        if decision.action == "list_documents":
            docs = get_document_store().list_documents()
            await self._send_json({"type": "doc_list", "documents": docs})
            response_text = decision.response_text or (
                "Available documents are: " + ", ".join(doc["title"] for doc in docs)
            )
            await self._play_tts_turn(
                user_text=text,
                display_text=response_text,
                utterances=self._split_text_for_tts(response_text),
                llm_ms=llm_ms,
            )
            display_response = response_text

        elif decision.action == "ask_document_clarification":
            response_text = decision.response_text or "Which document would you like me to read?"
            await self._play_tts_turn(
                user_text=text,
                display_text=response_text,
                utterances=self._split_text_for_tts(response_text),
                llm_ms=llm_ms,
            )
            display_response = response_text

        elif decision.action == "pause_reading":
            self._resume_from_sentence_idx = None
            response_text = decision.response_text or "Pausing here."
            await self._send_json({"type": "doc_reading_pause"})
            await self._play_tts_turn(
                user_text=text,
                display_text=response_text,
                utterances=self._split_text_for_tts(response_text),
                llm_ms=llm_ms,
            )
            display_response = response_text

        elif decision.action in {"read_document", "continue_reading"}:
            if decision.action == "read_document" and not user_explicitly_named_document(text) and not self._active_document_id:
                response_text = "Which document should I read?"
                await self._play_tts_turn(
                    user_text=text,
                    display_text=response_text,
                    utterances=self._split_text_for_tts(response_text),
                    llm_ms=llm_ms,
                )
                display_response = response_text
            else:
                target_doc = resolve_document_by_name(
                    decision.document_name,
                    active_document_id=self._active_document_id if decision.action == "continue_reading" else None,
                )
                if target_doc is None:
                    response_text = "I couldn't tell which document to read. Which one do you want?"
                    await self._play_tts_turn(
                        user_text=text,
                        display_text=response_text,
                        utterances=self._split_text_for_tts(response_text),
                        llm_ms=llm_ms,
                    )
                    display_response = response_text
                else:
                    # continue_reading must never restart; guard against LLM hallucination.
                    if decision.action == "continue_reading":
                        decision.restart_from_beginning = False
                    if decision.restart_from_beginning:
                        self._last_read_sentence_idx = -1
                        self._resume_from_sentence_idx = None

                    if decision.action == "continue_reading":
                        _word_idx = self.find_resume_sentence_idx(target_doc.sentences)
                        start_idx = _word_idx if _word_idx is not None else self.get_read_start_idx(restart_from_beginning=False)
                    else:
                        start_idx = self.get_read_start_idx(restart_from_beginning=decision.restart_from_beginning)

                    logger.info(
                        "session_id={} event=llm_path_resume action={} last_words={!r} start_idx={}",
                        self._session_id, decision.action, self._last_read_words, start_idx,
                    )
                    display_response = await self._run_document_read(
                        doc_id=target_doc.doc_id,
                        user_text=text,
                        start_idx=start_idx,
                        llm_ms=llm_ms,
                    )

        elif decision.action == "save_note":
            sentence_idx = decision.sentence_idx
            if sentence_idx is None or sentence_idx < 0:
                sentence_idx = max(0, self._last_read_sentence_idx)
            note_text = decision.note_text or self._last_answer_text or decision.response_text
            if not self._active_document_id or not note_text:
                response_text = "I do not have a note to save yet."
            else:
                snippet = get_document_store().save_snippet(
                    doc_id=self._active_document_id,
                    term="AI note",
                    explanation=note_text,
                    sentence_idx=sentence_idx,
                    word_idx=-1,
                )
                await self._send_json({"type": "doc_note_saved", "snippet": snippet})
                response_text = decision.response_text or "Saved that note on the sentence."
            await self._play_tts_turn(
                user_text=text,
                display_text=response_text,
                utterances=self._split_text_for_tts(response_text),
                llm_ms=llm_ms,
            )
            display_response = response_text

        elif decision.action == "highlight_sentence":
            sentence_idx = decision.sentence_idx
            # If user said "highlight this/that sentence" (vague reference to what's being read),
            # ignore whatever index the LLM guessed and use the actual last-read position.
            if sentence_idx is None or sentence_idx < 0 or _refers_to_current_sentence(text):
                sentence_idx = max(0, self._last_read_sentence_idx)
            color = decision.highlight_color or "yellow"
            if not self._active_document_id:
                response_text = "I need an open document before I can highlight a sentence."
            else:
                get_document_store().save_highlight(self._active_document_id, sentence_idx, color)
                await self._send_json({
                    "type": "doc_highlight_saved",
                    "sentence_idx": sentence_idx,
                    "color": color,
                })
                response_text = decision.response_text or "Highlighted that sentence."
            await self._play_tts_turn(
                user_text=text,
                display_text=response_text,
                utterances=self._split_text_for_tts(response_text),
                llm_ms=llm_ms,
            )
            display_response = response_text

        elif decision.action == "open_document":
            target_doc = resolve_document_by_name(decision.document_name, active_document_id=None)
            if target_doc is None:
                response_text = decision.response_text or "I couldn't find that document. Which one would you like?"
                await self._play_tts_turn(
                    user_text=text,
                    display_text=response_text,
                    utterances=self._split_text_for_tts(response_text),
                    llm_ms=llm_ms,
                )
                display_response = response_text
            else:
                self._active_document_id = target_doc.doc_id
                self._last_read_sentence_idx = -1
                self._resume_from_sentence_idx = None
                annotations = get_document_store().load_annotations(target_doc.doc_id)
                await self._send_json({
                    "type": "doc_opened",
                    "doc_id": target_doc.doc_id,
                    "title": target_doc.title,
                    "raw_markdown": target_doc.raw_markdown,
                    "sentences": target_doc.sentences,
                    "annotations": annotations,
                })
                display_response = await self._run_document_read(
                    doc_id=target_doc.doc_id,
                    user_text=text,
                    start_idx=0,
                    llm_ms=llm_ms,
                )

        elif decision.action == "web_search":
            query = decision.response_text or text
            results = await web_search(query, max_results=self._settings.web_search_max_results)
            await self._send_json({"type": "doc_search_result", "query": query, "results": results})
            if results:
                snippets = [r.get("snippet", "") for r in results[:3] if r.get("snippet")]
                spoken_text = ". ".join(snippets) if snippets else "Here is what I found online."
                source_lines = "\n".join(
                    f"• {r['title']} — {r['url']}"
                    for r in results[:3] if r.get("url") and r.get("title")
                )
                display_text = spoken_text + (f"\n\nSources:\n{source_lines}" if source_lines else "")
            else:
                spoken_text = "I searched but could not find relevant results."
                display_text = spoken_text
            await self._play_tts_turn(
                user_text=text,
                display_text=display_text,
                utterances=self._split_text_for_tts(spoken_text),
                llm_ms=llm_ms,
            )
            display_response = display_text

        else:
            response_text = decision.response_text or "I'm not sure how to help with that yet."
            await self._play_tts_turn(
                user_text=text,
                display_text=response_text,
                utterances=self._split_text_for_tts(response_text),
                llm_ms=llm_ms,
            )
            display_response = response_text

        return display_response
