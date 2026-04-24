"""
webrtc_session.py — Minimal WebRTC session orchestration demo for learning.

What this teaches:
  - A WebRTC session is mostly an event router plus state machine
  - The data channel carries JSON control messages
  - Reading, interruption, Q&A, and resume all depend on preserved state

Why this script is fake on purpose:
  - No browser
  - No RTP audio
  - No aiortc peer connection

Instead, it focuses on the control flow used by the real session module.

Usage:
  uv run --project backend python scripts/webrtc_session.py
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from dataclasses import dataclass

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


@dataclass
class Document:
    doc_id: str
    title: str
    sentences: list[str]


class DemoDocumentStore:
    """
    Minimal document store used only by this learning script.

    It preserves the one state value that matters most for reading continuity:
    the sentence index to resume from after an interruption.
    """

    def __init__(self) -> None:
        self._documents = {
            "doc_001": Document(
                doc_id="doc_001",
                title="Attention Is All You Need",
                sentences=[
                    "Transformers replaced recurrence with attention.",
                    "Self-attention lets each token look at other tokens in the sequence.",
                    "Positional information is added because token order still matters.",
                    "This design allows much more parallel training than recurrent networks.",
                ],
            )
        }
        self._reading_positions: dict[str, int] = {}

    def get_document(self, doc_id: str) -> Document | None:
        return self._documents.get(doc_id)

    def save_reading_position(self, doc_id: str, sentence_idx: int) -> None:
        self._reading_positions[doc_id] = sentence_idx
        log("store", "saved reading_position doc_id={} sentence_idx={}", doc_id, sentence_idx)

    def get_reading_position(self, doc_id: str) -> int | None:
        return self._reading_positions.get(doc_id)


class FakeDataChannel:
    """
    The real session sends JSON over an RTCDataChannel.

    Here we just log each outbound payload so you can see what the frontend
    would receive.
    """

    async def send_json(self, payload: dict) -> None:
        log("outbound", "{}", payload)


class WebRTCSessionDemo:
    """
    Small orchestration model of the real WebRTC session.

    Important state:
      - active document
      - last sentence read
      - resume sentence
      - whether the agent is currently speaking
    """

    def __init__(self, store: DemoDocumentStore) -> None:
        self.session_id = uuid.uuid4().hex[:8]
        self.store = store
        self.dc = FakeDataChannel()
        self.active_document_id: str | None = None
        self.last_read_sentence_idx = -1
        self.resume_from_sentence_idx: int | None = None
        self.is_agent_speaking = False
        self.reading_task: asyncio.Task | None = None
        self.interrupt_event = asyncio.Event()

    async def start(self) -> None:
        log("session", "session_id={} ready", self.session_id)
        await self.dc.send_json({"type": "ready", "request_id": self.session_id})

    async def handle_message(self, payload: dict) -> None:
        """
        Route one frontend-style event.

        This mirrors the role of `_handle_dc_message()` in the real session.
        """
        event_type = payload.get("type")
        log("inbound", "received {}", payload)

        if event_type == "doc_load":
            await self._handle_doc_load(str(payload["doc_id"]))
        elif event_type == "doc_read":
            await self._handle_doc_read(str(payload["doc_id"]))
        elif event_type == "interrupt":
            await self._handle_interrupt()
        elif event_type == "user_question":
            await self._answer_question(str(payload["text"]))
        elif event_type == "continue_reading":
            await self._continue_reading()

    async def _handle_doc_load(self, doc_id: str) -> None:
        document = self.store.get_document(doc_id)
        if document is None:
            await self.dc.send_json({"type": "doc_error", "message": "Document not found"})
            return

        self.active_document_id = doc_id
        self.resume_from_sentence_idx = self.store.get_reading_position(doc_id)
        if self.resume_from_sentence_idx is not None:
            self.last_read_sentence_idx = self.resume_from_sentence_idx - 1

        await self.dc.send_json(
            {
                "type": "doc_opened",
                "doc_id": document.doc_id,
                "title": document.title,
                "sentences": document.sentences,
                "resume_from_sentence_idx": self.resume_from_sentence_idx,
            }
        )

    async def _handle_doc_read(self, doc_id: str) -> None:
        document = self.store.get_document(doc_id)
        if document is None:
            await self.dc.send_json({"type": "doc_error", "message": "Document not found"})
            return

        self.active_document_id = doc_id
        start_idx = self.store.get_reading_position(doc_id) or 0
        await self._start_reading(document, start_idx)

    async def _continue_reading(self) -> None:
        if self.active_document_id is None:
            await self.dc.send_json({"type": "doc_error", "message": "No active document"})
            return

        document = self.store.get_document(self.active_document_id)
        if document is None:
            await self.dc.send_json({"type": "doc_error", "message": "Document not found"})
            return

        start_idx = self.resume_from_sentence_idx or max(self.last_read_sentence_idx + 1, 0)
        await self._start_reading(document, start_idx)

    async def _start_reading(self, document: Document, start_idx: int) -> None:
        # Cancel any existing read before starting another one.
        if self.reading_task and not self.reading_task.done():
            self.reading_task.cancel()
            try:
                await self.reading_task
            except asyncio.CancelledError:
                pass

        self.interrupt_event.clear()
        self.reading_task = asyncio.create_task(self._run_document_read(document, start_idx))

    async def _run_document_read(self, document: Document, start_idx: int) -> None:
        """
        Simulate the TTS sentence pipeline used during reading mode.

        Each loop iteration stands in for:
          sentence -> TTS -> frontend highlight -> audio playback
        """
        self.is_agent_speaking = True
        await self.dc.send_json(
            {
                "type": "doc_read_start",
                "doc_id": document.doc_id,
                "title": document.title,
                "start_idx": start_idx,
            }
        )

        try:
            for sentence_idx in range(start_idx, len(document.sentences)):
                if self.interrupt_event.is_set():
                    break

                sentence = document.sentences[sentence_idx]
                self.last_read_sentence_idx = sentence_idx
                self.resume_from_sentence_idx = sentence_idx
                self.store.save_reading_position(document.doc_id, sentence_idx)

                await self.dc.send_json(
                    {
                        "type": "doc_highlight",
                        "sentence_idx": sentence_idx,
                        "sentence_text": sentence,
                    }
                )
                await self.dc.send_json(
                    {
                        "type": "tts_audio",
                        "sentence_idx": sentence_idx,
                        "sentence_text": sentence,
                    }
                )

                # Sleep represents playback time. During this gap, an interrupt can
                # arrive and stop the read mid-stream.
                await asyncio.sleep(0.35)

            if self.interrupt_event.is_set():
                await self.dc.send_json({"type": "tts_interrupted"})
            else:
                self.resume_from_sentence_idx = None
                await self.dc.send_json({"type": "tts_done"})
        finally:
            self.is_agent_speaking = False

    async def _handle_interrupt(self) -> None:
        """
        Stop speech immediately but keep reading state.

        This is the key contract requirement: interruption must not erase the
        resume point.
        """
        self.interrupt_event.set()
        self.is_agent_speaking = False

        if self.reading_task and not self.reading_task.done():
            self.reading_task.cancel()
            try:
                await self.reading_task
            except asyncio.CancelledError:
                log("interrupt", "reading task cancelled")

        await self.dc.send_json(
            {
                "type": "doc_reading_pause",
                "resume_from_sentence_idx": self.resume_from_sentence_idx,
            }
        )

    async def _answer_question(self, user_text: str) -> None:
        """
        Simulate the Q&A branch of the session.

        The production system calls STT, then LLM, then TTS. Here we skip STT and
        focus on how the answer is grounded in recent reading context.
        """
        if self.active_document_id is None:
            await self.dc.send_json({"type": "llm_error", "message": "No active document"})
            return

        document = self.store.get_document(self.active_document_id)
        if document is None:
            await self.dc.send_json({"type": "llm_error", "message": "Document not found"})
            return

        context = self._recent_context(document, self.last_read_sentence_idx, window=2)
        answer = (
            "Using recent document context: self-attention means each token can use "
            "information from other tokens in the same sequence."
        )

        await self.dc.send_json({"type": "llm_start", "user_text": user_text})
        await self.dc.send_json({"type": "llm_partial", "text": answer})
        await self.dc.send_json({"type": "llm_final", "text": answer, "context": context})
        await self.dc.send_json({"type": "tts_audio", "sentence_text": answer})
        await self.dc.send_json({"type": "tts_done"})

    def _recent_context(self, document: Document, last_idx: int, window: int) -> list[str]:
        if last_idx < 0:
            return document.sentences[:window]
        start = max(0, last_idx - window + 1)
        end = min(len(document.sentences), last_idx + 1)
        return document.sentences[start:end]


async def run_demo() -> None:
    """
    Demonstrate the full lifecycle:
      1. open session
      2. load document
      3. start reading
      4. interrupt
      5. ask question
      6. continue from the saved sentence
    """
    store = DemoDocumentStore()
    session = WebRTCSessionDemo(store)

    await session.start()
    await session.handle_message({"type": "doc_load", "doc_id": "doc_001"})
    await session.handle_message({"type": "doc_read", "doc_id": "doc_001"})

    # Let the reader speak for a moment before simulating barge-in.
    await asyncio.sleep(0.8)
    await session.handle_message({"type": "interrupt"})

    await session.handle_message(
        {
            "type": "user_question",
            "text": "What does self attention mean in this document?",
        }
    )

    await session.handle_message({"type": "continue_reading"})

    if session.reading_task is not None:
        try:
            await session.reading_task
        except asyncio.CancelledError:
            pass

    log(
        "summary",
        "final_state active_document_id={} last_read_sentence_idx={} resume_from_sentence_idx={}",
        session.active_document_id,
        session.last_read_sentence_idx,
        session.resume_from_sentence_idx,
    )


if __name__ == "__main__":
    setup_logging()
    asyncio.run(run_demo())
