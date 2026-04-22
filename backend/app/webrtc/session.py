"""WebRTC session: RTP audio → STT → LLM → TTS pipeline over a data channel."""

import asyncio
import base64
import json
import re
import wave
from contextlib import suppress
from time import perf_counter

import av
import numpy as np
from aiortc import MediaStreamTrack, RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from loguru import logger

from app.prompts.system import DOCUMENT_READING_PROMPT
from app.services.document_store import get_document_store
from app.services.llm import stream_llm_response
from app.services.search import web_search
from app.services.stt import get_stt_service
from app.services.tts import get_tts_service
from app.services.vad import StreamingVAD, get_vad_service
from app.utils.action_tags import extract_doc_actions
from app.utils.emotion import clean_for_tts, strip_emotion_tags
from config.settings import get_settings

_PAUSE_PATTERN = re.compile(
    r"^\s*(wait|hold on|hold up|one moment|one sec(?:ond)?|just a (?:moment|second|sec)|"
    r"give me a (?:second|moment|sec)|hang on|please wait|just wait|ok wait|okay wait|"
    r"stop|stop it|stop please|please stop|ok stop|okay stop)\s*[.!?,]?\s*$",
    re.IGNORECASE,
)
_RESTART_FROM_BEGINNING_PATTERN = re.compile(
    r"\b(start|read|begin|restart)\b.*\b(beginning|start|top)\b|\bstart over\b|\brestart\b",
    re.IGNORECASE,
)
_DIRECT_READ_REQUEST_PATTERN = re.compile(
    r"\b(read|start reading|read aloud|continue reading|resume reading|keep reading)\b",
    re.IGNORECASE,
)
_SENT_BOUNDARY = re.compile(r"[.!?](?:\s|$)")
_MIN_SENTENCE_CHARS = 15

# Server-side VAD: energy threshold (normalised to [-1, 1]) to trigger barge-in.
# Slightly lower than the client-side 0.15 because there is no browser AGC on the
# raw Opus-decoded frames we see here.
_BARGE_IN_THRESHOLD = 0.15
_BARGE_IN_FRAMES = 3

_RTC_CONFIG = RTCConfiguration(
    iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
)


class WebRTCSession:
    """
    One WebRTC peer-connection per browser tab.

    Audio arrives as RTP (Opus → PCM via aiortc/av).
    All JSON signalling (ready / partial / llm_* / tts_*) travels over an
    ordered RTCDataChannel named "signaling" — same message schema as the
    WebSocket path so the frontend handler is reusable.

    Args:
        session_id: Short hex ID shared with the frontend as `request_id`.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._settings = get_settings()
        self.pc = RTCPeerConnection(_RTC_CONFIG)
        self.dc = None

        # Audio accumulation
        self._pcm_buffer = bytearray()
        self._chunk_count = 0
        self._last_emit_at = 0.0
        self._last_text_sent = ""
        self._sample_rate = 16_000  # resampled server-side; client value ignored

        # Concurrency helpers
        self._send_lock = asyncio.Lock()
        self._llm_task: asyncio.Task | None = None
        self._tts_task: asyncio.Task | None = None
        self._pending_llm_call: tuple[str, str] | None = None
        self._latest_llm_input = ""
        self._interrupt_event = asyncio.Event()
        self._silence_debounce_task: asyncio.Task | None = None
        self._speech_finalization_task: asyncio.Task | None = None

        # Conversation state
        self._conversation_history: list[dict[str, str]] = []
        self._llm_responded = False
        self._active_document_id: str | None = None
        self._last_read_sentence_idx: int = -1

        # Barge-in state
        self._is_agent_speaking = False
        self._barge_in_count = 0
        self._vad_stream: StreamingVAD | None = (
            get_vad_service().create_stream() if self._settings.stream_vad_enabled else None
        )

        # Background tasks
        self._audio_task: asyncio.Task | None = None
        self._closed = False


        self._register_pc_handlers()

    # ── Peer-connection lifecycle ────────────────────────────────────────────

    def _register_pc_handlers(self) -> None:
        @self.pc.on("track")
        def on_track(track: MediaStreamTrack) -> None:
            if track.kind == "audio":
                self._audio_task = asyncio.ensure_future(self._consume_audio(track))

        @self.pc.on("datachannel")
        def on_datachannel(channel) -> None:
            self.dc = channel

            @channel.on("open")
            def on_open() -> None:
                asyncio.ensure_future(self._on_dc_open())

            @channel.on("message")
            def on_message(message: str) -> None:
                asyncio.ensure_future(self._handle_dc_message(message))

        @self.pc.on("connectionstatechange")
        async def on_state_change() -> None:
            state = self.pc.connectionState
            logger.info(
                "session_id={} event=rtc_state_change state={}", self.session_id, state
            )
            if state in ("failed", "closed", "disconnected"):
                await self._cleanup()

    async def setup(self, offer_sdp: str, offer_type: str) -> RTCSessionDescription:
        """
        Complete SDP offer/answer exchange.

        Args:
            offer_sdp: SDP string from the browser offer.
            offer_type: SDP type, typically ``"offer"``.

        Returns:
            The local RTCSessionDescription answer to send back to the browser.
        """
        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=offer_sdp, type=offer_type)
        )
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        return self.pc.localDescription

    # ── Data-channel open / message ──────────────────────────────────────────

    async def _on_dc_open(self) -> None:
        logger.info("session_id={} event=dc_open", self.session_id)
        await self._send_json({"type": "ready", "request_id": self.session_id})
        asyncio.ensure_future(self._run_welcome())

    async def _handle_dc_message(self, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return

        event_type = payload.get("type")

        if event_type == "start":
            # sample_rate from client is informational only — we always resample to 16 kHz
            logger.info("session_id={} event=stream_started", self.session_id)

        elif event_type == "interrupt":
            logger.info("session_id={} event=interrupt_received", self.session_id)
            await self._handle_interrupt()

        elif event_type == "doc_load":
            doc_id = str(payload.get("doc_id", ""))
            doc = get_document_store().get_document(doc_id)
            if doc:
                self._active_document_id = doc_id
                annotations = get_document_store().load_annotations(doc_id)
                await self._send_json({
                    "type": "doc_opened",
                    "doc_id": doc_id,
                    "title": doc.title,
                    "raw_markdown": doc.raw_markdown,
                    "sentences": doc.sentences,
                    "annotations": annotations,
                })
                logger.info("session_id={} event=doc_loaded doc_id={}", self.session_id, doc_id)
            else:
                await self._send_json({"type": "doc_error", "message": f"Document '{doc_id}' not found."})

        elif event_type == "doc_unload":
            self._active_document_id = None

        elif event_type == "continue_reading":
            if self._active_document_id and (self._llm_task is None or self._llm_task.done()):
                next_idx = self._last_read_sentence_idx + 1
                doc = get_document_store().get_document(self._active_document_id)
                if doc and next_idx < doc.sentence_count:
                    self._llm_task = asyncio.ensure_future(
                        self._run_llm(f"continue reading from sentence {next_idx}", "auto_continue")
                    )

        elif event_type == "doc_save_highlight":
            doc_id = str(payload.get("doc_id", ""))
            sentence_idx = int(payload.get("sentence_idx", -1))
            if doc_id and sentence_idx >= 0:
                get_document_store().save_highlight(doc_id, sentence_idx)
                await self._send_json({"type": "doc_highlight_saved", "sentence_idx": sentence_idx})

        elif event_type == "stop":
            # Cancel silence debounce so it doesn't double-fire after the final transcript.
            if self._silence_debounce_task and not self._silence_debounce_task.done():
                self._silence_debounce_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._silence_debounce_task
                self._silence_debounce_task = None
            if self._speech_finalization_task and not self._speech_finalization_task.done():
                self._speech_finalization_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._speech_finalization_task
                self._speech_finalization_task = None

            if self._llm_task and not self._llm_task.done():
                logger.info(
                    "session_id={} event=stop_skipped_final_stt reason=llm_in_flight",
                    self.session_id,
                )
                return

            if self._pcm_buffer:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self._transcribe_buffer)
                await self._send_json({"type": "final", **result})
                final_text = str(result.get("text", "")).strip()
                if final_text and final_text != self._latest_llm_input:
                    self._schedule_llm(final_text, "final")

    # ── RTP audio consumer ────────────────────────────────────────────────────

    async def _consume_audio(self, track: MediaStreamTrack) -> None:
        """
        Continuously drain RTP frames from the peer track.

        Opus frames are decoded by aiortc and arrive as ``av.AudioFrame`` at 48 kHz.
        We resample to 16 kHz (matching Whisper's expected rate) and accumulate into
        ``self._pcm_buffer``.  Server-side VAD runs on every frame to detect barge-in
        while the agent is speaking.
        """
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16_000)
        logger.info("session_id={} event=audio_consumer_started", self.session_id)

        while not self._closed:
            try:
                frame = await asyncio.wait_for(track.recv(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            except (MediaStreamError, asyncio.CancelledError):
                break
            except Exception as exc:
                logger.warning(
                    "session_id={} event=audio_recv_error error={}", self.session_id, exc
                )
                break

            for resampled in resampler.resample(frame):
                pcm = resampled.to_ndarray().tobytes()
                self._pcm_buffer.extend(pcm)
                self._chunk_count += 1

                if self._vad_stream is not None:
                    for vad_event in self._vad_stream.process_pcm16(pcm):
                        if vad_event.event == "start":
                            logger.info(
                                "session_id={} event=vad_speech_start sample_index={} speech_prob={}",
                                self.session_id,
                                vad_event.sample_index,
                                round(vad_event.speech_prob, 4),
                            )
                            if self._silence_debounce_task and not self._silence_debounce_task.done():
                                self._silence_debounce_task.cancel()
                            self._silence_debounce_task = None
                            if self._is_agent_speaking:
                                logger.info(
                                    "session_id={} event=server_vad_barge_in sample_index={} speech_prob={}",
                                    self.session_id,
                                    vad_event.sample_index,
                                    round(vad_event.speech_prob, 4),
                                )
                                asyncio.ensure_future(self._handle_interrupt())
                        elif vad_event.event == "end":
                            logger.info(
                                "session_id={} event=vad_speech_end sample_index={} speech_prob={}",
                                self.session_id,
                                vad_event.sample_index,
                                round(vad_event.speech_prob, 4),
                            )
                            if not self._is_agent_speaking:
                                self._schedule_speech_finalization("vad_end")
                elif self._is_agent_speaking:
                    # Fallback RMS gate when the dedicated VAD is disabled.
                    samples = resampled.to_ndarray().astype(np.float32) / 32_768.0
                    rms = float(np.sqrt(np.mean(samples ** 2)))
                    if rms > _BARGE_IN_THRESHOLD:
                        self._barge_in_count += 1
                        if self._barge_in_count >= _BARGE_IN_FRAMES:
                            logger.info(
                                "session_id={} event=server_vad_barge_in rms={}",
                                self.session_id,
                                round(rms, 4),
                            )
                            asyncio.ensure_future(self._handle_interrupt())
                    else:
                        self._barge_in_count = 0

                await self._maybe_emit_stt()

        logger.info("session_id={} event=audio_consumer_stopped", self.session_id)

    async def _maybe_emit_stt(self) -> None:
        """Emit a partial STT result when buffer and time thresholds are met."""
        if not self._pcm_buffer:
            return
        if self._speech_finalization_task and not self._speech_finalization_task.done():
            self._last_emit_at = perf_counter()
            return
        if self._llm_task and not self._llm_task.done():
            self._last_emit_at = perf_counter()
            return
        buffered_ms = len(self._pcm_buffer) / 2 / self._sample_rate * 1000
        now = perf_counter()
        if not (
            buffered_ms >= self._settings.stream_min_audio_ms
            and (now - self._last_emit_at) * 1000 >= self._settings.stream_emit_interval_ms
        ):
            return

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._transcribe_buffer)
        if self._speech_finalization_task and not self._speech_finalization_task.done():
            self._last_emit_at = perf_counter()
            logger.info(
                "session_id={} event=partial_stt_skipped reason=turn_finalizing",
                self.session_id,
            )
            return
        if self._llm_task and not self._llm_task.done():
            self._last_emit_at = perf_counter()
            logger.info(
                "session_id={} event=partial_stt_skipped reason=llm_in_flight",
                self.session_id,
            )
            return
        current_text = str(result.get("text", ""))
        if current_text != self._last_text_sent:
            await self._send_json({"type": "partial", **result})
            self._last_text_sent = current_text

            if self._silence_debounce_task and not self._silence_debounce_task.done():
                self._silence_debounce_task.cancel()
            self._silence_debounce_task = asyncio.create_task(
                self._silence_debounce_then_fire(current_text, "debounced_partial")
            )
        self._last_emit_at = perf_counter()

    def _transcribe_buffer(self) -> dict:
        """
        Write ``self._pcm_buffer`` to a temp WAV and run faster-whisper.

        Returns:
            Dict with keys ``text``, ``timings_ms``, ``debug``.
        """
        started_at = perf_counter()
        temp_path = self._settings.temp_dir / f"{self.session_id}_rtc.wav"
        try:
            with wave.open(str(temp_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._sample_rate)
                wf.writeframes(bytes(self._pcm_buffer))

            service = get_stt_service()
            result = service.transcribe(
                file_path=temp_path,
                request_id=self.session_id,
                filename=f"rtc_{self.session_id}.wav",
                audio_bytes=len(self._pcm_buffer),
            )
            result.timings_ms.buffered_audio_ms = round(
                len(self._pcm_buffer) / 2 / self._sample_rate * 1000, 2
            )
            result.timings_ms.total_ms = round((perf_counter() - started_at) * 1000, 2)
            result.debug.sample_rate = self._sample_rate
            result.debug.chunks_received = self._chunk_count
            return {
                "text": result.text,
                "timings_ms": result.timings_ms.model_dump(),
                "debug": result.debug.model_dump(),
            }
        finally:
            temp_path.unlink(missing_ok=True)

    def _schedule_speech_finalization(self, trigger: str) -> None:
        if self._speech_finalization_task and not self._speech_finalization_task.done():
            return
        self._speech_finalization_task = asyncio.create_task(
            self._finalize_speech_turn(trigger)
        )

    async def _finalize_speech_turn(self, trigger: str) -> None:
        try:
            if not self._pcm_buffer:
                return
            if self._is_agent_speaking:
                logger.info(
                    "session_id={} event=speech_finalization_skipped reason=agent_speaking",
                    self.session_id,
                )
                return
            if self._llm_task and not self._llm_task.done():
                logger.info(
                    "session_id={} event=speech_finalization_skipped reason=llm_in_flight",
                    self.session_id,
                )
                return

            if self._silence_debounce_task and not self._silence_debounce_task.done():
                self._silence_debounce_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._silence_debounce_task
                self._silence_debounce_task = None

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._transcribe_buffer)

            if self._llm_task and not self._llm_task.done():
                logger.info(
                    "session_id={} event=speech_finalization_skipped reason=llm_started_during_transcribe",
                    self.session_id,
                )
                return

            final_text = str(result.get("text", "")).strip()
            if not final_text:
                logger.info(
                    "session_id={} event=speech_finalization_empty trigger={}",
                    self.session_id,
                    trigger,
                )
                return

            self._last_text_sent = final_text
            await self._send_json({"type": "final", **result})
            if final_text != self._latest_llm_input:
                self._schedule_llm(final_text, trigger)
        finally:
            self._speech_finalization_task = None

    # ── Interrupt ─────────────────────────────────────────────────────────────

    async def _handle_interrupt(self) -> None:
        self._interrupt_event.set()
        self._pending_llm_call = None
        self._is_agent_speaking = False
        self._barge_in_count = 0
        self._pcm_buffer.clear()
        self._last_text_sent = ""
        self._chunk_count = 0
        self._last_emit_at = 0.0
        if self._vad_stream is not None:
            self._vad_stream.reset()

        if self._silence_debounce_task and not self._silence_debounce_task.done():
            self._silence_debounce_task.cancel()
            self._silence_debounce_task = None

        if self._speech_finalization_task and not self._speech_finalization_task.done():
            self._speech_finalization_task.cancel()
            self._speech_finalization_task = None

        # Cancel TTS pipeline first so it doesn't send more audio after the new
        # _run_llm clears interrupt_event.
        if self._tts_task and not self._tts_task.done():
            self._tts_task.cancel()
            self._tts_task = None

        if self._llm_task and not self._llm_task.done():
            self._llm_task.cancel()
            self._llm_task = None

    # ── LLM scheduling ───────────────────────────────────────────────────────

    async def _silence_debounce_then_fire(self, text: str, trigger: str) -> None:
        try:
            await asyncio.sleep(self._settings.stream_llm_silence_ms / 1000)
        except asyncio.CancelledError:
            return
        if self._vad_stream is not None:
            self._schedule_speech_finalization(trigger)
            return
        self._schedule_llm(text, trigger)

    def _schedule_llm(self, text: str, trigger: str) -> None:
        normalized = text.strip()
        if len(normalized) < self._settings.stream_llm_min_chars:
            return
        if normalized == self._latest_llm_input:
            return
        if _PAUSE_PATTERN.match(normalized):
            return

        if self._llm_task and not self._llm_task.done():
            logger.info(
                "session_id={} event=llm_cancel_for_newer_text", self.session_id
            )
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

    # ── LLM + TTS pipeline ───────────────────────────────────────────────────

    async def _tts_sentence_pipeline(self, queue: asyncio.Queue, enable_barge_in: bool = True) -> None:
        """
        Consume sentences from *queue* and synthesise + stream each one immediately.

        Runs concurrently with ``_run_llm`` so the agent starts speaking after the
        first sentence boundary — not after the full LLM response completes.

        Args:
            queue: Unbounded asyncio queue carrying ``str`` sentences or a ``None``
                   sentinel that signals end-of-stream.
            enable_barge_in: When ``False``, server-side VAD is not activated — used
                             for the welcome message to prevent ambient noise from
                             cancelling synthesis before the user has spoken.
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
            sentence, sentence_idx = item
            try:
                wav_bytes, sr = await tts_service.synthesize(sentence)
            except Exception as err:
                logger.warning(
                    "session_id={} event=tts_error error={}", self.session_id, err
                )
                continue
            if self._interrupt_event.is_set():
                break
            if not tts_started:
                tts_started = True
                tts_t0 = perf_counter()
                await self._send_json({"type": "tts_start"})
            tts_ms = round((perf_counter() - tts_t0) * 1000, 2)
            wav_b64 = base64.b64encode(wav_bytes).decode()
            await self._send_json({
                "type": "tts_audio",
                "data": wav_b64,
                "sample_rate": sr,
                "tts_ms": tts_ms,
                "sentence_text": sentence,
                "sentence_idx": sentence_idx,
            })

        self._is_agent_speaking = False
        if self._interrupt_event.is_set():
            await self._send_json({"type": "tts_interrupted"})
        else:
            await self._send_json({"type": "tts_done"})

    async def _handle_doc_search(self, query: str) -> None:
        results = await web_search(query, max_results=self._settings.web_search_max_results)
        await self._send_json({"type": "doc_search_result", "query": query, "results": results})

    async def _run_llm(self, text: str, trigger: str) -> None:
        self._llm_responded = False
        self._interrupt_event.clear()
        self._latest_llm_input = text

        full_response = ""
        display_response = ""
        processed_chars = 0
        llm_t0 = perf_counter()
        call_error: str | None = None
        doc_actions_seen: set[str] = set()
        pending_tts_sentence_idxs: list[int | None] = []
        reading_turn = False
        rendered_reading_segments: list[str] = []
        restart_from_beginning = bool(_RESTART_FROM_BEGINNING_PATTERN.search(text))

        if restart_from_beginning:
            self._last_read_sentence_idx = -1

        # Build document context when a document is loaded
        document_context: str | None = None
        if self._active_document_id:
            doc = get_document_store().get_document(self._active_document_id)
            if doc:
                docs_list = get_document_store().list_documents()
                doc_names = ", ".join(f"'{d['title']}' (id: {d['doc_id']})" for d in docs_list)
                position_hint = (
                    f"Reading position: last read sentence {self._last_read_sentence_idx}. "
                    f"To continue, start from sentence {self._last_read_sentence_idx + 1}.\n"
                    if self._last_read_sentence_idx >= 0 else ""
                )
                restart_hint = (
                    "The user explicitly asked to start from the beginning. Start from sentence 0 and ignore any saved reading position.\n"
                    if restart_from_beginning else ""
                )
                document_context = (
                    f"{DOCUMENT_READING_PROMPT}\n\n"
                    f"Available documents: {doc_names}\n"
                    f"Currently loaded: '{doc.title}' (id: {doc.doc_id})\n"
                    f"Total sentences: {doc.sentence_count}\n"
                    f"{restart_hint}{position_hint}\n"
                    f"Document content (sentences numbered for action tags):\n"
                    + "\n".join(f"[{i}] {s}" for i, s in enumerate(doc.sentences))
                )
        elif get_document_store().list_documents():
            docs_list = get_document_store().list_documents()
            doc_names = ", ".join(f"'{d['title']}' (id: {d['doc_id']})" for d in docs_list)
            document_context = (
                f"{DOCUMENT_READING_PROMPT}\n\n"
                f"Available documents: {doc_names}\n"
                f"No document currently loaded. Ask the user which one they want to explore."
            )

        sent_queue: asyncio.Queue[tuple[str, int | None] | None] = asyncio.Queue()
        tts_task = asyncio.create_task(self._tts_sentence_pipeline(sent_queue))
        self._tts_task = tts_task

        direct_read_requested = bool(
            self._active_document_id
            and _DIRECT_READ_REQUEST_PATTERN.search(text)
            and ("?" not in text or "read" in text.lower())
        )

        if direct_read_requested:
            doc = get_document_store().get_document(self._active_document_id)
            if doc:
                start_idx = 0 if restart_from_beginning or self._last_read_sentence_idx < 0 else self._last_read_sentence_idx + 1
                end_idx = min(start_idx + 6, len(doc.sentences))
                selected_sentences = [
                    (idx, doc.sentences[idx])
                    for idx in range(start_idx, end_idx)
                    if 0 <= idx < len(doc.sentences)
                ]
                await self._send_json({"type": "llm_start", "user_text": text})
                await self._send_json({
                    "type": "doc_read_start",
                    "doc_id": doc.doc_id,
                    "sentences": doc.sentences,
                    "title": doc.title,
                })

                rendered_text = "\n".join(sentence for _, sentence in selected_sentences)
                for sentence_idx, sentence in selected_sentences:
                    self._last_read_sentence_idx = sentence_idx
                    await self._send_json({
                        "type": "doc_highlight",
                        "sentence_idx": sentence_idx,
                        "word_count": len(sentence.split()),
                    })
                    await sent_queue.put((clean_for_tts(sentence), sentence_idx))

                await self._send_json({"type": "llm_partial", "text": rendered_text})
                await self._send_json({"type": "llm_final", "text": rendered_text, "llm_ms": 0.0})
                await sent_queue.put(None)
                await tts_task
                return

        try:
            await self._send_json({"type": "llm_start", "user_text": text})
            async for token in stream_llm_response(
                text,
                conversation_history=list(self._conversation_history),
                document_context=document_context,
            ):
                if self._interrupt_event.is_set():
                    break
                full_response += token

                cleaned, actions = extract_doc_actions(full_response)
                display_response = strip_emotion_tags(cleaned)

                for action in actions:
                    sig = f"{action.kind}:{':'.join(action.params)}"
                    if sig in doc_actions_seen:
                        continue
                    doc_actions_seen.add(sig)
                    if action.kind == "list_docs":
                        docs = get_document_store().list_documents()
                        await self._send_json({"type": "doc_list", "documents": docs})
                    elif action.kind == "read" and action.params:
                        reading_turn = True
                        doc_id = action.params[0]
                        doc = get_document_store().get_document(doc_id)
                        if doc:
                            self._active_document_id = doc_id
                            if restart_from_beginning:
                                self._last_read_sentence_idx = -1
                            await self._send_json({
                                "type": "doc_read_start",
                                "doc_id": doc_id,
                                "sentences": doc.sentences,
                                "title": doc.title,
                            })
                    elif action.kind == "highlight" and len(action.params) >= 2:
                        try:
                            reading_turn = True
                            sent_idx = int(action.params[0])
                            self._last_read_sentence_idx = sent_idx
                            pending_tts_sentence_idxs.append(sent_idx)
                            doc = get_document_store().get_document(self._active_document_id) if self._active_document_id else None
                            sentence_text = (
                                doc.sentences[sent_idx]
                                if doc and 0 <= sent_idx < len(doc.sentences)
                                else None
                            )
                            if sentence_text:
                                rendered_reading_segments.append(sentence_text)
                                await sent_queue.put((clean_for_tts(sentence_text), sent_idx))
                            await self._send_json({
                                "type": "doc_highlight",
                                "sentence_idx": sent_idx,
                                "word_count": int(action.params[1]),
                            })
                        except (ValueError, IndexError):
                            pass
                    elif action.kind == "search" and action.params:
                        query = ":".join(action.params)
                        await self._send_json({"type": "doc_search_start", "query": query})
                        asyncio.ensure_future(self._handle_doc_search(query))
                    elif action.kind == "save_snippet" and action.params:
                        await self._send_json({"type": "doc_save_snippet", "term": ":".join(action.params)})
                    elif action.kind == "export" and action.params:
                        fmt = action.params[0]
                        if self._active_document_id and fmt in ("pdf", "docx"):
                            await self._send_json({
                                "type": "doc_export",
                                "format": fmt,
                                "download_url": f"/documents/{self._active_document_id}/export/{fmt}",
                            })
                    elif action.kind == "reading_pause":
                        await self._send_json({"type": "doc_reading_pause"})
                    elif action.kind == "reading_resume":
                        reading_turn = True
                        await self._send_json({"type": "doc_reading_resume"})

                transcript_text = "\n".join(rendered_reading_segments) if reading_turn and rendered_reading_segments else display_response
                await self._send_json(
                    {"type": "llm_partial", "text": transcript_text}
                )
                if not reading_turn:
                    tail = display_response[processed_chars:]
                    while True:
                        m = _SENT_BOUNDARY.search(tail)
                        if not m or m.end() < _MIN_SENTENCE_CHARS:
                            break
                        sentence = clean_for_tts(tail[: m.end()].strip())
                        if sentence:
                            sentence_idx = pending_tts_sentence_idxs.pop(0) if pending_tts_sentence_idxs else None
                            await sent_queue.put((sentence, sentence_idx))
                        processed_chars += m.end()
                        tail = display_response[processed_chars:]

            llm_ms = round((perf_counter() - llm_t0) * 1000, 2)
            final_text = "\n".join(rendered_reading_segments) if reading_turn and rendered_reading_segments else display_response
            await self._send_json(
                {"type": "llm_final", "text": final_text, "llm_ms": llm_ms}
            )
        except Exception as err:
            call_error = str(err)
            logger.warning(
                "session_id={} event=llm_error error={}", self.session_id, err
            )
            await self._send_json(
                {"type": "llm_error", "message": "LLM unavailable — is Ollama running?"}
            )
        finally:
            if not self._interrupt_event.is_set() and call_error is None:
                remaining = clean_for_tts(display_response[processed_chars:].strip())
                if remaining:
                    sent_queue.put_nowait(remaining)
            sent_queue.put_nowait(None)

        with suppress(asyncio.CancelledError):
            await tts_task
        self._tts_task = None

        # Clear audio buffer after the full turn (LLM + TTS) completes so any speech
        # the user started during the agent's response isn't fed into the next query.
        if not self._interrupt_event.is_set():
            self._pcm_buffer.clear()
            self._last_text_sent = ""
            self._chunk_count = 0
            self._last_emit_at = 0.0
            if self._vad_stream is not None:
                self._vad_stream.reset()

        if display_response and call_error is None and not self._interrupt_event.is_set():
            self._llm_responded = True
            self._conversation_history.append({"role": "user", "content": text})
            self._conversation_history.append({"role": "assistant", "content": display_response})
            max_msgs = self._settings.llm_max_history_turns * 2
            if len(self._conversation_history) > max_msgs:
                self._conversation_history[:] = self._conversation_history[-max_msgs:]

        if self._pending_llm_call:
            next_text, next_trigger = self._pending_llm_call
            self._pending_llm_call = None
            if next_text != self._latest_llm_input:
                self._llm_task = asyncio.create_task(
                    self._run_llm(next_text, next_trigger)
                )
                return
        self._llm_task = None

    # ── Welcome message ──────────────────────────────────────────────────────

    async def _run_welcome(self) -> None:
        welcome = self._settings.welcome_message
        if not welcome:
            return
        await self._send_json({"type": "llm_start", "user_text": ""})
        await self._send_json({"type": "llm_final", "text": welcome, "llm_ms": 0})

        # Split on sentence-ending punctuation so each sentence gets its own tts_audio
        # message with sentence_text — matching the regular pipeline's synced text reveal.
        sent_queue: asyncio.Queue[str | None] = asyncio.Queue()
        for raw in re.split(r"(?<=[.!?])\s+", welcome.strip()):
            sentence = clean_for_tts(raw.strip())
            if sentence:
                sent_queue.put_nowait(sentence)
        sent_queue.put_nowait(None)

        await self._tts_sentence_pipeline(sent_queue, enable_barge_in=False)

    # ── Helpers ──────────────────────────────────────────────────────────────

    async def _send_json(self, payload: dict) -> None:
        """
        Send a JSON message via the data channel.

        Args:
            payload: Dict to serialise and send.
        """
        async with self._send_lock:
            if self.dc and self.dc.readyState == "open":
                try:
                    self.dc.send(json.dumps(payload))
                except Exception as exc:
                    logger.debug(
                        "session_id={} event=dc_send_error error={}", self.session_id, exc
                    )

    async def _cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        for task in (
            self._audio_task,
            self._silence_debounce_task,
            self._speech_finalization_task,
            self._tts_task,
            self._llm_task,
        ):
            if task and not task.done():
                task.cancel()
        logger.info("session_id={} event=rtc_session_closed", self.session_id)
