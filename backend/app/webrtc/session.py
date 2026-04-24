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

from app.services.document_store import get_document_store
from app.services.llm import complete_llm_response
from app.services.search import web_search
from app.services.stt import get_stt_service
from app.services.tts import get_available_tts_voices, get_default_tts_voice, get_tts_service
from app.services.vad import StreamingVAD, get_vad_service
from app.utils.document_turns import (
    DocumentTurnDecision,
    build_document_turn_context,
    detect_direct_read_intent,
    parse_document_turn_response,
    resolve_document_by_name,
    user_explicitly_named_document,
)
from app.utils.emotion import clean_for_tts
from config.settings import get_settings

_PAUSE_PATTERN = re.compile(
    r"^\s*(wait|hold on|hold up|one moment|one sec(?:ond)?|just a (?:moment|second|sec)|"
    r"give me a (?:second|moment|sec)|hang on|please wait|just wait|ok wait|okay wait|"
    r"stop|stop it|stop please|please stop|ok stop|okay stop)\s*[.!?,]?\s*$",
    re.IGNORECASE,
)
_CONTINUE_READING_PATTERN = re.compile(
    r"^\s*("
    r"keep reading|continue reading|continue|resume reading|resume|"
    r"start reading from where (you|we) left( off)?|"
    r"(continue|carry on|go on) from where (you|we) (stopped|left|paused|were)|"
    r"pick up (where|from) (you|we) left( off)?|"
    r"go on|carry on"
    r")\s*[.!?,]?\s*$",
    re.IGNORECASE,
)
_READ_FROM_BEGINNING_PATTERN = re.compile(
    r"\b(read|start reading|read aloud|resume reading|continue reading)\b.*\b(beginning|start|top)\b|\bstart over\b|\brestart\b",
    re.IGNORECASE,
)

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

    def __init__(self, session_id: str, initial_voice: str | None = None) -> None:
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
        self._resume_from_sentence_idx: int | None = None
        self._last_answer_text: str = ""
        self._tts_voice: str = self._resolve_tts_voice(initial_voice)
        self._tts_speed: float = 1.0

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

    def _resolve_tts_voice(self, voice: str | None) -> str:
        valid_voices = {item["id"] for item in get_available_tts_voices()}
        if voice and voice in valid_voices:
            return voice
        return get_default_tts_voice()

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

        elif event_type == "tts_voice":
            voice = str(payload.get("voice", ""))
            resolved_voice = self._resolve_tts_voice(voice)
            self._tts_voice = resolved_voice
            logger.info("session_id={} event=tts_voice_selected voice={}", self.session_id, resolved_voice)

        elif event_type == "tts_speed":
            self._tts_speed = max(0.8, min(1.3, float(payload.get("speed", 1.0))))
            logger.info("session_id={} event=tts_speed_selected speed={}", self.session_id, self._tts_speed)

        elif event_type == "interrupt":
            logger.info("session_id={} event=interrupt_received", self.session_id)
            await self._handle_interrupt()

        elif event_type == "doc_load":
            doc_id = str(payload.get("doc_id", ""))
            doc = get_document_store().get_document(doc_id)
            if doc:
                self._active_document_id = doc_id
                self._last_read_sentence_idx = -1
                self._resume_from_sentence_idx = None
                annotations = get_document_store().load_annotations(doc_id)
                if rp := annotations.get("reading_position"):
                    saved_idx = rp.get("last_sentence_idx")
                    if isinstance(saved_idx, int) and saved_idx >= 0:
                        self._resume_from_sentence_idx = saved_idx
                        self._last_read_sentence_idx = max(-1, saved_idx - 1)
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
            self._last_read_sentence_idx = -1
            self._resume_from_sentence_idx = None

        elif event_type == "pause_reading":
            if self._llm_task and not self._llm_task.done():
                await self._handle_interrupt()
                await self._send_json({"type": "doc_reading_pause"})
                logger.info("session_id={} event=pause_reading_button", self.session_id)

        elif event_type == "continue_reading":
            # If session is fresh (e.g. new WebRTC connection after pause), restore doc state
            fallback_doc_id = str(payload.get("doc_id", ""))
            if not self._active_document_id and fallback_doc_id:
                doc = get_document_store().get_document(fallback_doc_id)
                if doc:
                    self._active_document_id = fallback_doc_id
                    annotations = get_document_store().load_annotations(fallback_doc_id)
                    if rp := annotations.get("reading_position"):
                        saved_idx = rp.get("last_sentence_idx")
                        if isinstance(saved_idx, int) and saved_idx >= 0:
                            self._resume_from_sentence_idx = saved_idx
                            self._last_read_sentence_idx = max(-1, saved_idx - 1)
                    logger.info(
                        "session_id={} event=continue_reading_restored doc_id={} resume_idx={}",
                        self.session_id, fallback_doc_id, self._resume_from_sentence_idx,
                    )

            if self._active_document_id and (self._llm_task is None or self._llm_task.done()):
                doc = get_document_store().get_document(self._active_document_id)
                if doc:
                    start_idx = self._get_read_start_idx(restart_from_beginning=False)
                    if start_idx < doc.sentence_count:
                        await self._send_json({"type": "doc_reading_resume"})
                        self._llm_task = asyncio.ensure_future(
                            self._run_document_read(
                                doc_id=doc.doc_id,
                                user_text="",
                                start_idx=start_idx,
                                llm_ms=0.0,
                            )
                        )
                    else:
                        await self._send_json({"type": "doc_reading_pause"})
                else:
                    await self._send_json({"type": "doc_reading_pause"})

        elif event_type == "doc_read":
            doc_id = str(payload.get("doc_id", ""))
            restart_from_beginning = bool(payload.get("restart_from_beginning"))
            doc = get_document_store().get_document(doc_id)
            if not doc:
                await self._send_json({"type": "doc_error", "message": f"Document '{doc_id}' not found."})
                return
            self._active_document_id = doc_id
            annotations = get_document_store().load_annotations(doc_id)
            self._resume_from_sentence_idx = None
            self._last_read_sentence_idx = -1
            if not restart_from_beginning and (rp := annotations.get("reading_position")):
                saved_idx = rp.get("last_sentence_idx")
                if isinstance(saved_idx, int) and saved_idx >= 0:
                    self._resume_from_sentence_idx = saved_idx
                    self._last_read_sentence_idx = max(-1, saved_idx - 1)
            self._llm_task = asyncio.ensure_future(
                self._run_document_read(
                    doc_id=doc.doc_id,
                    user_text="",
                    start_idx=self._get_read_start_idx(restart_from_beginning=restart_from_beginning),
                    llm_ms=0.0,
                )
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

    async def _tts_sentence_pipeline(
        self,
        queue: asyncio.Queue[tuple[str, int | None, str | None] | None],
        enable_barge_in: bool = True,
    ) -> None:
        """
        Consume sentences from *queue* and synthesise + stream each one immediately.

        Runs concurrently with ``_run_llm`` so the agent starts speaking after the
        first sentence boundary — not after the full LLM response completes.

        Args:
            queue: Unbounded asyncio queue carrying ``(tts_text, sentence_idx,
                   display_text)`` tuples or a ``None`` sentinel that signals end-of-stream.
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
            tts_text, sentence_idx, display_text = item
            try:
                wav_bytes, sr = await tts_service.synthesize(tts_text, voice=self._tts_voice, speed=self._tts_speed)
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
            if sentence_idx is not None:
                self._last_read_sentence_idx = sentence_idx
                self._resume_from_sentence_idx = sentence_idx
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

    async def _handle_doc_search(self, query: str) -> None:
        results = await web_search(query, max_results=self._settings.web_search_max_results)
        await self._send_json({"type": "doc_search_result", "query": query, "results": results})

    async def _play_tts_turn(
        self,
        *,
        user_text: str,
        display_text: str,
        utterances: list[tuple[str, int | None, str | None]],
        llm_ms: float,
        enable_barge_in: bool = True,
    ) -> None:
        await self._send_json({"type": "llm_start", "user_text": user_text})
        await self._send_json({"type": "llm_partial", "text": display_text})
        await self._send_json({"type": "llm_final", "text": display_text, "llm_ms": llm_ms})
        if not utterances:
            return

        sent_queue: asyncio.Queue[tuple[str, int | None, str | None] | None] = asyncio.Queue()
        for utterance in utterances:
            await sent_queue.put(utterance)
        await sent_queue.put(None)

        tts_task = asyncio.create_task(self._tts_sentence_pipeline(sent_queue, enable_barge_in=enable_barge_in))
        self._tts_task = tts_task
        with suppress(asyncio.CancelledError):
            await tts_task
        self._tts_task = None

    def _split_text_for_tts(self, text: str) -> list[tuple[str, int | None, str | None]]:
        utterances: list[tuple[str, int | None, str | None]] = []
        for raw in re.split(r"(?<=[.!?])\s+", text.strip()):
            sentence = clean_for_tts(raw.strip())
            if sentence:
                utterances.append((sentence, None, None))
        return utterances

    def _get_read_start_idx(self, *, restart_from_beginning: bool) -> int:
        if restart_from_beginning:
            return 0
        if self._resume_from_sentence_idx is not None:
            return max(0, self._resume_from_sentence_idx)
        if self._last_read_sentence_idx >= 0:
            return self._last_read_sentence_idx + 1
        return 0

    async def _run_document_read(
        self,
        *,
        doc_id: str,
        user_text: str,
        start_idx: int,
        llm_ms: float,
    ) -> str:
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
            self._resume_from_sentence_idx = None
            await self._send_json({"type": "doc_reading_pause"})
            return ""

        self._active_document_id = doc.doc_id
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

    async def _run_llm(self, text: str, trigger: str) -> None:
        self._llm_responded = False
        self._interrupt_event.clear()
        self._latest_llm_input = text

        display_response = ""
        llm_t0 = perf_counter()
        call_error: str | None = None

        try:
            direct_read = detect_direct_read_intent(text)
            if direct_read and self._active_document_id:
                if direct_read.restart_from_beginning:
                    self._last_read_sentence_idx = -1
                    self._resume_from_sentence_idx = None
                elif direct_read.action == "continue_reading" and self._resume_from_sentence_idx is None and self._last_read_sentence_idx < 0:
                    self._resume_from_sentence_idx = 0
                display_response = await self._run_document_read(
                    doc_id=self._active_document_id,
                    user_text=text,
                    start_idx=self._get_read_start_idx(restart_from_beginning=direct_read.restart_from_beginning),
                    llm_ms=0.0,
                )
                call_error = None
                return

            document_context = build_document_turn_context(
                user_text=text,
                active_document_id=self._active_document_id,
                last_read_sentence_idx=self._last_read_sentence_idx,
            )
            full_response = await complete_llm_response(
                text,
                conversation_history=list(self._conversation_history),
                document_context=document_context,
            )
            llm_ms = round((perf_counter() - llm_t0) * 1000, 2)
            decision = parse_document_turn_response(full_response) if document_context else None
            if self._active_document_id and _CONTINUE_READING_PATTERN.match(text):
                active_doc = get_document_store().get_document(self._active_document_id)
                if active_doc:
                    decision = DocumentTurnDecision(
                        action="continue_reading",
                        document_name=active_doc.title,
                        response_text="",
                        restart_from_beginning=False,
                    )
            elif self._active_document_id and _READ_FROM_BEGINNING_PATTERN.search(text):
                active_doc = get_document_store().get_document(self._active_document_id)
                if active_doc:
                    decision = DocumentTurnDecision(
                        action="read_document",
                        document_name=active_doc.title,
                        response_text="",
                        restart_from_beginning=True,
                    )

            if decision is not None:
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
                            if decision.restart_from_beginning:
                                self._last_read_sentence_idx = -1
                                self._resume_from_sentence_idx = None
                            elif decision.action == "continue_reading" and self._resume_from_sentence_idx is None and self._last_read_sentence_idx < 0:
                                self._resume_from_sentence_idx = 0
                            display_response = await self._run_document_read(
                                doc_id=target_doc.doc_id,
                                user_text=text,
                                start_idx=self._get_read_start_idx(restart_from_beginning=decision.restart_from_beginning),
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
                    if sentence_idx is None or sentence_idx < 0:
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
                    target_doc = resolve_document_by_name(
                        decision.document_name,
                        active_document_id=None,
                    )
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
                        # Spoken text — answer only, no URLs
                        snippets = [r.get("snippet", "") for r in results[:3] if r.get("snippet")]
                        spoken_text = ". ".join(snippets) if snippets else "Here is what I found online."
                        # Display text — answer + source citations (shown in chat, not read aloud)
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
            else:
                display_response = full_response.strip()
                await self._play_tts_turn(
                    user_text=text,
                    display_text=display_response,
                    utterances=self._split_text_for_tts(display_response),
                    llm_ms=llm_ms,
                )
        except Exception as err:
            call_error = str(err)
            logger.warning(
                "session_id={} event=llm_error error={}", self.session_id, err
            )
            await self._send_json(
                {"type": "llm_error", "message": "LLM unavailable — is Ollama running?"}
            )

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
        sent_queue: asyncio.Queue[tuple[str, int | None, str | None] | None] = asyncio.Queue()
        for raw in re.split(r"(?<=[.!?])\s+", welcome.strip()):
            sentence = clean_for_tts(raw.strip())
            if sentence:
                sent_queue.put_nowait((sentence, None, None))
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
