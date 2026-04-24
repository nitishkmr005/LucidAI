"""WebRTC session: RTP audio → STT → AgentPipeline over a data channel."""

import asyncio
import json
import wave
from contextlib import suppress
from time import perf_counter

import av
import numpy as np
from aiortc import MediaStreamTrack, RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from loguru import logger

from app.services.document_store import get_document_store
from app.services.pipeline import AgentPipeline
from app.services.stt import get_stt_service
from app.services.vad import StreamingVAD, get_vad_service
from config.settings import get_settings

_BARGE_IN_THRESHOLD = 0.15
_BARGE_IN_FRAMES = 3

_RTC_CONFIG = RTCConfiguration(
    iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
)


class WebRTCSession:
    """One WebRTC peer-connection per browser tab: RTP audio → VAD → STT → AgentPipeline."""

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
        self._sample_rate = 16_000

        # Concurrency helpers
        self._send_lock = asyncio.Lock()
        self._silence_debounce_task: asyncio.Task | None = None
        self._speech_finalization_task: asyncio.Task | None = None

        # Barge-in state
        self._barge_in_count = 0
        self._vad_stream: StreamingVAD | None = (
            get_vad_service().create_stream() if self._settings.stream_vad_enabled else None
        )

        # Background tasks
        self._audio_task: asyncio.Task | None = None
        self._closed = False

        # Agent pipeline — owns all LLM/TTS/conversation state
        self._pipeline = AgentPipeline(
            session_id=session_id,
            send_json_fn=self._send_json,
            initial_voice=initial_voice,
            on_turn_complete=self._clear_audio_buffer,
        )

        self._register_pc_handlers()

    def _clear_audio_buffer(self) -> None:
        """Resets PCM buffer and VAD after a turn completes so stale audio never re-triggers LLM."""
        self._pcm_buffer.clear()
        self._last_text_sent = ""
        self._chunk_count = 0
        self._last_emit_at = 0.0
        if self._vad_stream is not None:
            self._vad_stream.reset()

    # ── Peer-connection lifecycle ────────────────────────────────────────────

    def _register_pc_handlers(self) -> None:
        """Attaches aiortc callbacks for audio track, data channel, and connection state."""
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
        """Complete the SDP offer/answer exchange and return the local answer.

        Args:
            offer_sdp: SDP string from the browser's RTCPeerConnection offer.
            offer_type: SDP type, typically ``"offer"``.

        Returns:
            Local RTCSessionDescription to send back to the browser.
        """
        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=offer_sdp, type=offer_type)
        )
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        return self.pc.localDescription

    # ── Data-channel open / message ──────────────────────────────────────────

    async def _on_dc_open(self) -> None:
        """Sends the ready signal and starts the spoken welcome message."""
        logger.info("session_id={} event=dc_open", self.session_id)
        await self._send_json({"type": "ready", "request_id": self.session_id})
        asyncio.ensure_future(self._pipeline.run_welcome())

    async def _handle_dc_message(self, raw: str) -> None:
        """Route each frontend data-channel event to pipeline state updates or interrupt.

        Args:
            raw: Raw JSON string received over the data channel.
        """
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return

        event_type = payload.get("type")

        if event_type == "start":
            logger.info("session_id={} event=stream_started", self.session_id)

        elif event_type == "tts_voice":
            self._pipeline.tts_voice = str(payload.get("voice", ""))
            logger.info(
                "session_id={} event=tts_voice_selected voice={}", self.session_id, self._pipeline.tts_voice
            )

        elif event_type == "tts_speed":
            self._pipeline.tts_speed = float(payload.get("speed", 1.0))
            logger.info(
                "session_id={} event=tts_speed_selected speed={}", self.session_id, self._pipeline.tts_speed
            )

        elif event_type == "interrupt":
            logger.info("session_id={} event=interrupt_received", self.session_id)
            await self._handle_interrupt()

        elif event_type == "doc_load":
            doc_id = str(payload.get("doc_id", ""))
            doc = get_document_store().get_document(doc_id)
            if doc:
                self._pipeline.active_document_id = doc_id
                self._pipeline.last_read_sentence_idx = -1
                self._pipeline.resume_from_sentence_idx = None
                annotations = get_document_store().load_annotations(doc_id)
                if rp := annotations.get("reading_position"):
                    saved_idx = rp.get("last_sentence_idx")
                    if isinstance(saved_idx, int) and saved_idx >= 0:
                        self._pipeline.resume_from_sentence_idx = saved_idx
                        self._pipeline.last_read_sentence_idx = max(-1, saved_idx - 1)
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
            self._pipeline.active_document_id = None
            self._pipeline.last_read_sentence_idx = -1
            self._pipeline.resume_from_sentence_idx = None

        elif event_type == "pause_reading":
            if self._pipeline.llm_task and not self._pipeline.llm_task.done():
                await self._handle_interrupt()
                await self._send_json({"type": "doc_reading_pause"})
                logger.info("session_id={} event=pause_reading_button", self.session_id)

        elif event_type == "continue_reading":
            fallback_doc_id = str(payload.get("doc_id", ""))
            if not self._pipeline.active_document_id and fallback_doc_id:
                # Restore document state when resuming after a WebRTC reconnect.
                doc = get_document_store().get_document(fallback_doc_id)
                if doc:
                    self._pipeline.active_document_id = fallback_doc_id
                    annotations = get_document_store().load_annotations(fallback_doc_id)
                    if rp := annotations.get("reading_position"):
                        saved_idx = rp.get("last_sentence_idx")
                        if isinstance(saved_idx, int) and saved_idx >= 0:
                            self._pipeline.resume_from_sentence_idx = saved_idx
                            self._pipeline.last_read_sentence_idx = max(-1, saved_idx - 1)
                    logger.info(
                        "session_id={} event=continue_reading_restored doc_id={} resume_idx={}",
                        self.session_id, fallback_doc_id, self._pipeline.resume_from_sentence_idx,
                    )

            if self._pipeline.active_document_id and (
                self._pipeline.llm_task is None or self._pipeline.llm_task.done()
            ):
                doc = get_document_store().get_document(self._pipeline.active_document_id)
                if doc:
                    start_idx = self._pipeline.get_read_start_idx(restart_from_beginning=False)
                    if start_idx < doc.sentence_count:
                        await self._send_json({"type": "doc_reading_resume"})
                        self._pipeline.schedule_document_read(doc.doc_id, start_idx)
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
            self._pipeline.active_document_id = doc_id
            self._pipeline.resume_from_sentence_idx = None
            self._pipeline.last_read_sentence_idx = -1
            annotations = get_document_store().load_annotations(doc_id)
            if not restart_from_beginning and (rp := annotations.get("reading_position")):
                saved_idx = rp.get("last_sentence_idx")
                if isinstance(saved_idx, int) and saved_idx >= 0:
                    self._pipeline.resume_from_sentence_idx = saved_idx
                    self._pipeline.last_read_sentence_idx = max(-1, saved_idx - 1)
            start_idx = self._pipeline.get_read_start_idx(restart_from_beginning=restart_from_beginning)
            self._pipeline.schedule_document_read(doc.doc_id, start_idx)

        elif event_type == "doc_save_highlight":
            doc_id = str(payload.get("doc_id", ""))
            sentence_idx = int(payload.get("sentence_idx", -1))
            if doc_id and sentence_idx >= 0:
                get_document_store().save_highlight(doc_id, sentence_idx)
                await self._send_json({"type": "doc_highlight_saved", "sentence_idx": sentence_idx})

        elif event_type == "stop":
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

            if self._pipeline.llm_task and not self._pipeline.llm_task.done():
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
                if final_text and final_text != self._pipeline.latest_llm_input:
                    self._pipeline.schedule_llm(final_text, "final")

    # ── RTP audio consumer ────────────────────────────────────────────────────

    async def _consume_audio(self, track: MediaStreamTrack) -> None:
        """Drain RTP frames, resample to 16 kHz mono PCM, run VAD/barge-in, and feed partial STT.

        Args:
            track: Incoming audio MediaStreamTrack from the WebRTC peer connection.
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
                                self.session_id, vad_event.sample_index, round(vad_event.speech_prob, 4),
                            )
                            if self._silence_debounce_task and not self._silence_debounce_task.done():
                                self._silence_debounce_task.cancel()
                            self._silence_debounce_task = None
                            if self._pipeline.is_agent_speaking:
                                logger.info(
                                    "session_id={} event=server_vad_barge_in sample_index={} speech_prob={}",
                                    self.session_id, vad_event.sample_index, round(vad_event.speech_prob, 4),
                                )
                                asyncio.ensure_future(self._handle_interrupt())
                        elif vad_event.event == "end":
                            logger.info(
                                "session_id={} event=vad_speech_end sample_index={} speech_prob={}",
                                self.session_id, vad_event.sample_index, round(vad_event.speech_prob, 4),
                            )
                            if not self._pipeline.is_agent_speaking:
                                self._schedule_speech_finalization("vad_end")
                elif self._pipeline.is_agent_speaking:
                    # Fallback RMS gate when dedicated VAD is disabled.
                    samples = resampled.to_ndarray().astype(np.float32) / 32_768.0
                    rms = float(np.sqrt(np.mean(samples ** 2)))
                    if rms > _BARGE_IN_THRESHOLD:
                        self._barge_in_count += 1
                        if self._barge_in_count >= _BARGE_IN_FRAMES:
                            logger.info(
                                "session_id={} event=server_vad_barge_in rms={}",
                                self.session_id, round(rms, 4),
                            )
                            asyncio.ensure_future(self._handle_interrupt())
                    else:
                        self._barge_in_count = 0

                await self._maybe_emit_stt()

        logger.info("session_id={} event=audio_consumer_stopped", self.session_id)

    async def _maybe_emit_stt(self) -> None:
        """Throttled partial STT emitter — skips when finalization or LLM is already in-flight."""
        if not self._pcm_buffer:
            return
        if self._speech_finalization_task and not self._speech_finalization_task.done():
            self._last_emit_at = perf_counter()
            return
        if self._pipeline.llm_task and not self._pipeline.llm_task.done():
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
                "session_id={} event=partial_stt_skipped reason=turn_finalizing", self.session_id
            )
            return
        if self._pipeline.llm_task and not self._pipeline.llm_task.done():
            self._last_emit_at = perf_counter()
            logger.info(
                "session_id={} event=partial_stt_skipped reason=llm_in_flight", self.session_id
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
        """Write the PCM buffer to a temp WAV and run STT; safe to call in a thread executor.

        Returns:
            Dict with keys ``text``, ``timings_ms``, and ``debug``.
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
        """Guard against duplicate finalization tasks — only one runs at a time.

        Args:
            trigger: Label forwarded to ``_finalize_speech_turn`` for logging.
        """
        if self._speech_finalization_task and not self._speech_finalization_task.done():
            return
        self._speech_finalization_task = asyncio.create_task(
            self._finalize_speech_turn(trigger)
        )

    async def _finalize_speech_turn(self, trigger: str) -> None:
        """Transcribe the completed utterance and fire the LLM — skips if agent is still speaking.

        Args:
            trigger: Label used in ``schedule_llm`` for session logging.
        """
        try:
            if not self._pcm_buffer:
                return
            if self._pipeline.is_agent_speaking:
                logger.info(
                    "session_id={} event=speech_finalization_skipped reason=agent_speaking",
                    self.session_id,
                )
                return
            if self._pipeline.llm_task and not self._pipeline.llm_task.done():
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

            if self._pipeline.llm_task and not self._pipeline.llm_task.done():
                logger.info(
                    "session_id={} event=speech_finalization_skipped reason=llm_started_during_transcribe",
                    self.session_id,
                )
                return

            final_text = str(result.get("text", "")).strip()
            if not final_text:
                logger.info(
                    "session_id={} event=speech_finalization_empty trigger={}",
                    self.session_id, trigger,
                )
                return

            self._last_text_sent = final_text
            await self._send_json({"type": "final", **result})
            if final_text != self._pipeline.latest_llm_input:
                self._pipeline.schedule_llm(final_text, trigger)
        finally:
            self._speech_finalization_task = None

    # ── Interrupt ─────────────────────────────────────────────────────────────

    async def _handle_interrupt(self) -> None:
        """Clears audio/VAD state and cancels pipeline tasks; preserves document reading position."""
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

        await self._pipeline.interrupt()

    # ── Silence debounce ──────────────────────────────────────────────────────

    async def _silence_debounce_then_fire(self, text: str, trigger: str) -> None:
        """Fire LLM (or speech finalization when VAD is on) after the configured silence gap.

        Args:
            text: Transcript text to pass to ``schedule_llm`` when VAD is disabled.
            trigger: Label for session logging.
        """
        try:
            await asyncio.sleep(self._settings.stream_llm_silence_ms / 1000)
        except asyncio.CancelledError:
            return
        if self._vad_stream is not None:
            self._schedule_speech_finalization(trigger)
            return
        self._pipeline.schedule_llm(text, trigger)

    # ── Helpers ──────────────────────────────────────────────────────────────

    async def _send_json(self, payload: dict) -> None:
        """Send JSON over the data channel; silently drops the message if the channel is closed.

        Args:
            payload: JSON-serialisable dict to send to the browser.
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
        """Idempotent teardown — cancels background tasks and interrupts the pipeline."""
        if self._closed:
            return
        self._closed = True
        for task in (
            self._audio_task,
            self._silence_debounce_task,
            self._speech_finalization_task,
        ):
            if task and not task.done():
                task.cancel()
        await self._pipeline.interrupt()
        logger.info("session_id={} event=rtc_session_closed", self.session_id)
