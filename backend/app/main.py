import asyncio
from contextlib import suppress
import json
import wave
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from app.models import HealthResponse, TranscriptionResponse
from app.routers.documents import router as documents_router
from app.services.document_store import get_document_store
from app.services.pipeline import AgentPipeline
from app.services.smart_turn import get_smart_turn_detector
from app.services.stt import get_stt_service
from app.services.tts import get_available_tts_voices, get_default_tts_voice, get_tts_service
from app.services.vad import get_vad_service
from app.utils.reading_patterns import PAUSE_PATTERN
from app.utils.session_logger import LLMCallLog, STTRunLog, SessionLog, TTSCallLog, _iso, write_session_log
from app.webrtc.router import router as webrtc_router
from config.logging import setup_logging
from config.settings import get_settings

setup_logging()
settings = get_settings()

_TTS_PREVIEW_TEXT = "Welcome to NeuroTalk. This is the selected reading voice."
_TTS_PREVIEW_MAX_CHARS = 160


class TTSPreviewRequest(BaseModel):
    voice: str | None = None
    text: str | None = None
    speed: float | None = None


app = FastAPI(title=settings.app_name, version="0.1.0")
app.include_router(webrtc_router)
app.include_router(documents_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _warmup_models() -> None:
    """Pre-load STT, VAD, and TTS models in the background on startup.

    Logs warmup latency for each model. Failures are logged as warnings so
    the application still starts if a model is unavailable.
    """
    loop = asyncio.get_event_loop()

    stt_t0 = perf_counter()
    try:
        await loop.run_in_executor(None, get_stt_service()._load_model)
        logger.info("event=stt_warmup_done ms={}", round((perf_counter() - stt_t0) * 1000))
    except Exception as err:
        logger.warning("event=stt_warmup_failed error={}", err)

    if settings.stream_vad_enabled:
        vad_t0 = perf_counter()
        try:
            await loop.run_in_executor(None, get_vad_service()._load_model)
            logger.info("event=vad_warmup_done ms={}", round((perf_counter() - vad_t0) * 1000))
        except Exception as err:
            logger.warning("event=vad_warmup_failed error={}", err)

    tts_t0 = perf_counter()
    try:
        await get_tts_service().synthesize("Hello.")
        logger.info("event=tts_warmup_done ms={}", round((perf_counter() - tts_t0) * 1000))
    except Exception as err:
        logger.warning("event=tts_warmup_failed error={}", err)

    if settings.stream_smart_turn_enabled:
        st_t0 = perf_counter()
        try:
            await loop.run_in_executor(None, get_smart_turn_detector().warm_up)
            logger.info("event=smart_turn_warmup_done ms={}", round((perf_counter() - st_t0) * 1000))
        except Exception as err:
            logger.warning("event=smart_turn_warmup_failed error={}", err)


@app.on_event("startup")
async def startup_event() -> None:
    """Create required directories and kick off background model warmup."""
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    settings.annotations_dir.mkdir(parents=True, exist_ok=True)
    settings.exports_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "event=startup app_name={} host={} port={} temp_dir={} cors_origins={}",
        settings.app_name,
        settings.app_host,
        settings.app_port,
        settings.temp_dir,
        settings.cors_origins,
    )
    asyncio.create_task(_warmup_models())


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return a 200 OK health check response.

    Returns:
        HealthResponse with default alive status.
    """
    return HealthResponse()


@app.get("/tts/voices")
async def list_tts_voices() -> dict[str, object]:
    """Return available TTS voice IDs and the current default voice.

    Returns:
        Dict with ``default_voice`` and ``voices`` list.
    """
    return {
        "default_voice": get_default_tts_voice(),
        "voices": get_available_tts_voices(),
    }


@app.post("/tts/preview")
async def preview_tts_voice(body: TTSPreviewRequest) -> Response:
    """Synthesise a short preview clip for the requested TTS voice.

    Args:
        body: Voice ID, optional preview text, and optional playback speed.

    Returns:
        WAV audio bytes for the preview clip.
    """
    valid_voices = {item["id"] for item in get_available_tts_voices()}
    voice = body.voice if body.voice in valid_voices else get_default_tts_voice()
    preview_text = (body.text or _TTS_PREVIEW_TEXT).strip()[:_TTS_PREVIEW_MAX_CHARS]
    if not preview_text:
        preview_text = _TTS_PREVIEW_TEXT
    wav_bytes, _sample_rate = await get_tts_service().synthesize(preview_text, voice=voice, speed=body.speed)
    return Response(content=wav_bytes, media_type="audio/wav")


def _write_pcm16_wav(*, pcm_bytes: bytes, sample_rate: int, file_path: Path) -> float:
    """Write raw PCM-16 mono samples to a WAV file.

    Args:
        pcm_bytes: Raw 16-bit little-endian PCM samples.
        sample_rate: Sample rate in Hz.
        file_path: Destination path for the WAV file.

    Returns:
        Time taken to write the file in milliseconds.
    """
    started_at = perf_counter()
    with wave.open(str(file_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return round((perf_counter() - started_at) * 1000, 2)


def _transcribe_stream_buffer(
    *,
    request_id: str,
    sample_rate: int,
    pcm_buffer: bytearray,
    chunk_count: int,
) -> dict[str, object]:
    """Transcribe the accumulated PCM buffer and return a structured result payload.

    Writes the buffer to a temporary WAV file, runs STT, then deletes the file.

    Args:
        request_id: Session request ID used to name the temporary WAV file.
        sample_rate: Sample rate of the PCM data in Hz.
        pcm_buffer: Accumulated raw PCM-16 bytes to transcribe.
        chunk_count: Number of WebSocket chunks in the buffer (for debug info).

    Returns:
        Dict with keys ``text``, ``timings_ms``, and ``debug``.
    """
    started_at = perf_counter()
    temp_path = settings.temp_dir / f"{request_id}_stream.wav"
    try:
        file_write_ms = _write_pcm16_wav(
            pcm_bytes=bytes(pcm_buffer), sample_rate=sample_rate, file_path=temp_path
        )
        service = get_stt_service()
        result = service.transcribe(
            file_path=temp_path,
            request_id=request_id,
            filename=f"stream_{request_id}.wav",
            audio_bytes=len(pcm_buffer),
        )
        result.timings_ms.file_write_ms = file_write_ms
        result.timings_ms.total_ms = round((perf_counter() - started_at) * 1000, 2)
        result.timings_ms.buffered_audio_ms = round(len(pcm_buffer) / 2 / sample_rate * 1000, 2)
        result.debug.sample_rate = sample_rate
        result.debug.chunks_received = chunk_count
        return {
            "text": result.text,
            "timings_ms": result.timings_ms.model_dump(),
            "debug": result.debug.model_dump(),
        }
    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(audio: UploadFile = File(...)) -> TranscriptionResponse:
    """Transcribe a single uploaded audio file.

    Args:
        audio: Uploaded audio file in any format supported by the STT service.

    Returns:
        TranscriptionResponse with transcript text, timing breakdowns, and debug info.

    Raises:
        HTTPException: If the uploaded file is empty.
    """
    request_id = uuid4().hex[:8]
    started_at = perf_counter()
    logger.info(
        "request_id={} event=request_received filename={} content_type={}",
        request_id,
        audio.filename,
        audio.content_type,
    )

    filename = audio.filename or "recording.webm"
    suffix = Path(filename).suffix or ".webm"
    temp_path = settings.temp_dir / f"{request_id}{suffix}"

    try:
        read_started_at = perf_counter()
        content = await audio.read()
        request_read_ms = round((perf_counter() - read_started_at) * 1000, 2)

        if not content:
            raise HTTPException(status_code=400, detail="Audio file is empty.")

        write_started_at = perf_counter()
        temp_path.write_bytes(content)
        file_write_ms = round((perf_counter() - write_started_at) * 1000, 2)

        service = get_stt_service()
        result = service.transcribe(
            file_path=temp_path,
            request_id=request_id,
            filename=filename,
            audio_bytes=len(content),
        )
        total_ms = round((perf_counter() - started_at) * 1000, 2)
        result.timings_ms.request_read_ms = request_read_ms
        result.timings_ms.file_write_ms = file_write_ms
        result.timings_ms.total_ms = total_ms

        logger.info(
            "request_id={} event=request_finished total_ms={} request_read_ms={} file_write_ms={} transcribe_ms={}",
            request_id,
            total_ms,
            request_read_ms,
            file_write_ms,
            result.timings_ms.transcribe_ms,
        )
        return TranscriptionResponse(
            text=result.text,
            timings_ms=result.timings_ms,
            debug=result.debug,
        )
    finally:
        temp_path.unlink(missing_ok=True)


@app.websocket("/ws/transcribe")
async def transcribe_stream(websocket: WebSocket) -> None:
    """Handle a WebSocket session for real-time streaming transcription and agent turns.

    Accepts binary PCM-16 audio chunks and JSON control messages. Emits partial
    and final transcripts, schedules LLM calls via AgentPipeline, and streams TTS
    audio back to the client.
    """
    await websocket.accept()
    request_id = uuid4().hex[:8]
    logger.info("request_id={} event=ws_connected client={}", request_id, websocket.client)

    sample_rate: int | None = None
    pcm_buffer = bytearray()
    chunk_count = 0
    last_emit_at = 0.0
    last_text_sent = ""
    send_lock = asyncio.Lock()
    silence_debounce_task: asyncio.Task[None] | None = None

    session_log = SessionLog(
        session_id=request_id,
        stt_model=settings.stt_model_size,
        stt_device=settings.stt_device,
        stt_compute_type=settings.stt_compute_type,
        stt_vad_filter=settings.stt_vad_filter,
        stt_beam_size=settings.stt_beam_size,
        llm_model=settings.llm_model,
        llm_host=settings.ollama_host,
    )

    async def send_json(payload: dict[str, object]) -> None:
        """Send a JSON payload to the WebSocket client, swallowing disconnect errors.

        Args:
            payload: JSON-serialisable dict to send.
        """
        async with send_lock:
            try:
                await websocket.send_json(payload)
            except (RuntimeError, WebSocketDisconnect):
                pass

    # ── Session-log callbacks passed to AgentPipeline ────────────────────────

    def on_turn_start() -> None:
        """Clear PCM buffer and dedup state before a new LLM turn begins."""
        nonlocal last_text_sent, chunk_count, last_emit_at
        pcm_buffer.clear()
        last_text_sent = ""
        chunk_count = 0
        last_emit_at = 0.0

    def on_llm_done(
        trigger: str,
        llm_ms: float,
        full_prompt: str,
        response: str,
        error: str | None,
    ) -> None:
        """Append an LLMCallLog entry to the session log.

        Args:
            trigger: What triggered the LLM call (e.g. ``"final"``).
            llm_ms: LLM inference latency in milliseconds.
            full_prompt: Full serialised prompt sent to the LLM.
            response: LLM response text.
            error: Error message if the call failed, otherwise ``None``.
        """
        session_log.llm_calls.append(
            LLMCallLog(
                timestamp=_iso(),
                trigger=trigger,
                latency_ms=llm_ms,
                model=settings.llm_model,
                host=settings.ollama_host,
                full_prompt=full_prompt,
                output_response=response,
                cancelled=False,
                error=error,
            )
        )

    def on_tts_synth(text: str, sample_rate: int, latency_ms: float, error: str | None) -> None:
        """Append a TTSCallLog entry to the session log.

        Args:
            text: Text that was synthesised.
            sample_rate: Sample rate of the output audio in Hz.
            latency_ms: TTS synthesis latency in milliseconds.
            error: Error message if synthesis failed, otherwise ``None``.
        """
        session_log.tts_calls.append(
            TTSCallLog(
                timestamp=_iso(),
                latency_ms=latency_ms,
                input_text=text,
                output_audio_bytes=0,
                output_sample_rate=sample_rate,
                error=error,
            )
        )

    pipeline = AgentPipeline(
        session_id=request_id,
        send_json_fn=send_json,
        on_turn_start=on_turn_start,
        on_llm_done=on_llm_done,
        on_tts_synth=on_tts_synth,
    )

    async def silence_debounce_then_fire(text: str, trigger: str) -> None:
        """Wait for the silence debounce window, then schedule an LLM call.

        Cancelled when new audio arrives before the window elapses.

        Args:
            text: Transcript text to send to the LLM.
            trigger: Label for session logging.
        """
        try:
            await asyncio.sleep(settings.stream_llm_silence_ms / 1000)
        except asyncio.CancelledError:
            return
        if PAUSE_PATTERN.match(text.strip()):
            await pipeline.interrupt()
            await send_json({"type": "doc_reading_pause"})
            return
        pipeline.schedule_llm(text, trigger)

    await send_json({"type": "ready", "request_id": request_id})
    asyncio.create_task(pipeline.run_welcome())

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()

            if message.get("text") is not None:
                payload = json.loads(message["text"])
                event_type = payload.get("type")

                if event_type == "start":
                    sample_rate = int(payload.get("sample_rate", 16000))
                    logger.info("request_id={} event=stream_started sample_rate={}", request_id, sample_rate)
                    continue

                if event_type == "tts_voice":
                    pipeline.tts_voice = str(payload.get("voice", ""))
                    logger.info("request_id={} event=tts_voice_selected voice={}", request_id, pipeline.tts_voice)
                    continue

                if event_type == "tts_speed":
                    pipeline.tts_speed = float(payload.get("speed", 1.0))
                    logger.info("request_id={} event=tts_speed_selected speed={}", request_id, pipeline.tts_speed)
                    continue

                if event_type == "doc_load":
                    doc_id = str(payload.get("doc_id", ""))
                    doc = get_document_store().get_document(doc_id)
                    if doc:
                        pipeline.active_document_id = doc_id
                        pipeline.last_read_sentence_idx = -1
                        pipeline.resume_from_sentence_idx = None
                        annotations = get_document_store().load_annotations(doc_id)
                        await send_json({
                            "type": "doc_opened",
                            "doc_id": doc_id,
                            "title": doc.title,
                            "raw_markdown": doc.raw_markdown,
                            "sentences": doc.sentences,
                            "annotations": annotations,
                        })
                        logger.info("request_id={} event=doc_loaded doc_id={}", request_id, doc_id)
                    else:
                        await send_json({"type": "doc_error", "message": f"Document '{doc_id}' not found."})
                    continue

                if event_type == "doc_unload":
                    pipeline.active_document_id = None
                    pipeline.last_read_sentence_idx = -1
                    pipeline.resume_from_sentence_idx = None
                    continue

                if event_type == "continue_reading":
                    fallback_doc_id = str(payload.get("doc_id", ""))
                    if not pipeline.active_document_id and fallback_doc_id:
                        doc = get_document_store().get_document(fallback_doc_id)
                        if doc:
                            pipeline.active_document_id = fallback_doc_id

                    if pipeline.active_document_id and (pipeline.llm_task is None or pipeline.llm_task.done()):
                        doc = get_document_store().get_document(pipeline.active_document_id)
                        if doc:
                            word_idx = pipeline.find_resume_sentence_idx(doc.sentences)
                            start_idx = word_idx if word_idx is not None else pipeline.get_read_start_idx(restart_from_beginning=False)
                            if start_idx < doc.sentence_count:
                                await send_json({"type": "doc_reading_resume"})
                                pipeline.schedule_document_read(doc.doc_id, start_idx)
                            else:
                                await send_json({"type": "doc_reading_pause"})
                        else:
                            await send_json({"type": "doc_reading_pause"})
                    continue

                if event_type == "doc_read":
                    doc_id = str(payload.get("doc_id", ""))
                    restart_from_beginning = bool(payload.get("restart_from_beginning"))
                    doc = get_document_store().get_document(doc_id)
                    if not doc:
                        await send_json({"type": "doc_error", "message": f"Document '{doc_id}' not found."})
                        continue
                    pipeline.active_document_id = doc_id
                    pipeline.last_read_sentence_idx = -1
                    pipeline.resume_from_sentence_idx = None
                    start_idx = pipeline.get_read_start_idx(restart_from_beginning=restart_from_beginning)
                    pipeline.schedule_document_read(doc.doc_id, start_idx)
                    continue

                if event_type == "doc_save_highlight":
                    doc_id = str(payload.get("doc_id", ""))
                    sentence_idx = int(payload.get("sentence_idx", -1))
                    if doc_id and sentence_idx >= 0:
                        get_document_store().save_highlight(doc_id, sentence_idx)
                        await send_json({"type": "doc_highlight_saved", "sentence_idx": sentence_idx})
                    continue

                if event_type == "interrupt":
                    logger.info("request_id={} event=interrupt_received", request_id)
                    if silence_debounce_task is not None and not silence_debounce_task.done():
                        silence_debounce_task.cancel()
                        silence_debounce_task = None
                    await pipeline.interrupt()
                    continue

                if event_type == "pause_reading":
                    logger.info("request_id={} event=pause_reading_button", request_id)
                    if silence_debounce_task is not None and not silence_debounce_task.done():
                        silence_debounce_task.cancel()
                        silence_debounce_task = None
                    await pipeline.interrupt()
                    await send_json({"type": "doc_reading_pause"})
                    continue

                if event_type == "stop":
                    if silence_debounce_task is not None and not silence_debounce_task.done():
                        silence_debounce_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await silence_debounce_task
                        silence_debounce_task = None

                    if sample_rate and pcm_buffer:
                        audio_duration_ms = round(len(pcm_buffer) / 2 / sample_rate * 1000, 2)
                        audio_file_path = str(settings.temp_dir / f"{request_id}_stream.wav")
                        _buf_snap = bytearray(pcm_buffer)
                        _cnt_snap = chunk_count
                        result_payload = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: _transcribe_stream_buffer(
                                request_id=request_id,
                                sample_rate=sample_rate,
                                pcm_buffer=_buf_snap,
                                chunk_count=_cnt_snap,
                            ),
                        )
                        await send_json({"type": "final", **result_payload})

                        timings = result_payload.get("timings_ms", {})
                        debug = result_payload.get("debug", {})
                        final_text = str(result_payload.get("text", "")).strip()

                        session_log.stt_runs.append(
                            STTRunLog(
                                timestamp=_iso(),
                                trigger="final",
                                latency_ms=timings.get("transcribe_ms", 0),
                                audio_file_path=audio_file_path,
                                audio_bytes=len(pcm_buffer),
                                audio_duration_ms=audio_duration_ms,
                                sample_rate=sample_rate,
                                transcript=final_text,
                                transcript_length_chars=len(final_text),
                                language_detected=debug.get("detected_language"),
                                segments=debug.get("segments", 0),
                            )
                        )

                        if final_text and PAUSE_PATTERN.match(final_text):
                            await pipeline.interrupt()
                            await send_json({"type": "doc_reading_pause"})
                        elif final_text and not pipeline.llm_responded and final_text != pipeline.latest_llm_input:
                            pipeline.schedule_llm(final_text, "final")
                        elif final_text and pipeline.llm_responded:
                            logger.info("request_id={} event=final_llm_skipped reason=already_responded", request_id)
                    else:
                        await send_json(
                            {
                                "type": "final",
                                "text": "",
                                "timings_ms": {
                                    "request_read_ms": 0,
                                    "file_write_ms": 0,
                                    "model_load_ms": 0,
                                    "transcribe_ms": 0,
                                    "total_ms": 0,
                                    "buffered_audio_ms": 0,
                                    "client_roundtrip_ms": None,
                                },
                                "debug": {
                                    "request_id": request_id,
                                    "filename": "stream.wav",
                                    "audio_bytes": 0,
                                    "detected_language": None,
                                    "segments": 0,
                                    "model_size": settings.stt_model_size,
                                    "device": settings.stt_device,
                                    "compute_type": settings.stt_compute_type,
                                    "sample_rate": sample_rate,
                                    "chunks_received": chunk_count,
                                },
                            }
                        )

                    # Wait for any in-flight LLM turn to finish before closing.
                    while pipeline.llm_task is not None:
                        current_task = pipeline.llm_task
                        await current_task
                        if pipeline.llm_task is current_task:
                            break
                    break

            if message.get("bytes") is None or sample_rate is None:
                continue

            pcm_buffer.extend(message["bytes"])
            chunk_count += 1
            buffered_audio_ms = len(pcm_buffer) / 2 / sample_rate * 1000
            now = perf_counter()
            should_emit = (
                buffered_audio_ms >= settings.stream_min_audio_ms
                and ((now - last_emit_at) * 1000) >= settings.stream_emit_interval_ms
            )

            if not should_emit:
                continue

            _buf_snap = bytearray(pcm_buffer)
            _cnt_snap = chunk_count
            result_payload = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _transcribe_stream_buffer(
                    request_id=request_id,
                    sample_rate=sample_rate,
                    pcm_buffer=_buf_snap,
                    chunk_count=_cnt_snap,
                ),
            )
            current_text = str(result_payload["text"])
            if current_text != last_text_sent:
                await send_json({"type": "partial", **result_payload})
                last_text_sent = current_text
                session_log.stt_partial_run_count += 1

                timings = result_payload.get("timings_ms", {})
                debug = result_payload.get("debug", {})
                session_log.stt_runs.append(
                    STTRunLog(
                        timestamp=_iso(),
                        trigger="partial",
                        latency_ms=timings.get("transcribe_ms", 0),
                        audio_file_path=str(settings.temp_dir / f"{request_id}_stream.wav"),
                        audio_bytes=len(pcm_buffer),
                        audio_duration_ms=round(len(pcm_buffer) / 2 / sample_rate * 1000, 2),
                        sample_rate=sample_rate,
                        transcript=current_text,
                        transcript_length_chars=len(current_text),
                        language_detected=debug.get("detected_language"),
                        segments=debug.get("segments", 0),
                    )
                )

                if silence_debounce_task is not None and not silence_debounce_task.done():
                    silence_debounce_task.cancel()
                silence_debounce_task = asyncio.create_task(
                    silence_debounce_then_fire(current_text, "debounced_partial")
                )

            last_emit_at = perf_counter()

    except WebSocketDisconnect:
        logger.info("request_id={} event=ws_disconnected chunks_received={}", request_id, chunk_count)
    except Exception as error:
        session_log.error = str(error)
        logger.exception("request_id={} event=ws_failed error={}", request_id, error)
        await send_json({"type": "error", "message": "Streaming transcription failed."})
    finally:
        if silence_debounce_task is not None and not silence_debounce_task.done():
            silence_debounce_task.cancel()
            with suppress(asyncio.CancelledError):
                await silence_debounce_task
        await pipeline.interrupt()
        write_session_log(session_log)
        try:
            await websocket.close()
        except RuntimeError:
            logger.debug("request_id={} event=ws_close_skipped", request_id)
