import asyncio
import base64
from contextlib import suppress
import json
import re
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
from app.services.llm import complete_llm_response
from app.services.search import web_search
from app.services.stt import get_stt_service
from app.services.tts import get_available_tts_voices, get_default_tts_voice, get_tts_service
from app.services.vad import get_vad_service
from app.utils.document_turns import (
    DocumentTurnDecision,
    build_document_turn_context,
    detect_direct_read_intent,
    parse_document_turn_response,
    resolve_document_by_name,
    user_explicitly_named_document,
)
from app.utils.emotion import clean_for_tts
from app.utils.session_logger import LLMCallLog, STTRunLog, SessionLog, TTSCallLog, _iso, write_session_log
from app.webrtc.router import router as webrtc_router
from config.logging import setup_logging
from config.settings import get_settings

setup_logging()
settings = get_settings()

_PAUSE_PATTERN = re.compile(
    r"^\s*(wait|hold on|hold up|one moment|one sec(?:ond)?|just a (?:moment|second|sec)|"
    r"give me a (?:second|moment|sec)|hang on|please wait|just wait|ok wait|okay wait|"
    r"stop|stop it|stop please|please stop|ok stop|okay stop)\s*[.!?,]?\s*$",
    re.IGNORECASE,
)
_CONTINUE_READING_PATTERN = re.compile(
    r"^\s*(keep reading|continue reading|continue|resume reading|resume)\s*[.!?,]?\s*$",
    re.IGNORECASE,
)
_READ_FROM_BEGINNING_PATTERN = re.compile(
    r"\b(read|start reading|read aloud|resume reading|continue reading)\b.*\b(beginning|start|top)\b|\bstart over\b|\brestart\b",
    re.IGNORECASE,
)
_TTS_PREVIEW_TEXT = "Welcome to NeuroTalk. This is the selected reading voice."
_TTS_PREVIEW_MAX_CHARS = 160


class TTSPreviewRequest(BaseModel):
    voice: str | None = None
    text: str | None = None

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


@app.on_event("startup")
async def startup_event() -> None:
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
    return HealthResponse()


@app.get("/tts/voices")
async def list_tts_voices() -> dict[str, object]:
    return {
        "default_voice": get_default_tts_voice(),
        "voices": get_available_tts_voices(),
    }


@app.post("/tts/preview")
async def preview_tts_voice(body: TTSPreviewRequest) -> Response:
    valid_voices = {item["id"] for item in get_available_tts_voices()}
    voice = body.voice if body.voice in valid_voices else get_default_tts_voice()
    preview_text = (body.text or _TTS_PREVIEW_TEXT).strip()[:_TTS_PREVIEW_MAX_CHARS]
    if not preview_text:
        preview_text = _TTS_PREVIEW_TEXT
    wav_bytes, _sample_rate = await get_tts_service().synthesize(preview_text, voice=voice)
    return Response(content=wav_bytes, media_type="audio/wav")


def write_pcm16_wav(*, pcm_bytes: bytes, sample_rate: int, file_path: Path) -> float:
    started_at = perf_counter()
    with wave.open(str(file_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return round((perf_counter() - started_at) * 1000, 2)


def transcribe_stream_buffer(
    *,
    request_id: str,
    sample_rate: int,
    pcm_buffer: bytearray,
    chunk_count: int,
) -> dict[str, object]:
    started_at = perf_counter()
    temp_path = settings.temp_dir / f"{request_id}_stream.wav"
    try:
        file_write_ms = write_pcm16_wav(pcm_bytes=bytes(pcm_buffer), sample_rate=sample_rate, file_path=temp_path)
        service = get_stt_service()
        result = service.transcribe(
            file_path=temp_path,
            request_id=request_id,
            filename=f"stream_{request_id}.wav",
            audio_bytes=len(pcm_buffer),
        )
        buffered_audio_ms = round(len(pcm_buffer) / 2 / sample_rate * 1000, 2)
        total_ms = round((perf_counter() - started_at) * 1000, 2)
        result.timings_ms.file_write_ms = file_write_ms
        result.timings_ms.total_ms = total_ms
        result.timings_ms.buffered_audio_ms = buffered_audio_ms
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
    await websocket.accept()
    request_id = uuid4().hex[:8]
    logger.info("request_id={} event=ws_connected client={}", request_id, websocket.client)

    sample_rate: int | None = None
    pcm_buffer = bytearray()
    chunk_count = 0
    last_emit_at = 0.0
    last_text_sent = ""
    send_lock = asyncio.Lock()
    llm_task: asyncio.Task | None = None
    active_tts_task: asyncio.Task | None = None
    pending_llm_call: tuple[str, str] | None = None
    latest_llm_input = ""
    interrupt_event = asyncio.Event()
    silence_debounce_task: asyncio.Task[None] | None = None
    conversation_history: list[dict[str, str]] = []  # grows each completed turn
    llm_responded = False  # True once a full response has been delivered this turn
    active_document_id: str | None = None  # currently loaded document
    last_read_sentence_idx: int = -1       # last sentence index highlighted during reading
    resume_from_sentence_idx: int | None = None
    selected_tts_voice: str = get_default_tts_voice()
    last_answer_text: str = ""

    # ── Session log scaffolding ───────────────────────────────────────────────
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
        async with send_lock:
            try:
                await websocket.send_json(payload)
            except (RuntimeError, WebSocketDisconnect):
                pass

    def _build_llm_prompt_dump(*, user_text: str, document_context: str | None) -> str:
        history = "\n".join(
            f"{entry['role']}: {entry['content']}" for entry in conversation_history
        )
        parts = [
            f"system: {settings.llm_system_prompt}",
            f"document_context: {document_context or ''}",
            f"history:\n{history}" if history else "history:",
            f"user: {user_text}",
        ]
        return "\n\n".join(parts)

    async def _tts_sentence_pipeline(queue: asyncio.Queue[tuple[str, int | None, str | None] | None]) -> None:
        """
        Consume queued utterances from *queue* and synthesise + stream each one immediately.

        Runs concurrently with the LLM token loop so the agent starts speaking
        as soon as the first sentence boundary is detected — without waiting for
        the full LLM response to complete.  A ``None`` sentinel signals the end.
        """
        nonlocal last_read_sentence_idx, resume_from_sentence_idx
        tts_service = get_tts_service()
        tts_started = False
        tts_t0 = perf_counter()

        while True:
            item = await queue.get()
            if item is None or interrupt_event.is_set():
                break
            tts_text, sentence_idx, display_text = item
            synth_started_at = perf_counter()
            try:
                wav_bytes, sr = await tts_service.synthesize(tts_text, voice=selected_tts_voice)
            except Exception as tts_err:
                session_log.tts_calls.append(
                    TTSCallLog(
                        timestamp=_iso(),
                        latency_ms=round((perf_counter() - synth_started_at) * 1000, 2),
                        input_text=tts_text,
                        output_audio_bytes=0,
                        output_sample_rate=0,
                        error=str(tts_err),
                    )
                )
                logger.warning("request_id={} event=tts_error error={}", request_id, tts_err)
                continue
            synth_latency_ms = round((perf_counter() - synth_started_at) * 1000, 2)
            session_log.tts_calls.append(
                TTSCallLog(
                    timestamp=_iso(),
                    latency_ms=synth_latency_ms,
                    input_text=tts_text,
                    output_audio_bytes=len(wav_bytes),
                    output_sample_rate=sr,
                )
            )
            if interrupt_event.is_set():
                break
            if not tts_started:
                tts_started = True
                tts_t0 = perf_counter()
                await send_json({"type": "tts_start"})
            if sentence_idx is not None:
                last_read_sentence_idx = sentence_idx
                resume_from_sentence_idx = sentence_idx
                if active_document_id:
                    get_document_store().save_reading_position(active_document_id, sentence_idx)
                await send_json({
                    "type": "doc_highlight",
                    "sentence_idx": sentence_idx,
                    "word_count": len((display_text or tts_text).split()),
                })
            tts_ms = round((perf_counter() - tts_t0) * 1000, 2)
            wav_b64 = base64.b64encode(wav_bytes).decode()
            await send_json({
                "type": "tts_audio",
                "data": wav_b64,
                "sample_rate": sr,
                "tts_ms": tts_ms,
                "sentence_text": display_text or tts_text,
                "sentence_idx": sentence_idx,
            })

        if interrupt_event.is_set():
            await send_json({"type": "tts_interrupted"})
        else:
            await send_json({"type": "tts_done"})

    async def _play_tts_turn(
        *,
        user_text: str,
        display_text: str,
        utterances: list[tuple[str, int | None, str | None]],
        llm_ms: float,
    ) -> None:
        nonlocal active_tts_task
        await send_json({"type": "llm_start", "user_text": user_text})
        await send_json({"type": "llm_partial", "text": display_text})
        await send_json({"type": "llm_final", "text": display_text, "llm_ms": llm_ms})
        if not utterances:
            return

        sent_queue: asyncio.Queue[tuple[str, int | None, str | None] | None] = asyncio.Queue()
        for utterance in utterances:
            await sent_queue.put(utterance)
        await sent_queue.put(None)

        tts_task = asyncio.create_task(_tts_sentence_pipeline(sent_queue))
        active_tts_task = tts_task
        with suppress(asyncio.CancelledError):
            await tts_task
        active_tts_task = None

    def _split_text_for_tts(text: str) -> list[tuple[str, int | None, str | None]]:
        utterances: list[tuple[str, int | None, str | None]] = []
        for raw in re.split(r"(?<=[.!?])\s+", text.strip()):
            sentence = clean_for_tts(raw.strip())
            if sentence:
                utterances.append((sentence, None, None))
        return utterances

    def _get_read_start_idx(*, restart_from_beginning: bool) -> int:
        if restart_from_beginning:
            return 0
        if resume_from_sentence_idx is not None:
            return max(0, resume_from_sentence_idx)
        if last_read_sentence_idx >= 0:
            return last_read_sentence_idx + 1
        return 0

    async def _run_document_read(
        *,
        doc_id: str,
        user_text: str,
        start_idx: int,
        llm_ms: float,
    ) -> str:
        nonlocal active_document_id, resume_from_sentence_idx
        doc = get_document_store().get_document(doc_id)
        if not doc:
            await _play_tts_turn(
                user_text=user_text,
                display_text="I couldn't find that document.",
                utterances=_split_text_for_tts("I couldn't find that document."),
                llm_ms=llm_ms,
            )
            return "I couldn't find that document."

        if start_idx >= doc.sentence_count:
            resume_from_sentence_idx = None
            await send_json({"type": "doc_reading_pause"})
            return ""

        active_document_id = doc.doc_id
        await send_json({
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
        await _play_tts_turn(user_text=user_text, display_text="", utterances=utterances, llm_ms=llm_ms)
        if not interrupt_event.is_set():
            resume_from_sentence_idx = None
        return f"Reading {doc.title} from sentence {start_idx}."

    async def _handle_doc_search(query: str) -> None:
        results = await web_search(query, max_results=settings.web_search_max_results)
        await send_json({"type": "doc_search_result", "query": query, "results": results})

    async def run_llm_stream(text: str, trigger: str) -> None:
        nonlocal llm_task, pending_llm_call, latest_llm_input, last_text_sent, chunk_count, last_emit_at
        nonlocal llm_responded, active_document_id, last_read_sentence_idx, resume_from_sentence_idx, last_answer_text
        llm_responded = False
        # Clear audio buffer — this utterance is captured; next audio is a new turn
        pcm_buffer.clear()
        last_text_sent = ""
        chunk_count = 0
        last_emit_at = 0.0
        interrupt_event.clear()
        llm_t0 = perf_counter()
        call_ts = _iso()
        display_response = ""
        llm_ms = 0.0
        call_error: str | None = None
        latest_llm_input = text

        try:
            direct_read = detect_direct_read_intent(text)
            if direct_read and active_document_id:
                if direct_read.restart_from_beginning:
                    last_read_sentence_idx = -1
                    resume_from_sentence_idx = None
                elif direct_read.action == "continue_reading" and resume_from_sentence_idx is None and last_read_sentence_idx < 0:
                    resume_from_sentence_idx = 0
                display_response = await _run_document_read(
                    doc_id=active_document_id,
                    user_text=text,
                    start_idx=_get_read_start_idx(restart_from_beginning=direct_read.restart_from_beginning),
                    llm_ms=0.0,
                )
                return

            document_context = build_document_turn_context(
                user_text=text,
                active_document_id=active_document_id,
                last_read_sentence_idx=last_read_sentence_idx,
            )
            full_prompt = _build_llm_prompt_dump(user_text=text, document_context=document_context)
            full_response = await complete_llm_response(
                text,
                conversation_history=list(conversation_history),
                document_context=document_context,
            )
            llm_ms = round((perf_counter() - llm_t0) * 1000, 2)
            logger.info("request_id={} event=llm_done llm_ms={}", request_id, llm_ms)
            decision = parse_document_turn_response(full_response) if document_context else None
            if active_document_id and _CONTINUE_READING_PATTERN.match(text):
                active_doc = get_document_store().get_document(active_document_id)
                if active_doc:
                    decision = DocumentTurnDecision(
                        action="continue_reading",
                        document_name=active_doc.title,
                        response_text="",
                        restart_from_beginning=False,
                    )
            elif active_document_id and _READ_FROM_BEGINNING_PATTERN.search(text):
                active_doc = get_document_store().get_document(active_document_id)
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
                    await send_json({"type": "doc_list", "documents": docs})
                    response_text = decision.response_text or (
                        "Available documents are: " + ", ".join(doc["title"] for doc in docs)
                    )
                    await _play_tts_turn(
                        user_text=text,
                        display_text=response_text,
                        utterances=_split_text_for_tts(response_text),
                        llm_ms=llm_ms,
                    )
                    display_response = response_text
                elif decision.action == "ask_document_clarification":
                    response_text = decision.response_text or "Which document would you like me to read?"
                    await _play_tts_turn(
                        user_text=text,
                        display_text=response_text,
                        utterances=_split_text_for_tts(response_text),
                        llm_ms=llm_ms,
                    )
                    display_response = response_text
                elif decision.action == "pause_reading":
                    resume_from_sentence_idx = None
                    response_text = decision.response_text or "Pausing here."
                    await send_json({"type": "doc_reading_pause"})
                    await _play_tts_turn(
                        user_text=text,
                        display_text=response_text,
                        utterances=_split_text_for_tts(response_text),
                        llm_ms=llm_ms,
                    )
                    display_response = response_text
                elif decision.action in {"read_document", "continue_reading"}:
                    if decision.action == "read_document" and not user_explicitly_named_document(text):
                        response_text = "Which document should I read?"
                        await _play_tts_turn(
                            user_text=text,
                            display_text=response_text,
                            utterances=_split_text_for_tts(response_text),
                            llm_ms=llm_ms,
                        )
                        display_response = response_text
                    else:
                        target_doc = resolve_document_by_name(
                            decision.document_name,
                            active_document_id=active_document_id if decision.action == "continue_reading" else None,
                        )
                        if target_doc is None:
                            response_text = "I couldn't tell which document to read. Which one do you want?"
                            await _play_tts_turn(
                                user_text=text,
                                display_text=response_text,
                                utterances=_split_text_for_tts(response_text),
                                llm_ms=llm_ms,
                            )
                            display_response = response_text
                        else:
                            if decision.restart_from_beginning:
                                last_read_sentence_idx = -1
                                resume_from_sentence_idx = None
                            elif decision.action == "continue_reading" and resume_from_sentence_idx is None and last_read_sentence_idx < 0:
                                resume_from_sentence_idx = 0
                            display_response = await _run_document_read(
                                doc_id=target_doc.doc_id,
                                user_text=text,
                                start_idx=_get_read_start_idx(restart_from_beginning=decision.restart_from_beginning),
                                llm_ms=llm_ms,
                            )
                elif decision.action == "save_note":
                    sentence_idx = decision.sentence_idx
                    if sentence_idx is None or sentence_idx < 0:
                        sentence_idx = max(0, last_read_sentence_idx)
                    note_text = decision.note_text or last_answer_text or decision.response_text
                    if not active_document_id or not note_text:
                        response_text = "I do not have a note to save yet."
                    else:
                        snippet = get_document_store().save_snippet(
                            doc_id=active_document_id,
                            term="AI note",
                            explanation=note_text,
                            sentence_idx=sentence_idx,
                            word_idx=-1,
                        )
                        await send_json({"type": "doc_note_saved", "snippet": snippet})
                        response_text = decision.response_text or "Saved that note on the sentence."
                    await _play_tts_turn(
                        user_text=text,
                        display_text=response_text,
                        utterances=_split_text_for_tts(response_text),
                        llm_ms=llm_ms,
                    )
                    display_response = response_text
                elif decision.action == "highlight_sentence":
                    sentence_idx = decision.sentence_idx
                    if sentence_idx is None or sentence_idx < 0:
                        sentence_idx = max(0, last_read_sentence_idx)
                    color = decision.highlight_color or "yellow"
                    if not active_document_id:
                        response_text = "I need an open document before I can highlight a sentence."
                    else:
                        get_document_store().save_highlight(active_document_id, sentence_idx, color)
                        await send_json({
                            "type": "doc_highlight_saved",
                            "sentence_idx": sentence_idx,
                            "color": color,
                        })
                        response_text = decision.response_text or "Highlighted that sentence."
                    await _play_tts_turn(
                        user_text=text,
                        display_text=response_text,
                        utterances=_split_text_for_tts(response_text),
                        llm_ms=llm_ms,
                    )
                    display_response = response_text
                else:
                    response_text = decision.response_text or "I'm not sure how to help with that yet."
                    await _play_tts_turn(
                        user_text=text,
                        display_text=response_text,
                        utterances=_split_text_for_tts(response_text),
                        llm_ms=llm_ms,
                    )
                    display_response = response_text
            else:
                display_response = full_response.strip()
                await _play_tts_turn(
                    user_text=text,
                    display_text=display_response,
                    utterances=_split_text_for_tts(display_response),
                    llm_ms=llm_ms,
                )

        except Exception as llm_err:
            llm_ms = round((perf_counter() - llm_t0) * 1000, 2)
            call_error = str(llm_err)
            logger.warning("request_id={} event=llm_error error={}", request_id, llm_err)
            with suppress(Exception):
                await send_json({"type": "llm_error", "message": "LLM unavailable — is Ollama running?"})

        session_log.llm_calls.append(
            LLMCallLog(
                timestamp=call_ts,
                trigger=trigger,
                latency_ms=llm_ms,
                model=settings.llm_model,
                host=settings.ollama_host,
                full_prompt=full_prompt,
                output_response=display_response,
                cancelled=False,
                error=call_error,
            )
        )

        if display_response and call_error is None and not interrupt_event.is_set():
            llm_responded = True
            if not display_response.startswith("Reading "):
                last_answer_text = display_response
            conversation_history.append({"role": "user", "content": text})
            conversation_history.append({"role": "assistant", "content": display_response})
            max_msgs = settings.llm_max_history_turns * 2
            if len(conversation_history) > max_msgs:
                conversation_history[:] = conversation_history[-max_msgs:]

        if pending_llm_call is None:
            llm_task = None
            return

        next_text, next_trigger = pending_llm_call
        pending_llm_call = None
        if next_text == latest_llm_input:
            llm_task = None
            return

        llm_task = asyncio.create_task(run_llm_stream(next_text, next_trigger))

    def schedule_llm_stream(text: str, trigger: str) -> None:
        nonlocal llm_task, pending_llm_call
        normalized_text = text.strip()
        if len(normalized_text) < settings.stream_llm_min_chars:
            return
        if normalized_text == latest_llm_input:
            return
        if _PAUSE_PATTERN.match(normalized_text):
            logger.info("request_id={} event=pause_command_detected text={}", request_id, normalized_text)
            return

        # If an LLM is already running for older text, cancel it — the user kept talking
        # and we have a more complete utterance now. Avoids "two replies for one query".
        if llm_task is not None and not llm_task.done():
            logger.info("request_id={} event=llm_cancel_for_newer_text", request_id)
            interrupt_event.set()
            pending_llm_call = None
            if active_tts_task is not None and not active_tts_task.done():
                active_tts_task.cancel()
                active_tts_task = None
            llm_task.cancel()
            llm_task = None

        pending_llm_call = (normalized_text, trigger)
        if llm_task is None or llm_task.done():
            next_text, next_trigger = pending_llm_call
            pending_llm_call = None
            llm_task = asyncio.create_task(run_llm_stream(next_text, next_trigger))

    async def _silence_debounce_then_fire(text: str, trigger: str) -> None:
        try:
            await asyncio.sleep(settings.stream_llm_silence_ms / 1000)
        except asyncio.CancelledError:
            return
        schedule_llm_stream(text, trigger)

    async def run_welcome() -> None:
        welcome = settings.welcome_message
        if not welcome:
            return
        await send_json({"type": "llm_start", "user_text": ""})
        await send_json({"type": "llm_final", "text": welcome, "llm_ms": 0})

        # Split on sentence-ending punctuation so each sentence gets its own tts_audio
        # message with sentence_text — matching the regular pipeline's synced text reveal.
        sent_queue: asyncio.Queue[tuple[str, int | None, str | None] | None] = asyncio.Queue()
        for raw in re.split(r"(?<=[.!?])\s+", welcome.strip()):
            sentence = clean_for_tts(raw.strip())
            if sentence:
                sent_queue.put_nowait((sentence, None, None))
        sent_queue.put_nowait(None)

        await _tts_sentence_pipeline(sent_queue)

    await send_json({"type": "ready", "request_id": request_id})
    asyncio.create_task(run_welcome())

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
                    voice = str(payload.get("voice", ""))
                    valid_voices = {item["id"] for item in get_available_tts_voices()}
                    selected_tts_voice = voice if voice in valid_voices else get_default_tts_voice()
                    logger.info("request_id={} event=tts_voice_selected voice={}", request_id, selected_tts_voice)
                    continue

                if event_type == "doc_load":
                    doc_id = str(payload.get("doc_id", ""))
                    doc = get_document_store().get_document(doc_id)
                    if doc:
                        active_document_id = doc_id
                        last_read_sentence_idx = -1
                        resume_from_sentence_idx = None
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
                    active_document_id = None
                    last_read_sentence_idx = -1
                    resume_from_sentence_idx = None
                    continue

                if event_type == "continue_reading":
                    if active_document_id and (llm_task is None or llm_task.done()):
                        doc = get_document_store().get_document(active_document_id)
                        if doc:
                            start_idx = _get_read_start_idx(restart_from_beginning=False)
                            if start_idx < doc.sentence_count:
                                llm_task = asyncio.create_task(
                                    _run_document_read(
                                        doc_id=doc.doc_id,
                                        user_text="",
                                        start_idx=start_idx,
                                        llm_ms=0.0,
                                    )
                                )
                            else:
                                await send_json({"type": "doc_reading_pause"})
                        else:
                            await send_json({"type": "doc_reading_pause"})
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
                    interrupt_event.set()
                    pending_llm_call = None
                    if silence_debounce_task is not None and not silence_debounce_task.done():
                        silence_debounce_task.cancel()
                        silence_debounce_task = None
                    if active_tts_task is not None and not active_tts_task.done():
                        active_tts_task.cancel()
                        active_tts_task = None
                    if llm_task is not None and not llm_task.done():
                        llm_task.cancel()
                        llm_task = None
                    continue

                if event_type == "stop":
                    # Cancel pending silence-debounce so it doesn't double-fire after the
                    # final transcript triggers its own LLM call.
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
                            lambda: transcribe_stream_buffer(
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

                        if final_text and not llm_responded and final_text.strip() != latest_llm_input:
                            schedule_llm_stream(final_text, "final")
                        elif final_text and llm_responded:
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

                    while llm_task is not None:
                        current_task = llm_task
                        await current_task
                        if llm_task is current_task:
                            llm_task = None
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
                lambda: transcribe_stream_buffer(
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
                # Debounce: only fire LLM once partial text has been stable for
                # `stream_llm_silence_ms` (i.e. user paused). Reset timer on every new partial.
                if silence_debounce_task is not None and not silence_debounce_task.done():
                    silence_debounce_task.cancel()
                silence_debounce_task = asyncio.create_task(
                    _silence_debounce_then_fire(current_text, "debounced_partial")
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
        if llm_task is not None and not llm_task.done():
            llm_task.cancel()
            with suppress(asyncio.CancelledError):
                await llm_task
        write_session_log(session_log)
        try:
            await websocket.close()
        except RuntimeError:
            logger.debug("request_id={} event=ws_close_skipped", request_id)
