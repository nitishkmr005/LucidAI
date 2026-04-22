"""
Structured per-session log writer for the NeuroTalk pipeline.

Produces one JSON file per WebSocket session under logs/.
Each file has clearly separated STT, LLM, and TTS sections so you
can see latency, tokens, errors, and I/O for every pipeline stage.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

_SESSIONS_DIR = Path("logs")


def _iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ── STT ──────────────────────────────────────────────────────────────────────

@dataclass
class STTRunLog:
    """Single STT transcription pass (either a partial-stream run or the final stop run)."""
    timestamp: str
    trigger: str                    # "partial" | "final"
    latency_ms: float
    audio_file_path: str            # path used (file already deleted by the time we log)
    audio_bytes: int
    audio_duration_ms: float
    sample_rate: int
    transcript: str
    transcript_length_chars: int
    language_detected: str | None
    segments: int
    error: str | None = None


# ── LLM ──────────────────────────────────────────────────────────────────────

@dataclass
class LLMCallLog:
    """Single LLM inference call (debounced during recording or triggered at stop)."""
    timestamp: str
    trigger: str                    # "debounced_partial" | "final"
    latency_ms: float
    model: str
    host: str
    full_prompt: str
    output_response: str
    cancelled: bool = False
    error: str | None = None


# ── TTS ──────────────────────────────────────────────────────────────────────

@dataclass
class TTSCallLog:
    timestamp: str
    latency_ms: float
    input_text: str
    output_audio_bytes: int
    output_sample_rate: int
    error: str | None = None


# ── Session ───────────────────────────────────────────────────────────────────

@dataclass
class SessionLog:
    """Top-level container written at the end of every WebSocket session."""
    session_id: str
    session_start: str = field(default_factory=_iso)
    session_end: str | None = None

    # STT config — set once at session start
    stt_model: str = ""
    stt_device: str = ""
    stt_compute_type: str = ""
    stt_vad_filter: bool = True
    stt_beam_size: int = 1

    # STT runtime — accumulated during session
    stt_partial_run_count: int = 0
    stt_runs: list[STTRunLog] = field(default_factory=list)

    # LLM config — set once at session start
    llm_model: str = ""
    llm_host: str = ""
    # LLM runtime — one entry per inference call
    llm_calls: list[LLMCallLog] = field(default_factory=list)

    # TTS runtime
    tts_calls: list[TTSCallLog] = field(default_factory=list)

    # Session-level error (WebSocket crash, etc.)
    error: str | None = None


# ── Writer ────────────────────────────────────────────────────────────────────

def write_session_log(log: SessionLog) -> Path | None:
    """Serialise *log* to a pretty-printed JSON file and prune old files."""
    log.session_end = _iso()
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = _SESSIONS_DIR / f"session_{ts}_{log.session_id}.json"

    # Separate STT runs by trigger for readability
    stt_partial_runs = [asdict(r) for r in log.stt_runs if r.trigger == "partial"]
    stt_final_run = next((asdict(r) for r in log.stt_runs if r.trigger == "final"), None)

    payload = {
        "session_id": log.session_id,
        "session_start": log.session_start,
        "session_end": log.session_end,
        "pipeline": {
            "stt": {
                "config": {
                    "model": log.stt_model,
                    "device": log.stt_device,
                    "compute_type": log.stt_compute_type,
                    "vad_filter": log.stt_vad_filter,
                    "beam_size": log.stt_beam_size,
                },
                "summary": {
                    "partial_runs": log.stt_partial_run_count,
                    "final_run": stt_final_run,
                },
                "partial_run_detail": stt_partial_runs,
            },
            "llm": {
                "config": {
                    "model": log.llm_model,
                    "host": log.llm_host,
                },
                "calls": [asdict(c) for c in log.llm_calls],
                "summary": {
                    "total_calls": len(log.llm_calls),
                    "completed_calls": sum(1 for c in log.llm_calls if not c.cancelled and not c.error),
                    "cancelled_calls": sum(1 for c in log.llm_calls if c.cancelled),
                    "failed_calls": sum(1 for c in log.llm_calls if c.error),
                },
            },
            "tts": {
                "calls": [asdict(c) for c in log.tts_calls],
                "summary": {
                    "total_calls": len(log.tts_calls),
                    "failed_calls": sum(1 for c in log.tts_calls if c.error),
                },
            },
        },
        "error": log.error,
    }

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("event=session_log_written path={}", path)

    _prune_old_session_logs(keep=5)
    return path


def _prune_old_session_logs(keep: int = 5) -> None:
    files = sorted(_SESSIONS_DIR.glob("session_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[keep:]:
        old.unlink(missing_ok=True)
        logger.debug("event=session_log_pruned path={}", old)
