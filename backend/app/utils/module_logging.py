from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

_BACKEND_DIR = Path(__file__).resolve().parents[2]
_LOGS_DIR = _BACKEND_DIR / "logs"
_MAX_LLM_TRACE_FILES = 5


def _prune_llm_trace_files() -> None:
    files = sorted(
        _LOGS_DIR.glob("llm_trace_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for old_file in files[_MAX_LLM_TRACE_FILES:]:
        old_file.unlink(missing_ok=True)


def log_module_io(
    *,
    module: str,
    latency_ms: float,
    input_payload: object,
    output_payload: object,
) -> None:
    if module != "llm":
        return

    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    trace_path = _LOGS_DIR / f"llm_trace_{timestamp}_{uuid4().hex[:8]}.json"
    payload = {
        "latency_ms": latency_ms,
        "input": input_payload,
        "output": output_payload,
    }
    trace_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _prune_llm_trace_files()
