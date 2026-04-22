from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

_LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"
_MAX_TRACE_FILES = 5


def _prune_llm_traces() -> None:
    files = sorted(_LOGS_DIR.glob("llm_trace_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[_MAX_TRACE_FILES:]:
        old.unlink(missing_ok=True)


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
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    path = _LOGS_DIR / f"llm_trace_{ts}.json"
    payload = {
        "module": module,
        "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "latency_ms": latency_ms,
        "input": input_payload,
        "output": output_payload,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _prune_llm_traces()
