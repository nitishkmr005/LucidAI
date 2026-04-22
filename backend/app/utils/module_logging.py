from __future__ import annotations

from loguru import logger


def log_module_io(
    *,
    module: str,
    latency_ms: float,
    input_payload: object,
    output_payload: object,
) -> None:
    logger.bind(
        module_io={
            "module": module,
            "latency_ms": latency_ms,
            "input": input_payload,
            "output": output_payload,
        }
    ).info("event=module_io")
