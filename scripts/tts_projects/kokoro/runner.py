"""
runner.py — Kokoro MLX standalone demo with timing output.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Final

# Point espeak-ng to the homebrew data dir when the espeakng_loader bundle
# has a broken CI-baked path (common on macOS with Apple Silicon).
_ESPEAK_DATA_CANDIDATES = [
    "/opt/homebrew/share/espeak-ng-data",
    "/usr/local/share/espeak-ng-data",
    "/usr/share/espeak-ng-data",
]
if "ESPEAK_DATA_PATH" not in os.environ:
    for _candidate in _ESPEAK_DATA_CANDIDATES:
        if Path(_candidate).is_dir():
            os.environ["ESPEAK_DATA_PATH"] = _candidate
            break

import soundfile as sf


MODEL_ID: Final[str] = "mlx-community/Kokoro-82M-bf16"
VOICE: Final[str] = "af_heart"
LANG_CODE: Final[str] = "a"
SPEED: Final[float] = 1.0
DEFAULT_TEXT: Final[str] = "Happy to help. Tell me what you need, and we will sort it out together."
WARMUP_TEXT: Final[str] = "Hello there."


def scripts_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def default_output_path() -> Path:
    return scripts_dir() / "speech" / "kokoro.wav"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--output", default=str(default_output_path()))
    args = parser.parse_args()

    from mlx_audio.tts.utils import load_model

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Model : {MODEL_ID}")
    print(f"Voice : {VOICE}")
    print(f"Lang  : {LANG_CODE}")

    model = load_model(MODEL_ID)

    for _ in model.generate(WARMUP_TEXT, voice=VOICE, speed=SPEED, lang_code=LANG_CODE):
        pass

    started_at = time.perf_counter()
    final_audio = None
    final_rate = 24000
    for result in model.generate(args.text, voice=VOICE, speed=SPEED, lang_code=LANG_CODE):
        final_audio = result.audio
        final_rate = getattr(result, "sample_rate", 24000)
    latency_ms = round((time.perf_counter() - started_at) * 1000, 1)

    if final_audio is None:
        print(json.dumps({"status": "error", "latency_ms": "n/a", "voice": VOICE, "output": str(output_path), "notes": "Kokoro returned no audio."}))
        return

    sf.write(output_path, final_audio, final_rate)
    print(json.dumps({"status": "ok", "latency_ms": latency_ms, "voice": VOICE, "output": str(output_path), "notes": f"MLX Kokoro with lang_code={LANG_CODE} speed={SPEED}."}))


if __name__ == "__main__":
    main()
