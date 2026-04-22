"""
runner.py — Chatterbox Turbo standalone demo with timing output.
"""

from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path
from typing import Any, Final


MODEL_ID: Final[str] = "ResembleAI/chatterbox-turbo"
VOICE_PATH: Final[Path] = Path("/tmp/voice_agent.wav")
EMOTION_TAGS: Final[str] = "[chuckle]"
WARMUP_TEXT: Final[str] = "Hello there."
DEFAULT_TEXT: Final[str] = "Happy to help. Tell me what you need, and we will sort it out together."


def scripts_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def default_output_path() -> Path:
    return scripts_dir() / "speech" / "chatterbox.wav"


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_wav(path: Path, audio: Any, sample_rate: int) -> None:
    import numpy as np
    import torch

    samples = audio.detach().cpu().squeeze().numpy() if torch.is_tensor(audio) else np.asarray(audio).squeeze()
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--output", default=str(default_output_path()))
    args = parser.parse_args()

    from chatterbox.tts_turbo import ChatterboxTurboTTS

    device = pick_device()
    final_text = f"{EMOTION_TAGS} {args.text}".strip()
    voice_exists = VOICE_PATH.exists()
    voice_label = str(VOICE_PATH) if voice_exists else "built-in default"

    print(f"Model: {MODEL_ID}")
    print(f"Device: {device}")
    print(f"Voice : {voice_label}")

    model = ChatterboxTurboTTS.from_pretrained(device=device)
    generate_kwargs: dict[str, str] = {}
    if voice_exists:
        generate_kwargs["audio_prompt_path"] = str(VOICE_PATH)

    _ = model.generate(WARMUP_TEXT, **generate_kwargs)

    started_at = time.perf_counter()
    waveform = model.generate(final_text, **generate_kwargs)
    latency_ms = round((time.perf_counter() - started_at) * 1000, 1)
    save_wav(Path(args.output), waveform, model.sr)

    notes = "Turbo tags via inline prompt."
    if not voice_exists:
        notes = f"{notes} Fell back to built-in default voice because {VOICE_PATH} was missing."

    print(json.dumps({"status": "ok", "latency_ms": latency_ms, "voice": voice_label, "output": args.output, "notes": notes}))


if __name__ == "__main__":
    main()
