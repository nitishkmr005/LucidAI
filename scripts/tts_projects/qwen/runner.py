"""
runner.py — Qwen3-TTS standalone demo with timing output.
"""

from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path
from typing import Any, Final


MODEL_ID: Final[str] = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
SPEAKER: Final[str] = "Ryan"
LANGUAGE: Final[str] = "English"
INSTRUCT: Final[str] = "Very happy and energetic."
WARMUP_TEXT: Final[str] = "Hello there."
DEFAULT_TEXT: Final[str] = "Happy to help. Tell me what you need, and we will sort it out together."


def scripts_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def default_output_path() -> Path:
    return scripts_dir() / "speech" / "qwen.wav"


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
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

    import torch
    from qwen_tts import Qwen3TTSModel

    device = pick_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model_kwargs: dict[str, Any] = {
        "device_map": "cuda:0" if device == "cuda" else device,
        "dtype": dtype,
    }
    if device == "cuda":
        model_kwargs["attn_implementation"] = "flash_attention_2"

    print(f"Model  : {MODEL_ID}")
    print(f"Device : {device}")
    print(f"Speaker: {SPEAKER}")

    model = Qwen3TTSModel.from_pretrained(MODEL_ID, **model_kwargs)
    model.generate_custom_voice(text=WARMUP_TEXT, language=LANGUAGE, speaker=SPEAKER, instruct="")

    started_at = time.perf_counter()
    wavs, sample_rate = model.generate_custom_voice(
        text=args.text,
        language=LANGUAGE,
        speaker=SPEAKER,
        instruct=INSTRUCT,
    )
    latency_ms = round((time.perf_counter() - started_at) * 1000, 1)
    save_wav(Path(args.output), wavs[0], sample_rate)

    print(json.dumps({"status": "ok", "latency_ms": latency_ms, "voice": f"speaker={SPEAKER}", "output": args.output, "notes": f"instruct={INSTRUCT!r}; device={device}"}))


if __name__ == "__main__":
    main()
