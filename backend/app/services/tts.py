from __future__ import annotations

import asyncio
import io
import os
import wave
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from loguru import logger

from app.utils.module_logging import log_module_io
from config.settings import get_settings

_WARMUP_TEXT = "Hello."

_DEFAULT_VOICE = "af_heart"
_VOICE_ACCENTS: dict[str, str] = {
    "a": "American",
    "b": "British",
    "e": "Spanish",
    "f": "French",
    "h": "Hindi",
    "i": "Italian",
    "j": "Japanese",
    "p": "Portuguese",
    "z": "Mandarin",
}
_VOICE_GENDERS: dict[str, str] = {
    "f": "Female",
    "m": "Male",
}
_KOKORO_SPEED = 1.0
_KOKORO_LANG = "a"
_ESPEAK_CANDIDATES = [
    "/opt/homebrew/share/espeak-ng-data",
    "/usr/local/share/espeak-ng-data",
    "/usr/share/espeak-ng-data",
]
_BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _title_voice_name(voice_id: str) -> str:
    parts = voice_id.split("_", 1)
    if len(parts) != 2 or len(parts[0]) != 2:
        return voice_id.replace("_", " ").title()
    prefix, name = parts
    accent = _VOICE_ACCENTS.get(prefix[0], prefix[0].upper())
    gender = _VOICE_GENDERS.get(prefix[1], prefix[1].upper())
    return f"{name.replace('_', ' ').title()} ({accent} {gender})"


def _voice_sort_key(voice_id: str) -> tuple[int, str]:
    if voice_id == _DEFAULT_VOICE:
        return (0, voice_id)
    return (1, voice_id)


def _local_tts_model_path() -> Path:
    model_path = get_settings().tts_model_path
    return model_path if model_path.is_absolute() else _BACKEND_ROOT / model_path


def _kokoro_voice_ids() -> list[str]:
    voices_dir = _local_tts_model_path() / "voices"
    if not voices_dir.exists():
        return [_DEFAULT_VOICE]
    voice_ids = sorted((path.stem for path in voices_dir.glob("*.safetensors")), key=_voice_sort_key)
    return voice_ids or [_DEFAULT_VOICE]


def _kokoro_voice_path(voice_id: str) -> Path:
    return _local_tts_model_path() / "voices" / f"{voice_id}.safetensors"


class TTSService:
    def __init__(self) -> None:
        self._model: Any = None
        self._load_lock = asyncio.Lock()
        self._backend: str = get_settings().tts_backend

    def _load_model(self) -> Any:
        logger.info("event=tts_load backend={}", self._backend)
        model = self._load_kokoro() if self._backend == "kokoro" else self._load_chatterbox()
        logger.info("event=tts_ready backend={}", self._backend)
        return model

    def _load_kokoro(self) -> Any:
        if "ESPEAK_DATA_PATH" not in os.environ:
            for candidate in _ESPEAK_CANDIDATES:
                if Path(candidate).is_dir():
                    os.environ["ESPEAK_DATA_PATH"] = candidate
                    break
        from mlx_audio.tts.utils import load_model
        model_path = _local_tts_model_path()
        if not model_path.exists():
            raise FileNotFoundError(f"Local TTS model path not found: {model_path}")
        model = load_model(model_path)
        for _ in model.generate(_WARMUP_TEXT, voice=str(_kokoro_voice_path(_DEFAULT_VOICE)), speed=_KOKORO_SPEED, lang_code=_KOKORO_LANG):
            pass
        return model

    def _load_chatterbox(self) -> Any:
        import torch
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        model = ChatterboxTurboTTS.from_pretrained(device=device)
        model.generate(_WARMUP_TEXT)
        return model

    def _resolve_voice(self, voice: str | None) -> str:
        if self._backend != "kokoro":
            return _DEFAULT_VOICE
        if voice and voice in set(_kokoro_voice_ids()) and _kokoro_voice_path(voice).exists():
            return voice
        return _DEFAULT_VOICE

    def _resolve_speed(self, speed: float | None) -> float:
        if speed is None:
            return _KOKORO_SPEED
        return max(0.8, min(1.3, float(speed)))

    def _run_inference(self, text: str, voice: str | None = None, speed: float | None = None) -> tuple[bytes, int]:
        resolved_speed = self._resolve_speed(speed)
        return self._run_kokoro(text, self._resolve_voice(voice), resolved_speed) if self._backend == "kokoro" else self._run_chatterbox(text)

    def _run_kokoro(self, text: str, voice: str, speed: float) -> tuple[bytes, int]:
        from app.utils.emotion import strip_emotion_tags
        clean = strip_emotion_tags(text)
        final_audio = None
        sample_rate = 24000
        voice_path = _kokoro_voice_path(voice)
        if not voice_path.exists():
            raise FileNotFoundError(f"Local Kokoro voice not found: {voice_path}")
        for result in self._model.generate(clean, voice=str(voice_path), speed=speed, lang_code=_KOKORO_LANG):
            final_audio = result.audio
            sample_rate = getattr(result, "sample_rate", 24000)
        if final_audio is None:
            raise RuntimeError("Kokoro returned no audio")
        samples = np.asarray(final_audio).squeeze()
        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue(), sample_rate

    def _run_chatterbox(self, text: str) -> tuple[bytes, int]:
        import torch
        waveform = self._model.generate(text)
        samples = (
            waveform.detach().cpu().squeeze().numpy()
            if torch.is_tensor(waveform)
            else np.asarray(waveform).squeeze()
        )
        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._model.sr)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue(), self._model.sr

    async def synthesize(self, text: str, voice: str | None = None, speed: float | None = None) -> tuple[bytes, int]:
        if self._model is None:
            async with self._load_lock:
                if self._model is None:
                    loop = asyncio.get_event_loop()
                    self._model = await loop.run_in_executor(None, self._load_model)
        loop = asyncio.get_event_loop()
        started_at = perf_counter()
        resolved_voice = self._resolve_voice(voice)
        resolved_speed = self._resolve_speed(speed)
        wav_bytes, sample_rate = await loop.run_in_executor(None, self._run_inference, text, resolved_voice, resolved_speed)
        latency_ms = round((perf_counter() - started_at) * 1000, 2)
        log_module_io(
            module="tts",
            latency_ms=latency_ms,
            input_payload={"text": text, "backend": self._backend, "voice": resolved_voice, "speed": resolved_speed},
            output_payload={"audio_bytes": len(wav_bytes), "sample_rate": sample_rate},
        )
        return wav_bytes, sample_rate


_tts_service: TTSService | None = None


def get_tts_service() -> TTSService:
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service


def get_available_tts_voices() -> list[dict[str, str]]:
    return [
        {"id": voice_id, "name": _title_voice_name(voice_id)}
        for voice_id in _kokoro_voice_ids()
    ]


def get_default_tts_voice() -> str:
    return _DEFAULT_VOICE
