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

_KOKORO_MODEL_ID = "mlx-community/Kokoro-82M-bf16"
_KOKORO_VOICE = "af_heart"
_KOKORO_SPEED = 1.0
_KOKORO_LANG = "a"
_ESPEAK_CANDIDATES = [
    "/opt/homebrew/share/espeak-ng-data",
    "/usr/local/share/espeak-ng-data",
    "/usr/share/espeak-ng-data",
]


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
        model = load_model(_KOKORO_MODEL_ID)
        for _ in model.generate(_WARMUP_TEXT, voice=_KOKORO_VOICE, speed=_KOKORO_SPEED, lang_code=_KOKORO_LANG):
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

    def _run_inference(self, text: str) -> tuple[bytes, int]:
        return self._run_kokoro(text) if self._backend == "kokoro" else self._run_chatterbox(text)

    def _run_kokoro(self, text: str) -> tuple[bytes, int]:
        from app.utils.emotion import strip_emotion_tags
        clean = strip_emotion_tags(text)
        final_audio = None
        sample_rate = 24000
        for result in self._model.generate(clean, voice=_KOKORO_VOICE, speed=_KOKORO_SPEED, lang_code=_KOKORO_LANG):
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

    async def synthesize(self, text: str) -> tuple[bytes, int]:
        if self._model is None:
            async with self._load_lock:
                if self._model is None:
                    loop = asyncio.get_event_loop()
                    self._model = await loop.run_in_executor(None, self._load_model)
        loop = asyncio.get_event_loop()
        started_at = perf_counter()
        wav_bytes, sample_rate = await loop.run_in_executor(None, self._run_inference, text)
        latency_ms = round((perf_counter() - started_at) * 1000, 2)
        log_module_io(
            module="tts",
            latency_ms=latency_ms,
            input_payload={"text": text, "backend": self._backend},
            output_payload={"audio_bytes": len(wav_bytes), "sample_rate": sample_rate},
        )
        return wav_bytes, sample_rate


_tts_service: TTSService | None = None


def get_tts_service() -> TTSService:
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service
