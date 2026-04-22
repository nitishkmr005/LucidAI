from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter

import numpy as np
from loguru import logger
from silero_vad import load_silero_vad
import torch

from config.settings import Settings, get_settings

_VAD_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class VADStreamEvent:
    event: str
    sample_index: int
    speech_prob: float


class StreamingVAD:
    def __init__(
        self,
        *,
        model: object,
        threshold: float,
        min_silence_duration_ms: int,
        speech_pad_ms: int,
        frame_samples: int,
    ) -> None:
        self._model = model
        self._threshold = threshold
        self._neg_threshold = max(threshold - 0.15, 0.01)
        self._min_silence_samples = _VAD_SAMPLE_RATE * min_silence_duration_ms / 1000
        self._speech_pad_samples = _VAD_SAMPLE_RATE * speech_pad_ms / 1000
        self._frame_samples = frame_samples
        self._pending = np.empty(0, dtype=np.float32)
        self._triggered = False
        self._temp_end = 0
        self._current_sample = 0
        self._last_speech_prob = 0.0
        self.reset()

    @property
    def in_speech(self) -> bool:
        return self._triggered

    @property
    def last_speech_prob(self) -> float:
        return self._last_speech_prob

    def reset(self) -> None:
        self._model.reset_states()
        self._pending = np.empty(0, dtype=np.float32)
        self._triggered = False
        self._temp_end = 0
        self._current_sample = 0
        self._last_speech_prob = 0.0

    def process_pcm16(self, pcm: bytes) -> list[VADStreamEvent]:
        if not pcm:
            return []

        samples = np.frombuffer(pcm, dtype=np.int16)
        if samples.size == 0:
            return []

        normalized = samples.astype(np.float32) / 32_768.0
        if self._pending.size:
            normalized = np.concatenate((self._pending, normalized))

        events: list[VADStreamEvent] = []
        offset = 0
        while normalized.size - offset >= self._frame_samples:
            frame = normalized[offset : offset + self._frame_samples]
            frame_tensor = torch.from_numpy(frame)
            self._current_sample += self._frame_samples
            speech_prob = float(self._model(frame_tensor, _VAD_SAMPLE_RATE).item())
            self._last_speech_prob = speech_prob

            if speech_prob >= self._threshold and self._temp_end:
                self._temp_end = 0

            if speech_prob >= self._threshold and not self._triggered:
                self._triggered = True
                speech_start = max(
                    0,
                    self._current_sample - self._speech_pad_samples - self._frame_samples,
                )
                events.append(
                    VADStreamEvent(
                        event="start",
                        sample_index=int(speech_start),
                        speech_prob=speech_prob,
                    )
                )

            elif speech_prob < self._neg_threshold and self._triggered:
                if not self._temp_end:
                    self._temp_end = self._current_sample
                if self._current_sample - self._temp_end >= self._min_silence_samples:
                    speech_end = self._temp_end + self._speech_pad_samples - self._frame_samples
                    self._temp_end = 0
                    self._triggered = False
                    events.append(
                        VADStreamEvent(
                            event="end",
                            sample_index=int(speech_end),
                            speech_prob=speech_prob,
                        )
                    )
            offset += self._frame_samples

        self._pending = normalized[offset:].copy()
        return events


class VoiceActivityService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: object | None = None

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model

        logger.info(
            "event=vad_model_load_started threshold={} frame_samples={} min_silence_ms={}",
            self._settings.stream_vad_threshold,
            self._settings.stream_vad_frame_samples,
            self._settings.stream_vad_min_silence_ms,
        )
        started_at = perf_counter()
        self._model = load_silero_vad()
        logger.info(
            "event=vad_model_load_finished model=silero-vad load_ms={}",
            round((perf_counter() - started_at) * 1000, 2),
        )
        return self._model

    def create_stream(self) -> StreamingVAD:
        return StreamingVAD(
            model=self._load_model(),
            threshold=self._settings.stream_vad_threshold,
            min_silence_duration_ms=self._settings.stream_vad_min_silence_ms,
            speech_pad_ms=self._settings.stream_vad_speech_pad_ms,
            frame_samples=self._settings.stream_vad_frame_samples,
        )


@lru_cache(maxsize=1)
def get_vad_service() -> VoiceActivityService:
    return VoiceActivityService(get_settings())
