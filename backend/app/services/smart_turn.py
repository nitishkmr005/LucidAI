# The Smart Turn v3.2 ONNX model used by this module is distributed under the
# BSD 2-Clause License:
#
#   Copyright (c) 2024-2025, Daily
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#   1. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#
#   2. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#   POSSIBILITY OF SUCH DAMAGE.
#
# Source: https://github.com/pipecat-ai/smart-turn

"""Smart Turn v3.2 — semantic end-of-turn detector for NeuroTalk.

Loads pipecat-ai/smart-turn-v3 ONNX model from local disk and runs
CPU inference (~12 ms) to predict whether the user has finished speaking.
Model must be present at models/smart_turn/smart-turn-v3.2-cpu.onnx.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
from loguru import logger

_SAMPLE_RATE = 16_000
_MAX_SAMPLES = 8 * _SAMPLE_RATE   # 128 000 samples = 8 s
_MODEL_FILENAME = "smart-turn-v3.2-cpu.onnx"


class SmartTurnDetector:
    """Load Smart Turn v3.2 ONNX model from local disk and classify end-of-turn."""

    def __init__(self, model_path: Path, threshold: float = 0.5) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Smart Turn model not found at {model_path}. "
                "Run: python -c \"from huggingface_hub import hf_hub_download; "
                "hf_hub_download('pipecat-ai/smart-turn-v3', 'smart-turn-v3.2-cpu.onnx', "
                "local_dir='models/smart_turn')\""
            )
        self._model_path = model_path
        self._threshold = threshold
        self._session = None
        self._feature_extractor = None

    def _load(self) -> None:
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        t0 = perf_counter()
        logger.info("event=smart_turn_load_started path={}", self._model_path)

        cpu_count = max(1, (os.cpu_count() or 2) // 2)
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = cpu_count
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(str(self._model_path), sess_options=so)
        self._feature_extractor = WhisperFeatureExtractor(chunk_length=8)

        logger.info("event=smart_turn_load_finished load_ms={}", round((perf_counter() - t0) * 1000, 2))

    def warm_up(self) -> None:
        """Load model and run one dummy inference to trigger ONNX JIT compilation.

        Call this at application startup so the first real user turn has no
        cold-start penalty.
        """
        if self._session is None:
            self._load()
        # Run a silent dummy inference to compile the ONNX graph.
        dummy = np.zeros(_MAX_SAMPLES, dtype=np.float32)
        inputs = self._feature_extractor(
            dummy,
            sampling_rate=_SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=_MAX_SAMPLES,
            truncation=True,
            do_normalize=True,
        )
        input_features = inputs.input_features.squeeze(0).astype(np.float32)[np.newaxis]
        self._session.run(None, {"input_features": input_features})
        logger.info("event=smart_turn_warmup_done path={}", self._model_path)

    def is_complete(self, pcm_bytes: bytes) -> tuple[bool, float]:
        """Run Smart Turn inference on PCM audio.

        Args:
            pcm_bytes: 16 kHz mono int16 PCM bytes (current user utterance).

        Returns:
            ``(is_complete, probability)`` — probability > threshold means turn is done.
        """
        if self._session is None:
            self._load()
        if not pcm_bytes:
            return False, 0.0

        t0 = perf_counter()
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Keep the last 8 s (most recent speech carries the turn signal).
        if len(audio) > _MAX_SAMPLES:
            audio = audio[-_MAX_SAMPLES:]
        elif len(audio) < _MAX_SAMPLES:
            audio = np.pad(audio, (0, _MAX_SAMPLES - len(audio)))

        inputs = self._feature_extractor(
            audio,
            sampling_rate=_SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=_MAX_SAMPLES,
            truncation=True,
            do_normalize=True,
        )
        input_features = inputs.input_features.squeeze(0).astype(np.float32)[np.newaxis]
        outputs = self._session.run(None, {"input_features": input_features})
        probability = float(outputs[0][0])
        is_done = probability > self._threshold

        logger.debug(
            "event=smart_turn_inference prob={} is_complete={} inference_ms={}",
            round(probability, 4), is_done, round((perf_counter() - t0) * 1000, 2),
        )
        return is_done, probability


@lru_cache(maxsize=1)
def get_smart_turn_detector() -> SmartTurnDetector:
    """Return the singleton SmartTurnDetector, model resolved relative to the backend root."""
    from config.settings import get_settings
    settings = get_settings()
    model_dir = settings.stream_smart_turn_model_path
    if not model_dir.is_absolute():
        backend_root = Path(__file__).resolve().parents[2]
        model_dir = backend_root / model_dir
    return SmartTurnDetector(
        model_path=model_dir / _MODEL_FILENAME,
        threshold=settings.stream_smart_turn_threshold,
    )
