from __future__ import annotations

import io
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.services.tts import TTSService


def make_mock_model(sample_rate: int = 24000, duration_samples: int = 24000) -> MagicMock:
    model = MagicMock()
    result = SimpleNamespace(
        audio=np.zeros(duration_samples, dtype=np.float32),
        sample_rate=sample_rate,
    )
    model.generate.return_value = [result]
    return model


@pytest.mark.asyncio
async def test_synthesize_returns_bytes_and_sample_rate():
    service = TTSService()
    service._model = make_mock_model()

    wav_bytes, sr = await service.synthesize("Hello world.")

    assert isinstance(wav_bytes, bytes)
    assert sr == 24000
    assert len(wav_bytes) > 44  # at least WAV header (44 bytes)


@pytest.mark.asyncio
async def test_synthesize_strips_emotion_tags_for_kokoro():
    service = TTSService()
    mock_model = make_mock_model()
    service._model = mock_model

    await service.synthesize("Happy to help. [chuckle] Let me check that.")

    mock_model.generate.assert_called_once()
    args, kwargs = mock_model.generate.call_args
    assert args == ("Happy to help. Let me check that.",)
    assert kwargs["voice"].endswith("models/tts/voices/af_heart.safetensors")
    assert kwargs["speed"] == 1.0
    assert kwargs["lang_code"] == "a"


@pytest.mark.asyncio
async def test_synthesize_output_is_valid_wav():
    service = TTSService()
    service._model = make_mock_model()

    wav_bytes, sr = await service.synthesize("Test audio.")

    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        assert wf.getframerate() == sr
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2


@pytest.mark.asyncio
async def test_synthesize_reuses_loaded_model():
    service = TTSService()
    mock_model = make_mock_model()
    service._model = mock_model

    await service.synthesize("First call.")
    await service.synthesize("Second call.")

    assert mock_model.generate.call_count == 2


@pytest.mark.asyncio
async def test_synthesize_uses_selected_kokoro_voice():
    service = TTSService()
    mock_model = make_mock_model()
    service._model = mock_model

    await service.synthesize("Voice test.", voice="am_adam")

    mock_model.generate.assert_called_once()
    args, kwargs = mock_model.generate.call_args
    assert args == ("Voice test.",)
    assert kwargs["voice"].endswith("models/tts/voices/am_adam.safetensors")
    assert kwargs["speed"] == 1.0
    assert kwargs["lang_code"] == "a"


def test_get_tts_service_returns_singleton():
    from app.services.tts import get_tts_service

    s1 = get_tts_service()
    s2 = get_tts_service()
    assert s1 is s2
