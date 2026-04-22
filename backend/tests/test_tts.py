from __future__ import annotations

import io
import wave
from unittest.mock import MagicMock

import pytest
import torch

from app.services.tts import TTSService


def make_mock_model(sample_rate: int = 24000, duration_samples: int = 24000) -> MagicMock:
    model = MagicMock()
    model.sr = sample_rate
    model.generate.return_value = torch.zeros(1, duration_samples)
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
async def test_synthesize_passes_text_with_emotion_tags_to_model():
    service = TTSService()
    mock_model = make_mock_model()
    service._model = mock_model

    await service.synthesize("Happy to help. [chuckle] Let me check that.")

    mock_model.generate.assert_called_once_with("Happy to help. [chuckle] Let me check that.")


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


def test_get_tts_service_returns_singleton():
    from app.services.tts import get_tts_service

    s1 = get_tts_service()
    s2 = get_tts_service()
    assert s1 is s2
