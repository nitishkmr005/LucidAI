from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter

from faster_whisper import WhisperModel
from loguru import logger

from app.models import DebugInfo, LatencyMetrics
from app.utils.module_logging import log_module_io
from config.settings import Settings, get_settings

_BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _resolve_model_path(path: Path) -> Path:
    return path if path.is_absolute() else _BACKEND_ROOT / path


@dataclass
class ServiceResult:
    text: str
    timings_ms: LatencyMetrics
    debug: DebugInfo


class SpeechToTextService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: WhisperModel | None = None

    def _load_model(self) -> tuple[WhisperModel, float]:
        if self._model is not None:
            return self._model, 0.0

        model_path = _resolve_model_path(self.settings.stt_model_path)
        logger.info(
            "event=model_load_started model_path={} model_size={} device={} compute_type={}",
            model_path,
            self.settings.stt_model_size,
            self.settings.stt_device,
            self.settings.stt_compute_type,
        )
        if not model_path.exists():
            raise FileNotFoundError(f"Local STT model path not found: {model_path}")
        started_at = perf_counter()
        self._model = WhisperModel(
            str(model_path),
            device=self.settings.stt_device,
            compute_type=self.settings.stt_compute_type,
        )
        model_load_ms = round((perf_counter() - started_at) * 1000, 2)
        logger.info("event=model_load_finished model_load_ms={}", model_load_ms)
        return self._model, model_load_ms

    def transcribe(self, *, file_path: Path, request_id: str, filename: str, audio_bytes: int) -> ServiceResult:
        model, model_load_ms = self._load_model()

        logger.info("request_id={} event=transcribe_started path={}", request_id, file_path)
        started_at = perf_counter()
        segments, info = model.transcribe(
            str(file_path),
            beam_size=self.settings.stt_beam_size,
            language=self.settings.stt_language or None,
            vad_filter=self.settings.stt_vad_filter,
            vad_parameters=dict(min_silence_duration_ms=500),
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            condition_on_previous_text=False,
        )
        segment_list = list(segments)
        transcribe_ms = round((perf_counter() - started_at) * 1000, 2)
        text = " ".join(segment.text.strip() for segment in segment_list if segment.text.strip()).strip()

        logger.info(
            "request_id={} event=transcribe_finished language={} segments={} text_length={} transcribe_ms={}",
            request_id,
            info.language,
            len(segment_list),
            len(text),
            transcribe_ms,
        )
        log_module_io(
            module="stt",
            latency_ms=transcribe_ms,
            input_payload={
                "request_id": request_id,
                "filename": filename,
                "file_path": str(file_path),
                "audio_bytes": audio_bytes,
            },
            output_payload={
                "text": text,
                "language": info.language,
                "segments": len(segment_list),
            },
        )

        return ServiceResult(
            text=text,
            timings_ms=LatencyMetrics(
                model_load_ms=model_load_ms,
                transcribe_ms=transcribe_ms,
            ),
            debug=DebugInfo(
                request_id=request_id,
                filename=filename,
                audio_bytes=audio_bytes,
                detected_language=info.language,
                segments=len(segment_list),
                model_size=str(_resolve_model_path(self.settings.stt_model_path)),
                device=self.settings.stt_device,
                compute_type=self.settings.stt_compute_type,
            ),
        )


@lru_cache(maxsize=1)
def get_stt_service() -> SpeechToTextService:
    return SpeechToTextService(get_settings())
