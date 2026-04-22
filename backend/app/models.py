from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class LatencyMetrics(BaseModel):
    request_read_ms: float = Field(default=0)
    file_write_ms: float = Field(default=0)
    model_load_ms: float = Field(default=0)
    transcribe_ms: float = Field(default=0)
    total_ms: float = Field(default=0)
    buffered_audio_ms: float = Field(default=0)
    client_roundtrip_ms: float | None = Field(default=None)


class DebugInfo(BaseModel):
    request_id: str
    filename: str
    audio_bytes: int
    detected_language: str | None = None
    segments: int
    model_size: str
    device: str
    compute_type: str
    sample_rate: int | None = None
    chunks_received: int | None = None


class TranscriptionResponse(BaseModel):
    text: str
    timings_ms: LatencyMetrics
    debug: DebugInfo
