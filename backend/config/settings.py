from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.prompts.system import VOICE_AGENT_PROMPT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ─────────────────────────────────────────────────────────────────────────
    # APP — server identity, networking, and logging
    #
    # app_host / app_port   Change if another process owns 8000, or to bind
    #                       only on loopback (127.0.0.1) for local-only use.
    # log_level             DEBUG floods every audio frame. INFO is the right
    #                       default. WARNING/ERROR for production silence.
    # cors_origins_raw      Comma-separated list of browser origins allowed to
    #                       connect. Add your deployed frontend URL here; wrong
    #                       value causes the browser to block all API calls.
    # ─────────────────────────────────────────────────────────────────────────
    app_name: str = "NeuroTalk STT Backend"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"          # DEBUG | INFO | WARNING | ERROR
    cors_origins_raw: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        alias="CORS_ORIGINS",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STT — faster-whisper speech-to-text
    #
    # stt_model_size        Accuracy vs speed trade-off.
    #                         tiny/base  — fastest, lower accuracy
    #                         small      — good balance (default)
    #                         medium     — noticeably better, ~2× slower
    #                         large-v3   — best accuracy, needs GPU
    # stt_device            cpu works everywhere. mps for Apple Silicon,
    #                       cuda for NVIDIA. float16/float32 compute types
    #                       require GPU; int8 is the only option on cpu.
    # stt_beam_size         1 = greedy decode (fastest). Higher values run
    #                       beam search for better accuracy at higher latency.
    # stt_vad_filter        Whisper's internal VAD pre-pass. Keeps it True to
    #                       skip silent frames and speed up transcription.
    #                       Disable only if you notice words being clipped.
    # stt_language          Force a language code (e.g. "en") to skip the
    #                       language-detection step and shave ~50 ms. Leave
    #                       empty ("") to auto-detect (slower, multilingual).
    # ─────────────────────────────────────────────────────────────────────────
    stt_model_size: str = "small"
    stt_model_path: Path = Path("models/stt")
    stt_device: str = "cpu"          # cpu | mps | cuda
    stt_compute_type: str = "int8"   # int8 (cpu) | float16 | float32 (gpu)
    stt_beam_size: int = 1
    stt_vad_filter: bool = True
    stt_language: str = "en"

    # ─────────────────────────────────────────────────────────────────────────
    # TTS — text-to-speech synthesis
    #
    # tts_backend           Selects the synthesis engine. Must match the uv
    #                       dependency group that was installed:
    #                         kokoro      — fast, high quality, local
    #                                       (uv sync --group kokoro_model)
    #                         chatterbox  — expressive, slower
    #                                       (uv sync --group chatterbox_model)
    #                       Changing this without re-syncing dependencies
    #                       will raise an ImportError at startup.
    # tts_model_path        Directory where the TTS model weights live.
    #                       Changing it to a non-existent path causes startup
    #                       to fail with FileNotFoundError during warmup.
    # welcome_message       Spoken aloud when a new session connects.
    #                       Set to "" to start sessions in silence.
    # ─────────────────────────────────────────────────────────────────────────
    tts_backend: str = "kokoro"      # kokoro | chatterbox | qwen | vibevoice | omnivoice
    tts_model_path: Path = Path("models/tts")
    welcome_message: str = "Hello! I'm your Neurotalk voice assistant. How can I assist you today?"

    # ─────────────────────────────────────────────────────────────────────────
    # VAD — Silero voice activity detection (speech start/end + barge-in)
    #
    # stream_vad_enabled        Disable to fall back to RMS energy barge-in.
    #                           Keep True; RMS fallback is much less accurate.
    # stream_vad_threshold      Probability above which a frame is speech.
    #                           Lower (0.4) = more sensitive, more false starts.
    #                           Higher (0.8) = misses soft speech.
    # stream_vad_min_silence_ms How long silence must last before VAD fires
    #                           "end". This is the primary gating delay:
    #                             400 ms — snappy but cuts off slow speakers
    #                             800 ms — balanced default
    #                            1200 ms — tolerant of long pauses, feels slow
    #                           Smart Turn runs AFTER this window, so this
    #                           value is also the minimum latency floor.
    # stream_vad_speech_pad_ms  Extra audio kept before/after speech edges.
    #                           Too low clips word starts; too high includes
    #                           ambient noise in the STT buffer.
    # stream_vad_frame_samples  Silero processes audio in these chunk sizes
    #                           (at 16 kHz). 512 = 32 ms per frame. Must be
    #                           512 or 1024 — other values break the model.
    # ─────────────────────────────────────────────────────────────────────────
    stream_vad_enabled: bool = True
    stream_vad_threshold: float = 0.6
    stream_vad_min_silence_ms: int = 2000. # After 2s of silence, call smart turn 
    stream_vad_speech_pad_ms: int = 250
    stream_vad_frame_samples: int = 512

    # ─────────────────────────────────────────────────────────────────────────
    # SMART TURN — semantic end-of-turn detection (pipecat-ai/smart-turn-v3)
    #
    # Runs after Silero VAD fires "end" to confirm the user actually finished
    # speaking, using a Whisper-Tiny-based classifier on the raw audio.
    # Total tolerance from user pause = stream_vad_min_silence_ms
    #                                 + stream_smart_turn_max_budget_ms.
    #
    # stream_smart_turn_enabled     Disable to revert to VAD-only endpointing
    #                               (old behaviour: VAD end → STT immediately).
    #                               Useful for debugging turn detection issues.
    # stream_smart_turn_threshold   Probability above which the turn is
    #                               considered complete and STT fires right away.
    #                               Lower (0.35) = fires sooner, may cut off.
    #                               Higher (0.65) = waits longer, safer for
    #                               slow speakers but adds latency.
    # stream_smart_turn_model_path  Directory containing smart-turn-v3.2-cpu.onnx.
    #                               Changing to a path without the model file
    #                               causes a FileNotFoundError at startup.
    # stream_smart_turn_base_wait_ms
    #                               Wait added between re-checks when the turn
    #                               is incomplete: wait = base_wait × (1 − prob).
    #                                 prob=0.08 → wait ≈ 920 ms (mid-sentence)
    #                                 prob=0.40 → wait ≈ 600 ms (borderline)
    #                               Increase for very slow/deliberate speakers.
    #                               Decrease to make the agent more interruption-
    #                               prone (faster but cuts off hesitant speech).
    # stream_smart_turn_max_budget_ms
    #                               Hard ceiling on how long Smart Turn can
    #                               keep waiting before firing STT regardless.
    #                               Combined with VAD silence (800 ms default),
    #                               the agent will always respond within:
    #                                 800 ms + 3000 ms = 3.8 s of silence.
    #                               Raise to support speakers who pause 4–5 s
    #                               mid-thought; lower for call-centre speed.
    # ─────────────────────────────────────────────────────────────────────────
    stream_smart_turn_enabled: bool = True
    stream_smart_turn_threshold: float = 0.75
    stream_smart_turn_model_path: Path = Path("models/smart_turn")
    stream_smart_turn_base_wait_ms: int = 1000
    stream_smart_turn_max_budget_ms: int = 3000. # if smart turn says incomplete, wait up to 3s and if still incomplete, then fire STT anyway

    # ─────────────────────────────────────────────────────────────────────────
    # STREAMING / PARTIAL STT — audio buffering and fallback debounce
    #
    # These control partial transcript emission and the silence-based fallback
    # that fires when VAD is disabled or Smart Turn is off.
    #
    # stream_emit_interval_ms   Minimum gap between partial STT emissions.
    #                           Lower = more frequent live transcript updates
    #                           in the UI but more STT CPU usage.
    # stream_min_audio_ms       Minimum audio buffered before STT is attempted.
    #                           Too low produces empty or single-word results.
    # stream_llm_min_chars      Transcript must be at least this many characters
    #                           before the LLM is called. Prevents the agent
    #                           from reacting to "um", "uh", breath noises.
    # stream_llm_silence_ms     Fallback silence debounce: fires STT → LLM
    #                           after this many ms of no new partial text.
    #                           Only active when VAD+Smart Turn are disabled.
    #                           With VAD on, Smart Turn takes over this role.
    #                             Lower → faster but may split one utterance
    #                             Higher → more reliable, higher latency
    # ─────────────────────────────────────────────────────────────────────────
    stream_emit_interval_ms: int = 250
    stream_min_audio_ms: int = 300
    stream_llm_min_chars: int = 8
    stream_llm_silence_ms: int = 950

    # ─────────────────────────────────────────────────────────────────────────
    # LLM — language model provider and behaviour
    #
    # llm_provider          Selects the inference backend. Switch via the
    #                       LLM_PROVIDER env var without restarting.
    #                         ollama    — fully local, no API key needed
    #                         openai    — hosted, fast, requires API key
    #                         anthropic — hosted, requires API key
    #                         gemini    — hosted, requires API key
    # llm_model             Model name for the selected Ollama backend.
    #                       For hosted providers use the provider-specific
    #                       fields below (openai_model, etc.).
    #                       Recommended Ollama models:
    #                         qwen3:4b      — fast, strong tool-calling (recommended)
    #                         qwen3:8b      — better quality, ~2× slower
    #                         gemma3:1b     — fastest, lower quality
    #                         gemma4:latest — high quality, 9.6 GB
    # llm_max_tokens        Hard cap on the agent's response length.
    #                       Too low truncates answers mid-sentence.
    #                       Too high slows TTS and makes responses verbose.
    # llm_max_history_turns Number of user+assistant turn pairs kept in
    #                       context. Higher = better conversational memory
    #                       but longer prompts and higher latency per turn.
    # llm_json_retry_attempts
    #                       Extra LLM calls when the JSON response fails
    #                       Pydantic validation. 0 = fail immediately.
    #                       Each retry adds full LLM latency to the turn.
    # llm_reading_context_sentences
    #                       How many recently-read document sentences are
    #                       included when the user asks a question. Higher
    #                       gives better context for Q&A but longer prompts.
    # ─────────────────────────────────────────────────────────────────────────
    llm_provider: str = "openai"     # ollama | openai | anthropic | gemini

    ollama_host: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"

    openai_api_key: str = ""
    openai_model: str = "gpt-5.4-nano"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    llm_max_tokens: int = 100
    llm_max_history_turns: int = 6
    llm_json_retry_attempts: int = 2
    llm_reading_context_sentences: int = 3
    llm_system_prompt: str = VOICE_AGENT_PROMPT

    # ─────────────────────────────────────────────────────────────────────────
    # WEB SEARCH — DuckDuckGo context enrichment
    #
    # web_search_enabled    Disable to prevent the agent from ever going online,
    #                       useful in air-gapped or offline deployments.
    # web_search_max_results
    #                       How many results are fetched and summarised.
    #                       Higher = richer answers, higher latency.
    # web_search_timeout_s  Hard timeout per search request. Increase on slow
    #                       connections; decrease to fail fast and keep the
    #                       agent responsive even when the network is flaky.
    # ─────────────────────────────────────────────────────────────────────────
    web_search_enabled: bool = True
    web_search_max_results: int = 3
    web_search_timeout_s: float = 5.0

    # ─────────────────────────────────────────────────────────────────────────
    # STORAGE — file system paths
    #
    # temp_dir      Scratch directory for in-flight audio WAV files. Cleared
    #               automatically after each STT call. Must be writable.
    # docs_dir      Where uploaded document files are persisted.
    # annotations_dir
    #               Highlights and notes JSON files, one per document.
    # exports_dir   Scratch directory for PDF/DOCX export files.
    # max_document_size_bytes
    #               Upload size limit. Increase for large research papers;
    #               the entire document is held in memory during a session.
    # ─────────────────────────────────────────────────────────────────────────
    temp_dir: Path = Path(".cache/audio")
    docs_dir: Path = Path("data/documents")
    annotations_dir: Path = Path("data/annotations")
    exports_dir: Path = Path(".cache/exports")
    max_document_size_bytes: int = 5 * 1024 * 1024  # 5 MB

    @property
    def cors_origins(self) -> list[str]:
        """Parse the comma-separated CORS_ORIGINS env var into a list of origin strings."""
        return [item.strip() for item in self.cors_origins_raw.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the global Settings singleton, loaded once from .env."""
    return Settings()
