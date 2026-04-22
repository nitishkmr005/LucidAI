# NeuroTalk — Open-Source Model Analysis (2026)

Target: real-time voice assistant on **macOS / Apple Silicon**, English-first, low-latency streaming.

## Current stack

| Layer | Current choice | Config |
|---|---|---|
| STT | `faster-whisper small` | `int8`, CPU, `beam_size=1`, `vad_filter=True`, `language="en"` |
| VAD | Silero VAD (inside faster-whisper) | `min_silence_duration_ms=500` |
| LLM | `gemma3:1b` via Ollama | `max_tokens=100`, 6-turn history |
| TTS | Kokoro-82M (`mlx-community/Kokoro-82M-bf16`) | MLX, voice `af_heart`, speed 1.0 |

---

## 1. Speech-to-Text (STT)

| Model | Params / Size | Target HW | English WER (clean) | Speed | Streaming | Fit |
|---|---|---|---|---|---|---|
| **faster-whisper small** (current) | 244 M / ~470 MB | CPU/GPU | ~10–12% | ~4× realtime on CPU | With chunking | Safe baseline, multilingual, but leaves perf on the table |
| faster-whisper distil-large-v3 | 756 M | CPU/GPU | ~8–9% | ~6× faster than large-v3 | OK | Better accuracy if CPU budget allows |
| faster-whisper large-v3-turbo | 809 M | GPU preferred | ~7–8% | Slow on pure CPU | Borderline | Skip for CPU real-time |
| **Moonshine Base** | 61 M / ~190 MB | CPU/edge | ~matches Whisper small | **~148 ms per utterance** | **Yes — Ergodic Streaming Encoder** | Strong pure-CPU pick; English-only |
| Moonshine Tiny | 27 M / ~27 MB | CPU/edge | ~matches Whisper tiny.en | ~50 ms | Yes | Smallest; quality dips slightly |
| **Parakeet-TDT 0.6B v2 (MLX)** | 600 M | Apple Silicon (MLX) | **~1.69% LibriSpeech** | **~24× realtime on M4**, RTFx >2000 class | Yes | **Best fit for this machine** — already using MLX for Kokoro; highest-accuracy fast model |
| Canary-Qwen 2.5B | 2.5 B | GPU | 5.63% (open ASR leader) | RTFx 418 | Not realistic on CPU | Skip |
| Qwen3-ASR 0.6B / 1.7B | 0.6 B / 1.7 B | GPU preferred | Competitive | Good | OK | Skip unless 52-lang needed |

**Recommendation:** switch to **Parakeet-TDT 0.6B v2 via `parakeet-mlx`**. MLX runtime is already a dependency (Kokoro TTS). Expect best-in-class WER (~1.7%) and ~24× realtime on M-series. Fallback: **Moonshine Base** if you want to stay pure-CPU and avoid MLX coupling.

---

## 2. Voice Activity Detection (VAD)

VAD runs **before** STT to cut non-speech audio and gate the LLM trigger. Currently Silero VAD is used implicitly inside faster-whisper.

| Engine | Backend | CPU cost | Accuracy (noisy) | Streaming | Integration surface | Fit |
|---|---|---|---|---|---|---|
| **Silero VAD** (current, via whisper) | PyTorch / ONNX | RTF 0.004 (ONNX ~2–3× faster) | Strong (6000+ langs trained) | Yes, 32 ms frames | `silero-vad` pip, or `faster-whisper vad_filter=True` | **Keep** — best accuracy/cost balance |
| WebRTC VAD | C (tiny, 158 KB) | <<1 ms per 30 ms frame | Weak — noise/speech confusion | Yes | Simple, `webrtcvad` pip | Good for pre-filter only; not sole judge |
| Pyannote VAD | PyTorch | Heavy (~440× slower than PocketSphinx on CPU) | State-of-the-art with GPU | Yes with GPU | `pyannote.audio` | Skip — GPU-oriented, too heavy here |
| Cobra VAD | C / native | Low | Strong | Yes | Picovoice SDK, non-permissive license | Skip for OSS-only setups |
| Yamnet VAD | TFLite | Low | Decent | Yes | TFLite dependency | Overkill unless classifying sounds |

**Extra wins inside Silero VAD (when invoked directly, not via whisper):**
- Stream-gate the mic: only forward audio to STT when VAD sees speech. Cuts STT CPU ~50% during quiet periods and kills most "Thank you." hallucinations at the source.
- Use VAD `speech_end_silence_ms` as the debounce instead of `stream_llm_silence_ms` in `settings.py`. More accurate turn-end detection than a fixed timer.
- Run as a separate pre-STT stage so the STT model doesn't see non-speech frames at all.

**Recommendation:** promote Silero VAD from "inside whisper" to **a first-class gate** in the WebSocket audio loop. Two immediate benefits: fewer hallucinations, sharper turn-end detection.

---

## 3. Large Language Model (LLM)

Criteria: TTFT (time-to-first-token) dominates perceived latency because we already stream tokens → TTS. Good small models + Apple Silicon unified memory = sub-second first token at 20–60 tok/s.

| Model | Params (Q4) | RAM (Q4_K_M) | Speed on M-series | Voice-reply quality | Fit |
|---|---|---|---|---|---|
| **gemma3:1b** (current) | 1 B | ~0.8 GB | Fastest, 100+ tok/s | Lower quality, short replies | Keep if speed > quality |
| **Llama 3.2 3B** | 3 B | ~2 GB | 60–80 tok/s | Strong short-reply quality | **Best TTFT/quality trade for voice** |
| **Phi-4-mini 3.8 B** | 3.8 B | ~2.5 GB | Comparable to Llama 3.2 3B | Strong reasoning for size | Good alt, only viable ≤8 GB machines |
| Qwen 3 4B | 4 B | ~2.5 GB | Fast, good tool-calling | Strong | Good pick if you ever add tools |
| Qwen 3 7B | 7 B | ~4 GB | 40–60 tok/s on M3/M4 | Noticeably better reasoning | Use if you have 16+ GB RAM |
| Mistral Small 3 7B | 7 B | ~4 GB | Highest tok/s on mid-range | Strong all-round | Alt to Qwen 3 7B |
| Llama 3.3 8B | 8 B | ~5 GB | 35–50 tok/s | Best 7–8 B all-rounder | Use if quality > latency |
| Qwen 3 14B (Q4) | 14 B | ~9 GB | 35–50 tok/s on M4 Max | Best quality-to-speed at this tier | Only on 32 GB+ machines |

**TTFT note:** first inference after a cold model triggers Metal shader compilation (adds several seconds). Keep the model warm with a keep-alive ping or Ollama's `--keepalive`.

**Recommendation:** upgrade current `gemma3:1b` → **`llama3.2:3b`** for the best quality-per-latency. `qwen3:4b` is an equally valid alternative if tool-calling is on the roadmap. Stay on Ollama (MLX backend now shipping).

---

## 4. Text-to-Speech (TTS)

| Model | Params | MOS | Latency | Voice cloning | Runtime | Fit |
|---|---|---|---|---|---|---|
| **Kokoro-82M** (current) | 82 M | 4.2 | **40–70 ms/sentence on GPU, ~200 ms on CPU** | No | MLX / PyTorch | **Keep** — best speed/quality balance for voice-assistant sentences |
| Chatterbox (alt, Resemble AI) | ~500 M | 4.0 | <200 ms | Yes | PyTorch (MPS) | Use if voice-cloning is needed; heavier |
| Piper | ~60 M | 3.8 | Fast on CPU/Pi | No | ONNX | Very lightweight fallback; lower naturalness |
| XTTS-v2 (Coqui) | ~500 M | 4.0 | Moderate | Yes (6s sample, 17 langs) | PyTorch | Use if cloning + multilingual needed |
| F5-TTS | ~330 M | 4.1 | 7× realtime (33× Fast variant) | Yes (zero-shot) | PyTorch | Use if expressive cloning matters |
| CosyVoice2-0.5B | 500 M | ~4.1 | Streaming-optimized | Yes | PyTorch | Strong streaming TTS; heavier than Kokoro |
| Fish Speech V1.5 | ~1 B | ~4.2 | Moderate | Yes | PyTorch | Top quality, heavier |
| IndexTTS-2 | — | 4.2 | Moderate | Yes | PyTorch | Top quality, heavier |

**Recommendation:** **keep Kokoro-82M**. For this use case (fixed assistant voice, latency-critical), Kokoro is already the right answer — no open-source model beats its latency at this MOS level. Revisit only if you add voice cloning (→ F5-TTS or XTTS-v2).

**Latency win still available on Kokoro:** stream TTS in sentence-sized chunks (already partially done in `main.py:284-295`) but send each synthesized chunk as a separate `tts_audio` frame instead of concatenating (see separate latency analysis).

---

## 5. Integrated recommendation

Priority-ordered changes, biggest perceived-latency/quality gain first:

1. **STT → Parakeet-TDT 0.6B v2 via `parakeet-mlx`.** Biggest accuracy jump (~10–12% → ~1.7% WER) and large speed jump on Apple Silicon. Falls back cleanly to faster-whisper if MLX init fails.
2. **VAD → Silero VAD as a first-class gate** in the WebSocket audio loop (before STT). Kills "Thank you." hallucinations at source; improves turn-end detection vs the fixed `stream_llm_silence_ms` debounce.
3. **LLM → `llama3.2:3b`** (or `qwen3:4b`). Better replies for ~2× RAM, still fast enough for voice turn latency on Apple Silicon.
4. **TTS → stay on Kokoro-82M**, but stream per-sentence audio chunks instead of concatenating the full reply.

Keep the swap behind the existing `tts_backend`-style config toggle so each layer is one-env-var switchable.

---

## Sources

- [Northflank — Best open-source STT in 2026 (benchmarks)](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks)
- [SiliconFlow — Fastest open-source speech recognition models in 2026](https://www.siliconflow.com/articles/en/fastest-open-source-speech-recognition-models)
- [SiliconFlow — Best open-source models for real-time transcription](https://www.siliconflow.com/articles/en/best-open-source-models-for-real-time-transcription)
- [Ionio — 2025 Edge STT benchmark: Whisper vs. competitors](https://www.ionio.ai/blog/2025-edge-speech-to-text-model-benchmark-whisper-vs-competitors)
- [Moonshine — GitHub](https://github.com/moonshine-ai/moonshine)
- [Moonshine v2 paper — Ergodic Streaming Encoder](https://arxiv.org/abs/2410.15608)
- [parakeet-mlx — Apple Silicon implementation](https://github.com/senstella/parakeet-mlx)
- [MacParakeet — Whisper vs Parakeet on Apple Silicon](https://macparakeet.com/blog/whisper-to-parakeet-neural-engine/)
- [faster-whisper — GitHub (SYSTRAN)](https://github.com/SYSTRAN/faster-whisper)
- [BentoML — Best open-source TTS models in 2026](https://bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models)
- [DigitalOcean — F5-TTS, Kokoro, SparkTTS, Sesame CSM](https://www.digitalocean.com/community/tutorials/best-text-to-speech-models)
- [ocdevel — ElevenLabs alternatives: open-source TTS comparison](https://ocdevel.com/blog/20250720-tts)
- [Inferless — 12 best open-source TTS models compared](https://www.inferless.com/learn/comparing-different-text-to-speech---tts--models-part-2)
- [SitePoint — Best local LLM models 2026](https://www.sitepoint.com/best-local-llm-models-2026/)
- [SitePoint — Local LLMs on Apple Silicon Mac 2026](https://www.sitepoint.com/local-llms-apple-silicon-mac-2026/)
- [apxml — Best local LLMs on every Apple Silicon Mac in 2026](https://apxml.com/posts/best-local-llms-apple-silicon-mac)
- [Picovoice — Best VAD in 2026: Cobra vs Silero vs WebRTC](https://picovoice.ai/blog/best-voice-activity-detection-vad/)
- [Picovoice — Complete 2026 guide to VAD](https://picovoice.ai/blog/complete-guide-voice-activity-detection-vad/)
- [Silero VAD — GitHub](https://github.com/snakers4/silero-vad)
- [Pyannote VAD benchmarks (issue #604)](https://github.com/pyannote/pyannote-audio/issues/604)
