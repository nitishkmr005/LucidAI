"""
agent.py — Full voice agent pipeline: Audio → STT → LLM → TTS.

What this teaches:
  - How to chain STT, LLM, and TTS into a single synchronous pipeline
  - Measuring latency at each stage independently
  - The basic loop of a voice AI agent

Prerequisites:
  - Ollama running: ollama serve && ollama pull llama3.2
  - (Optional) A WAV file to transcribe; falls back to demo sine wave

Usage (from repo root):
  uv run --project backend python scripts/agent.py path/to/audio.wav
  uv run --project backend python scripts/agent.py   # uses demo audio
"""

import sys
import time
from pathlib import Path


def run_pipeline(audio_path: Path) -> None:
    from faster_whisper import WhisperModel  # type: ignore
    import ollama  # type: ignore

    MODEL_SIZE = "small.en"
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"
    OLLAMA_HOST = "http://localhost:11434"
    LLM_MODEL = "llama3.2"
    SYSTEM_PROMPT = (
        "You are a concise voice assistant for customer service. "
        "Respond in 1-3 sentences only. Plain spoken language — no markdown."
    )

    print(f"\n{'═'*52}")
    print("  NeuroTalk Voice Agent Pipeline")
    print(f"{'═'*52}")

    # ── Stage 1: STT ────────────────────────────────────────
    print(f"\n[1/3] STT  →  {audio_path.name}")
    t0 = time.perf_counter()
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    segments, _ = model.transcribe(str(audio_path), beam_size=1, vad_filter=True)
    transcript = " ".join(seg.text.strip() for seg in segments)
    stt_ms = round((time.perf_counter() - t0) * 1000, 1)
    print(f"    Transcript : \"{transcript or '[no speech]'}\"  ({stt_ms} ms)")

    if not transcript.strip():
        print("\n  [stop] No speech detected — skipping LLM and TTS.")
        return

    # ── Stage 2: LLM ────────────────────────────────────────
    print(f"\n[2/3] LLM  →  {LLM_MODEL} via Ollama")
    client = ollama.Client(host=OLLAMA_HOST)
    t1 = time.perf_counter()
    first_token_at: float | None = None
    llm_response = ""

    print("    Response   : ", end="", flush=True)
    for chunk in client.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
        stream=True,
    ):
        token = chunk["message"]["content"]
        if token:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            print(token, end="", flush=True)
            llm_response += token

    llm_ms = round((time.perf_counter() - t1) * 1000, 1)
    ttft_ms = round((first_token_at - t1) * 1000, 1) if first_token_at else 0
    print(f"\n    TTFT={ttft_ms} ms  total={llm_ms} ms")

    # ── Stage 3: TTS ────────────────────────────────────────
    print(f"\n[3/3] TTS  →  speaking response")
    import platform, subprocess  # noqa: E401
    t2 = time.perf_counter()
    system = platform.system()
    if system == "Darwin":
        subprocess.run(["say", llm_response], check=True)
    elif system == "Linux":
        subprocess.run(["espeak", llm_response], check=True)
    else:
        print(f"    [text] {llm_response}")
    tts_ms = round((time.perf_counter() - t2) * 1000, 1)
    print(f"    TTS latency: {tts_ms} ms")

    # ── Summary ─────────────────────────────────────────────
    total_ms = round((time.perf_counter() - t0) * 1000, 1)
    print(f"\n{'─'*52}")
    print(f"  STT  {stt_ms:>7} ms")
    print(f"  LLM  {llm_ms:>7} ms  (TTFT {ttft_ms} ms)")
    print(f"  TTS  {tts_ms:>7} ms")
    print(f"  ─────────────")
    print(f"  Total{total_ms:>7} ms")
    print(f"{'═'*52}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio = Path(sys.argv[1])
        if not audio.exists():
            print(f"Error: file not found → {audio}")
            sys.exit(1)
    else:
        # Generate demo audio inline
        import wave, struct, math  # noqa: E401
        audio = Path("/tmp/neurotalk_agent_demo.wav")
        sr = 16000
        with wave.open(str(audio), "w") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sr)
            frames = [struct.pack("<h", int(32767 * math.sin(2 * math.pi * 440 * i / sr))) for i in range(sr * 2)]
            f.writeframes(b"".join(frames))
        print(f"[demo] Generated 2 s sine wave → {audio}")

    try:
        run_pipeline(audio)
    except Exception as e:
        print(f"\n[error] {e}")
        print("Check: ollama serve && ollama pull llama3.2")
        sys.exit(1)
