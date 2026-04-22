"""
llm_call.py — Standalone LLM call demo using Ollama (local inference).

What this teaches:
  - Connecting to a local Ollama server
  - Streaming tokens from an LLM in real-time
  - Measuring Time-to-First-Token (TTFT) and total generation latency
  - System prompt vs. user message structure

Prerequisites:
  brew install ollama
  ollama pull llama3.2      # ~2 GB download
  ollama serve              # runs at http://localhost:11434

Usage (from repo root):
  uv run --project backend python scripts/llm_call.py "Hello, can you help me?"
  uv run --project backend python scripts/llm_call.py   # uses a default prompt
"""

import sys
import time

OLLAMA_HOST = "http://localhost:11434"
MODEL = "gemma4:latest"#"llama3.2"
SYSTEM_PROMPT = (
    "You are a concise voice assistant for customer service. "
    "Respond in 1-3 sentences only. Plain spoken language — no markdown."
)

DEFAULT_TRANSCRIPT = "What all capabilities do you have as a voice assistant?"


def stream_response(transcript: str) -> None:
    import ollama  # type: ignore

    print(f"\n{'─'*50}")
    print(f"  Model   : {MODEL}  host={OLLAMA_HOST}")
    print(f"  Prompt  : {transcript!r}")
    print(f"{'─'*50}")
    print("  Response: ", end="", flush=True)

    client = ollama.Client(host=OLLAMA_HOST)
    first_token_at: float | None = None
    full_text = ""
    t0 = time.perf_counter()

    for chunk in client.chat(
        model=MODEL,
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
            full_text += token

    total_ms = round((time.perf_counter() - t0) * 1000, 1)
    ttft_ms = round((first_token_at - t0) * 1000, 1) if first_token_at else 0

    print(f"\n\n  TTFT  : {ttft_ms} ms")
    print(f"  Total : {total_ms} ms  ({len(full_text)} chars)\n")


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_TRANSCRIPT
    try:
        stream_response(prompt)
    except Exception as e:
        print(f"\n[error] {e}")
        print(f"Make sure Ollama is running: ollama serve && ollama pull {MODEL}")
        sys.exit(1)
