"""
stt.py — Minimal live microphone speech-to-text demo.

What this teaches:
  - Capture microphone audio in short chunks
  - Write one temporary WAV file per chunk
  - Run faster-whisper on each chunk and print the transcript with timing

Usage (from repo root):
  uv run --project backend python scripts/stt.py
"""

import queue
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

MODEL_SIZE = "small"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
SAMPLE_RATE = 16_000
CHUNK_SECONDS = 2.5
LANGUAGE = None
VAD_FILTER = True


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm16.tobytes())


def transcribe_chunk(model: WhisperModel, audio: np.ndarray) -> tuple[str, float]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        write_wav(temp_path, audio)
        started_at = time.perf_counter()
        segments, _ = model.transcribe(
            str(temp_path),
            language=LANGUAGE,
            beam_size=1,
            vad_filter=VAD_FILTER,
        )
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
        return text, elapsed_ms
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> None:
    print("Loading faster-whisper model...")
    load_started_at = time.perf_counter()
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    load_ms = round((time.perf_counter() - load_started_at) * 1000, 1)
    print(f"Model ready in {load_ms} ms")
    print("Listening. Press Ctrl+C to stop.\n")

    audio_queue: queue.Queue[np.ndarray] = queue.Queue()

    def on_audio(indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        del frames, time_info
        if status:
            print(f"[audio] {status}")
        audio_queue.put(indata.copy().reshape(-1))

    buffered_chunks: list[np.ndarray] = []
    target_samples = int(SAMPLE_RATE * CHUNK_SECONDS)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=2048,
            callback=on_audio,
        ):
            while True:
                buffered_chunks.append(audio_queue.get())
                buffered_audio = np.concatenate(buffered_chunks)
                if buffered_audio.size < target_samples:
                    continue

                text, transcribe_ms = transcribe_chunk(model, buffered_audio[:target_samples])
                buffered_chunks = [buffered_audio[target_samples:]] if buffered_audio.size > target_samples else []

                transcript = text or "[no speech detected]"
                print(f"{transcript}  ({transcribe_ms} ms)")
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
