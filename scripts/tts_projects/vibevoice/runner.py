"""
runner.py — VibeVoice standalone demo with timing output.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Final


MODEL_ID: Final[str] = "microsoft/VibeVoice-Realtime-0.5B"
SPEAKER: Final[str] = "Emma"
CFG_SCALE: Final[str] = "1.5"
DEFAULT_TEXT: Final[str] = "Happy to help. Tell me what you need, and we will sort it out together."
DEVICE: Final[str] = "cpu"


def scripts_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def cache_dir() -> Path:
    return scripts_dir() / ".cache"


def repo_dir() -> Path:
    return cache_dir() / "VibeVoice"


def default_output_path() -> Path:
    return scripts_dir() / "speech" / "vibevoice.wav"


def ensure_repo() -> Path:
    path = repo_dir()
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/microsoft/VibeVoice.git", str(path)],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "Could not clone VibeVoice."
        raise RuntimeError(message)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--output", default=str(default_output_path()))
    args = parser.parse_args()

    repo_path = ensure_repo()
    demo_script = repo_path / "demo" / "realtime_model_inference_from_file.py"
    output_dir = Path(args.output).resolve().parent

    if not demo_script.exists():
        print(json.dumps({"status": "error", "latency_ms": "n/a", "voice": f"speaker={SPEAKER}", "output": str(Path(args.output).resolve()), "notes": f"Missing demo script: {demo_script}"}))
        return

    print(f"Model  : {MODEL_ID}")
    print(f"Speaker: {SPEAKER}")
    print(f"Repo   : {repo_path}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        tmp.write(args.text)
        text_file = Path(tmp.name)

    try:
        command = [
            sys.executable,
            str(demo_script),
            "--model_path",
            MODEL_ID,
            "--txt_path",
            str(text_file),
            "--speaker_name",
            SPEAKER,
            "--output_dir",
            str(output_dir),
            "--device",
            DEVICE,
            "--cfg_scale",
            CFG_SCALE,
        ]

        started_at = time.perf_counter()
        completed = subprocess.run(command, cwd=repo_path, capture_output=True, text=True)
        latency_ms = round((time.perf_counter() - started_at) * 1000, 1)
    finally:
        text_file.unlink(missing_ok=True)

    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "VibeVoice failed."
        print(json.dumps({"status": "error", "latency_ms": "n/a", "voice": f"speaker={SPEAKER}", "output": str(Path(args.output).resolve()), "notes": message}))
        return

    generated_path = output_dir / f"{text_file.stem}_generated.wav"
    if not generated_path.exists():
        print(json.dumps({"status": "error", "latency_ms": "n/a", "voice": f"speaker={SPEAKER}", "output": str(Path(args.output).resolve()), "notes": f"Expected output not found: {generated_path}"}))
        return

    final_output = Path(args.output).resolve()
    shutil.move(str(generated_path), final_output)

    print(json.dumps({"status": "ok", "latency_ms": latency_ms, "voice": f"speaker={SPEAKER}", "output": str(final_output), "notes": f"Ran via official realtime demo script on {DEVICE}."}))


if __name__ == "__main__":
    main()
