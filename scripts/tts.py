"""
tts.py — Compare 4 TTS models from one command.

Run from the repo root:
  python3 scripts/tts.py

Run from the scripts directory:
  python3 tts.py

This controller creates isolated uv environments under scripts/tts_projects/,
runs one model-local runner per project, then prints one final comparison table.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final


TEXT_OPTIONS: Final[dict[str, str]] = {
    "neutral_intro": "Hello! I am your voice assistant. How can I help you today?",
    "friendly_help": "Happy to help. Tell me what you need, and we will sort it out together.",
    "excited_launch": "We shipped it. The new build is live, and everything is finally working.",
    "playful_joke": "That meeting could have been an email, and honestly, maybe a short one.",
    "dramatic_warning": "Something is wrong. The lights are on, the logs are moving, and yet the whole system feels cursed.",
}

TEXT: Final[str] = TEXT_OPTIONS["friendly_help"]
RUN_MODELS: Final[list[str]] = ["chatterbox", "qwen", "vibevoice", "kokoro"]
FORCE_SYNC: Final[bool] = False

VOICE_PATHS: Final[dict[str, str]] = {
    "agent": "/tmp/voice_agent.wav",
    "warm": "/tmp/voice_warm.wav",
    "energetic": "/tmp/voice_energetic.wav",
    "calm": "/tmp/voice_calm.wav",
}


@dataclass
class ModelConfig:
    key: str
    name: str
    project_dir: str
    runner: str
    params: str
    disk: str
    voice: str
    official_latency: str


SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent
OUTPUT_DIR: Final[Path] = SCRIPT_DIR / "speech"
PROJECTS_DIR: Final[Path] = SCRIPT_DIR / "tts_projects"
REPORT_PATH: Final[Path] = OUTPUT_DIR / "tts_report.md"
REPORT_JSON_PATH: Final[Path] = OUTPUT_DIR / "tts_report.json"

MODELS: Final[dict[str, ModelConfig]] = {
    "chatterbox": ModelConfig(
        key="chatterbox",
        name="Chatterbox Turbo",
        project_dir="chatterbox",
        runner="runner.py",
        params="350M",
        disk="4.04 GB",
        voice="built-in default or /tmp/voice_agent.wav",
        official_latency="sub-200ms service / local runtime-dependent",
    ),
    "qwen": ModelConfig(
        key="qwen",
        name="Qwen3-TTS 0.6B CustomVoice",
        project_dir="qwen",
        runner="runner.py",
        params="0.6B",
        disk="2.5 GB",
        voice="speaker=Ryan",
        official_latency="as low as 97 ms",
    ),
    "vibevoice": ModelConfig(
        key="vibevoice",
        name="VibeVoice Realtime 0.5B",
        project_dir="vibevoice",
        runner="runner.py",
        params="0.5B",
        disk="2.04 GB",
        voice="speaker=Emma",
        official_latency="~200 ms first audio",
    ),
    "kokoro": ModelConfig(
        key="kokoro",
        name="Kokoro 82M MLX",
        project_dir="kokoro",
        runner="runner.py",
        params="82M",
        disk="355 MB",
        voice="af_heart",
        official_latency="fast local MLX runtime",
    ),
}


@dataclass
class RunResult:
    model: str
    status: str
    latency_ms: str
    params: str
    disk: str
    voice: str
    output: str
    notes: str


def clean_text(value: str) -> str:
    return " ".join(value.split())


def short_path(value: str) -> str:
    if value in {"", "-"}:
        return value

    try:
        path = Path(value)
    except Exception:
        return value

    if not path.is_absolute():
        return value

    try:
        return str(path.relative_to(SCRIPT_DIR.parent))
    except ValueError:
        return value


def trim_block(value: str, max_lines: int = 14) -> str:
    lines = value.strip().splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    trimmed = lines[:max_lines]
    trimmed.append("...")
    return "\n".join(trimmed)


def venv_python(project_path: Path) -> Path:
    if sys.platform == "win32":
        return project_path / ".venv" / "Scripts" / "python.exe"
    return project_path / ".venv" / "bin" / "python"


def ensure_env(project_path: Path) -> None:
    python_path = venv_python(project_path)
    if python_path.exists() and not FORCE_SYNC:
        return

    command = ["uv", "sync", "--project", str(project_path)]
    completed = subprocess.run(command, cwd=SCRIPT_DIR, capture_output=True, text=True)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or f"uv sync failed for {project_path.name}"
        raise RuntimeError(message)


def run_model(config: ModelConfig) -> RunResult:
    project_path = PROJECTS_DIR / config.project_dir
    runner_path = project_path / config.runner
    output_path = OUTPUT_DIR / f"{config.key}.wav"
    output_label = str(output_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        ensure_env(project_path)
    except RuntimeError as error:
        return RunResult(config.name, "error", "n/a", config.params, config.disk, config.voice, output_label, str(error))

    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)

    command = [
        "uv",
        "run",
        "--project",
        str(project_path),
        "python",
        str(runner_path),
        "--text",
        TEXT,
        "--output",
        str(output_path),
    ]

    completed = subprocess.run(command, cwd=SCRIPT_DIR, capture_output=True, text=True, env=env)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or f"{config.name} failed."
        return RunResult(config.name, "error", "n/a", config.params, config.disk, config.voice, output_label, message)

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return RunResult(config.name, "error", "n/a", config.params, config.disk, config.voice, output_label, "Runner returned no output.")

    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return RunResult(config.name, "error", "n/a", config.params, config.disk, config.voice, output_label, "Runner output was not valid JSON.")

    return RunResult(
        model=config.name,
        status=payload.get("status", "unknown"),
        latency_ms=str(payload.get("latency_ms", "n/a")),
        params=config.params,
        disk=config.disk,
        voice=str(payload.get("voice", config.voice)),
        output=str(payload.get("output", output_label)),
        notes=str(payload.get("notes", "")),
    )


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def build_row(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) + " |"

    divider = "|-" + "-|-".join("-" * width for width in widths) + "-|"
    lines = [build_row(headers), divider]
    lines.extend(build_row(row) for row in rows)
    return "\n".join(lines)


def write_report(text: str, results: list[RunResult], table: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_sections: list[str] = []
    for result in results:
        detail_sections.extend(
            [
                f"## {result.model}",
                "",
                f"- Status: `{result.status}`",
                f"- Latency: `{result.latency_ms} ms`" if result.latency_ms != "n/a" else "- Latency: `n/a`",
                f"- Output: `{short_path(result.output)}`",
                f"- Voice: `{result.voice}`",
                "",
                "Notes:",
                "```text",
                trim_block(result.notes or "None"),
                "```",
                "",
            ]
        )

    REPORT_PATH.write_text(
        "\n".join(
            [
                "# TTS Comparison Report",
                "",
                f"Text: `{text}`",
                "",
                "## Summary",
                "",
                table,
                "",
                "## Details",
                "",
                *detail_sections,
            ]
        )
    )
    REPORT_JSON_PATH.write_text(
        json.dumps(
            {
                "text": text,
                "results": [result.__dict__ for result in results],
            },
            indent=2,
        )
    )


def main() -> None:
    print(f"\nText: {TEXT!r}\n")
    results: list[RunResult] = []

    for model_key in RUN_MODELS:
        config = MODELS[model_key]
        print(f"Running {config.name}...")
        result = run_model(config)
        results.append(result)
        print(f"  status={result.status} latency_ms={result.latency_ms} notes={result.notes}")

    print("\nLatency comparison\n")
    def ms_to_s(value: str) -> str:
        try:
            return f"{float(value) / 1000:.2f}s"
        except (ValueError, TypeError):
            return value

    rows = [
        [
            result.model,
            result.status,
            f"{result.latency_ms} ms ({ms_to_s(result.latency_ms)})" if result.latency_ms != "n/a" else "n/a",
            MODELS[model_key].official_latency,
            result.params,
            result.disk,
            short_path(result.output),
        ]
        for model_key, result in zip(RUN_MODELS, results, strict=True)
    ]
    table = format_table(
        ["model", "status", "latency", "official_latency", "params", "disk", "output"],
        rows,
    )
    print(table)
    write_report(TEXT, results, table)
    print(f"\nReport saved to: {REPORT_PATH}")
    print(f"JSON saved to: {REPORT_JSON_PATH}")


if __name__ == "__main__":
    main()
