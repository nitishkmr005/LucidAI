"""
vad.py — Minimal streaming VAD state machine for learning.

What this teaches:
  - VAD is not just "speech or no speech"
  - A stream of frame scores becomes speech-start and speech-end events
  - Hysteresis and silence debounce prevent noisy toggling

Usage:
  uv run --project backend python scripts/vad.py
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass

from loguru import logger


def setup_logging() -> None:
    logger.remove()
    logger.add(
        sink=sys.stdout,
        level="INFO",
        colorize=True,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <7}</level> | "
            "<cyan>{extra[step]}</cyan> | "
            "<level>{message}</level>\n"
        ),
    )


def log(step: str, message: str, *args: object) -> None:
    logger.bind(step=step).info(message, *args)


@dataclass(frozen=True)
class VADStreamEvent:
    event: str
    frame_index: int
    speech_prob: float


class StreamingVADDemo:
    """
    Tiny version of the production VAD state machine.

    Input:
      one speech probability per frame

    Output:
      "start" and "end" events that the rest of the app can react to
    """

    def __init__(self, threshold: float = 0.60, min_silence_frames: int = 2) -> None:
        self.threshold = threshold
        self.neg_threshold = max(threshold - 0.15, 0.05)
        self.min_silence_frames = min_silence_frames
        self.in_speech = False
        self.pending_silence_frames = 0

    def process_frame(self, frame_index: int, speech_prob: float) -> list[VADStreamEvent]:
        events: list[VADStreamEvent] = []

        # When speech is confidently detected, clear pending silence and maybe
        # emit a speech-start event.
        if speech_prob >= self.threshold:
            self.pending_silence_frames = 0
            if not self.in_speech:
                self.in_speech = True
                events.append(VADStreamEvent("start", frame_index, speech_prob))
            return events

        # Once we are inside a speech segment, we do not stop immediately on the
        # first quiet frame. We wait a bit so pauses inside a sentence do not
        # break the turn into many fragments.
        if self.in_speech and speech_prob < self.neg_threshold:
            self.pending_silence_frames += 1
            if self.pending_silence_frames >= self.min_silence_frames:
                self.in_speech = False
                self.pending_silence_frames = 0
                events.append(VADStreamEvent("end", frame_index, speech_prob))

        return events


def run_demo() -> None:
    vad = StreamingVADDemo(threshold=0.60, min_silence_frames=2)

    # Simulated frame-by-frame speech probabilities.
    # Low values mean silence. High values mean the VAD model is confident
    # there is speech in the current frame.
    demo_probabilities = [
        0.03, 0.05, 0.08,
        0.66, 0.82, 0.91, 0.74,
        0.49,
        0.41, 0.35,
        0.78, 0.88,
        0.44, 0.32, 0.10,
    ]

    log("config", "threshold={} neg_threshold={} min_silence_frames={}", vad.threshold, vad.neg_threshold, vad.min_silence_frames)

    for frame_index, speech_prob in enumerate(demo_probabilities):
        state = "speech" if vad.in_speech else "silence"
        log("frame", "frame={} prob={:.2f} current_state={}", frame_index, speech_prob, state)

        events = vad.process_frame(frame_index, speech_prob)
        for event in events:
            log("event", "{}", asdict(event))

    final_state = "speech" if vad.in_speech else "silence"
    log("result", "final_state={}", final_state)
    log("result", "If you increase min_silence_frames, end events happen later.")
    log("result", "If you lower threshold, speech starts earlier but false positives increase.")


if __name__ == "__main__":
    setup_logging()
    run_demo()
