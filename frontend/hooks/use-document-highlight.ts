"use client";

import { useRef, useCallback } from "react";

type WordTickCallback = (sentenceIdx: number, wordIdx: number) => void;

type WordTiming = {
  startMs: number;
  endMs: number;
};

function getWordWeight(word: string): number {
  const trimmed = word.trim();
  if (!trimmed) return 1;

  let weight = Math.max(1, trimmed.replace(/[^A-Za-z0-9]/g, "").length * 0.18);
  if (/[,:;]$/.test(trimmed)) weight += 0.45;
  if (/[.!?]$/.test(trimmed)) weight += 0.8;
  if (/[-–—]/.test(trimmed)) weight += 0.18;

  return weight;
}

function buildWordTimings(words: string[], audioDurationMs: number): WordTiming[] {
  const usableDurationMs = Math.max(220, audioDurationMs * 0.92);
  const totalWeight = words.reduce((sum, word) => sum + getWordWeight(word), 0);
  let cursor = 0;

  return words.map((word, index) => {
    const remainingWords = words.length - index;
    const proportionalMs = usableDurationMs * (getWordWeight(word) / Math.max(totalWeight, 1));
    const minDurationMs = Math.min(220, Math.max(75, usableDurationMs / Math.max(words.length, 1) * 0.55));
    const remainingMinMs = (remainingWords - 1) * minDurationMs;
    const durationMs = Math.max(minDurationMs, Math.min(proportionalMs, usableDurationMs - cursor - remainingMinMs));
    const nextCursor = index === words.length - 1 ? usableDurationMs : cursor + durationMs;
    const timing = { startMs: cursor, endMs: nextCursor };
    cursor = nextCursor;
    return timing;
  });
}

export function useDocumentHighlight(onWordTick: WordTickCallback) {
  const rafRef = useRef<number | null>(null);

  const cancelHighlight = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);

  const startWordHighlight = useCallback(
    (sentenceIdx: number, sentenceText: string, audioDurationMs: number) => {
      cancelHighlight();
      const words = sentenceText.trim().split(/\s+/).filter(Boolean);
      if (words.length === 0) return;
      const timings = buildWordTimings(words, audioDurationMs);
      const startTime = performance.now();
      let lastEmittedIdx = -1;

      onWordTick(sentenceIdx, 0);
      lastEmittedIdx = 0;

      const tick = () => {
        const elapsed = performance.now() - startTime;
        let wordIdx = timings.findIndex((timing) => elapsed >= timing.startMs && elapsed < timing.endMs);
        if (wordIdx === -1) {
          wordIdx = elapsed >= timings[timings.length - 1].endMs ? words.length - 1 : 0;
        }

        if (wordIdx !== lastEmittedIdx) {
          lastEmittedIdx = wordIdx;
          onWordTick(sentenceIdx, wordIdx);
        }

        if (elapsed < audioDurationMs) {
          rafRef.current = requestAnimationFrame(tick);
        } else {
          onWordTick(sentenceIdx, words.length - 1);
          rafRef.current = null;
        }
      };

      rafRef.current = requestAnimationFrame(tick);
    },
    [cancelHighlight, onWordTick],
  );

  return { startWordHighlight, cancelHighlight };
}
