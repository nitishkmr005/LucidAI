export type VoiceState = "idle" | "listening" | "thinking" | "speaking" | "interrupted";

export type CircularWaveBar = {
  angle: number;
  emphasis: number;
  baseHeight: number;
  delay: number;
  side: "left" | "right";
};

export type MiniWaveBar = {
  baseHeight: number;
  delay: number;
  accent: "blue" | "purple";
};

export const voiceStateCopy: Record<VoiceState, { title: string; subtitle: string }> = {
  idle: {
    title: "Tap to speak",
    subtitle: "Microphone is muted. Tap the orb to start.",
  },
  listening: {
    title: "Listening...",
    subtitle: "Speak naturally, I'm here to help.",
  },
  thinking: {
    title: "Thinking...",
    subtitle: "Understanding your request.",
  },
  speaking: {
    title: "Speaking...",
    subtitle: "You can interrupt anytime.",
  },
  interrupted: {
    title: "Listening...",
    subtitle: "I stopped. Go ahead.",
  },
};

export function buildCircularWaveBars(totalBars = 120): CircularWaveBar[] {
  return Array.from({ length: totalBars }, (_, index) => {
    const angle = (index / totalBars) * 360;
    const radians = (angle * Math.PI) / 180;
    const sideBias = Math.pow(Math.abs(Math.cos(radians)), 3.4);
    const verticalFalloff = 1 - Math.pow(Math.abs(Math.sin(radians)), 1.6);
    const emphasis = Math.max(0.06, sideBias * verticalFalloff);
    const wobble = 0.72 + 0.28 * Math.sin(index * 0.61);
    return {
      angle,
      emphasis,
      baseHeight: 10 + emphasis * 62 * wobble,
      delay: index * 0.018,
      side: angle > 90 && angle < 270 ? "left" : "right",
    };
  });
}

export function buildMiniWaveBars(totalBars = 28): MiniWaveBar[] {
  return Array.from({ length: totalBars }, (_, index) => {
    const arc = Math.sin((index / Math.max(totalBars - 1, 1)) * Math.PI);
    const pulse = 0.72 + 0.28 * Math.cos(index * 0.92);
    return {
      baseHeight: 8 + arc * 14 * pulse,
      delay: index * 0.045,
      accent: index % 2 === 0 ? "blue" : "purple",
    };
  });
}

export function getStateIntensity(state: VoiceState): number {
  switch (state) {
    case "speaking":
      return 1;
    case "thinking":
      return 0.56;
    case "interrupted":
      return 0.38;
    case "idle":
      return 0.08;
    case "listening":
    default:
      return 0.72;
  }
}
