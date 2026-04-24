"use client";

import { useEffect, useMemo, useState, type CSSProperties } from "react";
import {
  buildMiniWaveBars,
  buildCircularWaveBars,
  getStateIntensity,
  type VoiceState,
  voiceStateCopy,
} from "./waveform-utils";

type VoiceOrbHeroProps = {
  state: VoiceState;
  audioLevel?: number;
  className?: string;
  disabled?: boolean;
  onActivate?: () => void;
};

const SPEC_BARS = buildCircularWaveBars(120);
const MINI_BARS = buildMiniWaveBars(26);

export function VoiceOrbHero({
  state,
  audioLevel = 0,
  className,
  disabled = false,
  onActivate,
}: VoiceOrbHeroProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const intensity = mounted ? Math.min(1, Math.max(0, audioLevel)) : 0.05;
  const stateIntensity = getStateIntensity(state);
  const effectiveEnergy = useMemo(
    () => Math.min(1, 0.12 + stateIntensity * 0.35 + intensity * 0.85),
    [intensity, stateIntensity],
  );
  const copy = voiceStateCopy[state];

  return (
    <div
      className={["nt-orb", className].filter(Boolean).join(" ")}
      data-voice-state={state}
      style={
        {
          "--nt-orb-energy": effectiveEnergy.toFixed(3),
          "--nt-orb-audio": intensity.toFixed(3),
        } as CSSProperties
      }
    >
      <button
        type="button"
        className="nt-orb__button"
        onClick={onActivate}
        disabled={disabled}
        aria-label={
          state === "speaking"
            ? "Stop conversation"
            : state === "idle"
              ? "Start conversation"
              : "Toggle voice"
        }
      >
        {/* Ambient glow halos */}
        <span className="nt-orb__glow nt-orb__glow--outer" aria-hidden="true" />
        <span className="nt-orb__glow nt-orb__glow--inner" aria-hidden="true" />

        {/* Concentric pulse rings */}
        <span className="nt-orb__ring nt-orb__ring--outer" aria-hidden="true" />
        <span className="nt-orb__ring nt-orb__ring--inner" aria-hidden="true" />

        {/* Circular spectrum bars — one arm per bar, bar child extends outward */}
        <span className="nt-orb__spectrum" aria-hidden="true">
          {SPEC_BARS.map((bar, i) => (
            <span
              key={i}
              className="nt-orb__spec-arm"
              style={
                {
                  "--spec-angle": `${bar.angle.toFixed(1)}deg`,
                } as CSSProperties
              }
            >
              <span
                className="nt-orb__spec-bar"
                style={
                  {
                    "--spec-emphasis": bar.emphasis.toFixed(3),
                    "--spec-delay": `${bar.delay.toFixed(3)}s`,
                  } as CSSProperties
                }
              />
            </span>
          ))}
        </span>

        {/* The sphere */}
        <span className="nt-orb__sphere" aria-hidden="true">
          <span className="nt-orb__sphere-highlight" />
          <span className="nt-orb__sphere-core" />
          <span className="nt-orb__sphere-shadow" />
        </span>
      </button>

      <div className="nt-orb__copy">
        <p className="nt-orb__title">{copy.title}</p>
        <p className="nt-orb__subtitle">{copy.subtitle}</p>
        <div className="nt-orb__miniwave" aria-hidden="true">
          {MINI_BARS.map((bar, index) => (
            <span
              key={`${bar.accent}-${index}`}
              className={`nt-orb__miniwave-bar nt-orb__miniwave-bar--${bar.accent}`}
              style={
                {
                  "--mini-height": `${(bar.baseHeight * (0.38 + Math.min(1, intensity) * 2.0)).toFixed(1)}px`,
                  "--mini-delay": `${bar.delay.toFixed(3)}s`,
                } as CSSProperties
              }
            />
          ))}
        </div>
      </div>
    </div>
  );
}

export type { VoiceState };
