"use client";

import { Fragment, startTransition, useCallback, useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { WebRTCTransport } from "./webrtc-transport";
import { useDocumentHighlight } from "../hooks/use-document-highlight";
import type { DocumentEvent } from "./document-panel";

type Mode = "listening" | "thinking" | "responding" | "speaking";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  isStreaming: boolean;
  isError: boolean;
};

type Metrics = {
  request_read_ms: number;
  file_write_ms: number;
  model_load_ms: number;
  transcribe_ms: number;
  total_ms: number;
  buffered_audio_ms: number;
  client_roundtrip_ms: number | null;
};

type DebugInfo = {
  request_id: string;
  filename: string;
  audio_bytes: number;
  detected_language: string | null;
  segments: number;
  model_size: string;
  device: string;
  compute_type: string;
  sample_rate: number | null;
  chunks_received: number | null;
};

type TtsVoice = {
  id: string;
  name: string;
};

type StreamMessage = {
  type: "ready" | "partial" | "final" | "error" | "llm_start" | "llm_partial" | "llm_final" | "llm_error" | "tts_start" | "tts_audio" | "tts_done" | "tts_interrupted"
      | "doc_list" | "doc_read_start" | "doc_opened" | "doc_highlight" | "doc_save_snippet"
      | "doc_search_start" | "doc_search_result" | "doc_export" | "doc_reading_pause" | "doc_reading_resume"
      | "doc_error" | "doc_highlight_saved" | "doc_note_saved" | "doc_list_requested";
  request_id?: string;
  text?: string;
  user_text?: string;
  message?: string;
  timings_ms?: Metrics;
  debug?: DebugInfo;
  llm_ms?: number;
  data?: string;
  tts_ms?: number;
  sentence_text?: string;
  // document fields
  documents?: unknown[];
  doc_id?: string;
  sentences?: string[];
  title?: string;
  raw_markdown?: string;
  annotations?: unknown;
  snippet?: unknown;
  sentence_idx?: number;
  word_count?: number;
  color?: string;
  term?: string;
  query?: string;
  results?: unknown[];
  format?: string;
  download_url?: string;
};

const modeConfig: Record<
  Mode,
  {
    eyebrow: string;
    headline: string;
    summary: string;
    accent: string;
  }
> = {
  listening: {
    eyebrow: "Listening",
    headline: "Listening",
    summary: "Realtime speech capture is ready.",
    accent: "active-listening",
  },
  thinking: {
    eyebrow: "Processing",
    headline: "Processing",
    summary: "Speech is being transcribed and routed.",
    accent: "deep-reasoning",
  },
  responding: {
    eyebrow: "Responding",
    headline: "Responding",
    summary: "The answer is being prepared.",
    accent: "voice-delivery",
  },
  speaking: {
    eyebrow: "Speaking",
    headline: "Speaking",
    summary: "Voice playback is active.",
    accent: "voice-delivery",
  },
};

const waveformHeights = [28, 46, 32, 64, 24, 58, 38, 72, 44, 30, 66, 35, 54, 26, 60, 40];
const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";
const websocketUrl = `${backendUrl.replace(/^http/, "ws")}/ws/transcribe`;
const initialWaveLevels = waveformHeights.map(() => 0.18);
const BARGE_IN_THRESHOLD = 0.15;
const BARGE_IN_FRAMES = 1;
const fallbackTtsVoiceIds = [
  "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
  "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa",
  "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
  "ef_dora", "em_alex", "em_santa", "ff_siwis", "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
  "if_sara", "im_nicola", "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
  "pf_dora", "pm_alex", "pm_santa", "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
  "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
];
const defaultTtsVoice = "af_heart";
const ttsVoiceStorageKey = "neurotalk-tts-voice";

function formatVoiceName(voiceId: string): string {
  const [prefix, rawName] = voiceId.split("_");
  const accent = ({ a: "American", b: "British", e: "Spanish", f: "French", h: "Hindi", i: "Italian", j: "Japanese", p: "Portuguese", z: "Mandarin" } as Record<string, string>)[prefix?.[0] ?? ""] ?? prefix?.[0]?.toUpperCase() ?? "";
  const gender = ({ f: "Female", m: "Male" } as Record<string, string>)[prefix?.[1] ?? ""] ?? prefix?.[1]?.toUpperCase() ?? "";
  const name = (rawName ?? voiceId).replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
  return accent && gender ? `${name} (${accent} ${gender})` : name;
}

const FALLBACK_TTS_VOICES: TtsVoice[] = fallbackTtsVoiceIds.map((id) => ({ id, name: formatVoiceName(id) }));

function getVoiceLabel(voice: TtsVoice): { displayName: string; detail: string } {
  const match = voice.name.match(/^(.+?)\s+\((.+)\)$/);
  if (!match) {
    return { displayName: voice.name, detail: "Assistant Voice" };
  }
  return { displayName: match[1], detail: match[2] };
}

function float32ToInt16(input: Float32Array): Int16Array {
  const output = new Int16Array(input.length);
  for (let index = 0; index < input.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, input[index]));
    output[index] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return output;
}

function getRmsAmplitude(input: Float32Array): number {
  let sum = 0;
  for (let index = 0; index < input.length; index += 1) {
    sum += input[index] * input[index];
  }

  return Math.sqrt(sum / input.length);
}

function formatSeconds(valueMs: number | null | undefined, options?: { cachedWhenZero?: boolean }): string {
  if (valueMs === null || valueMs === undefined) {
    return "--";
  }

  if (options?.cachedWhenZero && valueMs <= 0) {
    return "cached";
  }

  return `${(valueMs / 1000).toFixed(valueMs >= 1000 ? 2 : 3)} s`;
}

type TransportType = "websocket" | "webrtc";

type VoiceAgentConsoleProps = {
  children?: ReactNode;
  selectedDocumentId?: string | null;
  onDocumentEvent?: (event: DocumentEvent) => void;
  onSnippetExplanation?: (explanation: string) => void;
};

export function VoiceAgentConsole({ children, selectedDocumentId, onDocumentEvent, onSnippetExplanation }: VoiceAgentConsoleProps = {}) {
  const [isDark, setIsDark] = useState(true);
  const [transportType] = useState<TransportType>("webrtc");
  const [ttsVoices, setTtsVoices] = useState<TtsVoice[]>(FALLBACK_TTS_VOICES);
  const [selectedTtsVoice, setSelectedTtsVoice] = useState<string>(defaultTtsVoice);
  const [isVoiceSettingsOpen, setIsVoiceSettingsOpen] = useState(false);
  const [previewingVoice, setPreviewingVoice] = useState<string | null>(null);
  const [voicePreviewError, setVoicePreviewError] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("listening");
  const [isRecording, setIsRecording] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isFinalizing, setIsFinalizing] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [debugInfo, setDebugInfo] = useState<DebugInfo | null>(null);
  const [amplitude, setAmplitude] = useState(0.08);
  const [waveLevels, setWaveLevels] = useState(initialWaveLevels);
  const [copied, setCopied] = useState(false);
  const [llmLatencyMs, setLlmLatencyMs] = useState<number | null>(null);
  const [ttsLatencyMs, setTtsLatencyMs] = useState<number | null>(null);
  const ttsSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const ttsAudioReceivedRef = useRef(false);
  const interruptSentRef = useRef(false);
  const bargeinFrameCountRef = useRef(0);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const activeUserIdRef = useRef<string | null>(null);
  const activeAssistantIdRef = useRef<string | null>(null);
  const pendingAssistantTextRef = useRef<string>("");
  const revealRafRef = useRef<number | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);

  // Audio queue: sentences arrive one at a time; we play them sequentially.
  type TtsChunk = { buffer: AudioBuffer; text: string; docSentenceIdx?: number };
  const ttsQueueRef = useRef<TtsChunk[]>([]);
  const isTtsPlayingRef = useRef(false);
  const ttsAllChunksReceivedRef = useRef(false);
  // Text revealed so far in the current assistant turn (accumulates across chunks).
  const revealedTextRef = useRef("");
  // Counts tts_audio chunks whose decodeAudioData hasn't fired yet — prevents
  // premature finalisation when tts_done arrives before all decodes complete.
  const pendingDecodesRef = useRef(0);

  // Pending sentence for doc highlight sync
  const pendingDocSentenceIdxRef = useRef<number | null>(null);
  // True while the agent is reading a document aloud (drives auto-continue)
  const isReadingModeRef = useRef(false);

  const { startWordHighlight, cancelHighlight } = useDocumentHighlight(
    useCallback((sentenceIdx: number, wordIdx: number) => {
      onDocumentEvent?.({ type: "tts_word_tick", sentence_idx: sentenceIdx, word_idx: wordIdx });
    }, [onDocumentEvent])
  );

  const websocketRef = useRef<WebSocket | null>(null);
  const webrtcRef = useRef<WebRTCTransport | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorNodeRef = useRef<ScriptProcessorNode | null>(null);
  const gainNodeRef = useRef<GainNode | null>(null);
  const sessionStartedAtRef = useRef<number | null>(null);
  const streamReadyRef = useRef(false);
  const isFinalizingRef = useRef(false);
  const receivedFinalRef = useRef(false);
  const normalCloseRef = useRef(false);
  const isRecordingRef = useRef(false);
  const errorRef = useRef<string | null>(null);
  const amplitudeRef = useRef(0.08);
  const waveLevelsRef = useRef(initialWaveLevels);
  const startAttemptRef = useRef(0);
  const loadedDocumentIdRef = useRef<string | null>(null);
  const previewAudioRef = useRef<HTMLAudioElement | null>(null);
  const previewAudioUrlRef = useRef<string | null>(null);

  useEffect(() => {
    const saved = localStorage.getItem("nt-theme");
    const dark = saved !== "light";
    setIsDark(dark);
    document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");

    const savedVoice = localStorage.getItem(ttsVoiceStorageKey);
    if (savedVoice && FALLBACK_TTS_VOICES.some((voice) => voice.id === savedVoice)) {
      setSelectedTtsVoice(savedVoice);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(ttsVoiceStorageKey, selectedTtsVoice);
  }, [selectedTtsVoice]);

  useEffect(() => {
    let cancelled = false;
    void fetch(`${backendUrl}/tts/voices`)
      .then((response) => response.ok ? response.json() : null)
      .then((payload: { default_voice?: string; voices?: TtsVoice[] } | null) => {
        if (cancelled || !payload?.voices?.length) return;
        setTtsVoices(payload.voices);
        setSelectedTtsVoice((current) => (
          payload.voices?.some((voice) => voice.id === current)
            ? current
            : payload.default_voice ?? defaultTtsVoice
        ));
      })
      .catch(() => undefined);
    return () => { cancelled = true; };
  }, []);

  const toggleTheme = () => {
    const next = !isDark;
    setIsDark(next);
    const value = next ? "dark" : "light";
    localStorage.setItem("nt-theme", value);
    document.documentElement.setAttribute("data-theme", value);
  };

  useEffect(() => {
    errorRef.current = error;
  }, [error]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const stopAudioGraph = () => {
    processorNodeRef.current?.disconnect();
    sourceNodeRef.current?.disconnect();
    gainNodeRef.current?.disconnect();
    processorNodeRef.current = null;
    sourceNodeRef.current = null;
    gainNodeRef.current = null;
    ttsSourceRef.current?.stop();
    ttsSourceRef.current = null;
    if (revealRafRef.current !== null) {
      cancelAnimationFrame(revealRafRef.current);
      revealRafRef.current = null;
    }
    amplitudeRef.current = 0.08;
    setAmplitude(0.08);
    waveLevelsRef.current = initialWaveLevels;
    setWaveLevels(initialWaveLevels);

    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    mediaStreamRef.current = null;

    if (audioContextRef.current) {
      void audioContextRef.current.close();
      audioContextRef.current = null;
    }

    webrtcRef.current?.close();
    webrtcRef.current = null;

    // Clear audio queue
    ttsQueueRef.current = [];
    isTtsPlayingRef.current = false;
    ttsAllChunksReceivedRef.current = false;
    revealedTextRef.current = "";
    pendingDecodesRef.current = 0;
  };

  // Hoisted helper so playNextTtsChunk and message handlers can both call it.
  const updateMsg = useCallback((id: string, patch: Partial<ChatMessage>) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch } : m)));
  }, []);

  const clearTtsQueue = () => {
    ttsSourceRef.current?.stop();
    ttsSourceRef.current = null;
    ttsQueueRef.current = [];
    isTtsPlayingRef.current = false;
    ttsAllChunksReceivedRef.current = false;
    revealedTextRef.current = "";
    pendingDecodesRef.current = 0;
    pendingDocSentenceIdxRef.current = null;
    cancelHighlight();
    if (revealRafRef.current !== null) {
      cancelAnimationFrame(revealRafRef.current);
      revealRafRef.current = null;
    }
  };

  const playNextTtsChunk = useCallback(() => {
    if (ttsQueueRef.current.length === 0) {
      isTtsPlayingRef.current = false;
      // Only finalise once all decodeAudioData callbacks have fired — prevents
      // premature wrap-up when tts_done arrives before the last decode completes.
      if (ttsAllChunksReceivedRef.current && pendingDecodesRef.current === 0) {
        const aid = activeAssistantIdRef.current;
        const fullText = pendingAssistantTextRef.current;
        if (aid && fullText) updateMsg(aid, { text: fullText, isStreaming: false });
        activeAssistantIdRef.current = null;
        pendingAssistantTextRef.current = "";
        revealedTextRef.current = "";
        // WebRTC is a persistent session — add a fresh Listening bubble for the next turn.
        if (webrtcRef.current && !activeUserIdRef.current) {
          receivedFinalRef.current = false;
          const freshUid = crypto.randomUUID();
          activeUserIdRef.current = freshUid;
          setMessages((prev) => [
            ...prev,
            { id: freshUid, role: "user", text: "Listening…", isStreaming: true, isError: false },
          ]);
        }
        startTransition(() => { setMode("listening"); });
      }
      return;
    }

    const chunk = ttsQueueRef.current.shift()!;
    const audioCtx = audioContextRef.current ?? new AudioContext();
    if (!audioContextRef.current) audioContextRef.current = audioCtx;

    const source = audioCtx.createBufferSource();
    source.buffer = chunk.buffer;
    source.connect(audioCtx.destination);
    ttsSourceRef.current = source;
    interruptSentRef.current = false;
    bargeinFrameCountRef.current = 0;

    const aid = activeAssistantIdRef.current;
    const chunkText = chunk.text;
    const baseText = revealedTextRef.current;
    const durationMs = Math.max(200, chunk.buffer.duration * 1000);
    const startTime = performance.now();

    // Reveal just this sentence's text during its playback window.
    const tick = () => {
      if (!aid || ttsSourceRef.current !== source) { revealRafRef.current = null; return; }
      const elapsed = performance.now() - startTime;
      const progress = Math.min(1, elapsed / durationMs);
      const partial = chunkText.slice(0, Math.floor(progress * chunkText.length));
      const visible = baseText ? `${baseText} ${partial}` : partial;
      updateMsg(aid, { text: visible.trim(), isStreaming: true });
      if (progress < 1) {
        revealRafRef.current = requestAnimationFrame(tick);
      } else {
        revealRafRef.current = null;
      }
    };
    if (aid && chunkText) revealRafRef.current = requestAnimationFrame(tick);

    source.onended = () => {
      ttsSourceRef.current = null;
      if (revealRafRef.current !== null) { cancelAnimationFrame(revealRafRef.current); revealRafRef.current = null; }
      // Accumulate this sentence into the revealed text baseline.
      const newBase = baseText ? `${baseText} ${chunkText}` : chunkText;
      revealedTextRef.current = newBase.trim();
      if (aid) updateMsg(aid, { text: revealedTextRef.current, isStreaming: true });
      playNextTtsChunk();
    };

    isTtsPlayingRef.current = true;
    void audioCtx.resume().then(() => {
      source.start();
      // If this chunk is a document reading sentence, start word-level highlighting
      if (chunk.docSentenceIdx !== undefined && chunkText) {
        startWordHighlight(chunk.docSentenceIdx, chunkText, chunk.buffer.duration * 1000);
      }
    });
  }, [updateMsg, startWordHighlight]);

  useEffect(() => {
    return () => {
      stopAudioGraph();
      previewAudioRef.current?.pause();
      if (previewAudioUrlRef.current) {
        URL.revokeObjectURL(previewAudioUrlRef.current);
      }
      websocketRef.current?.close();
      websocketRef.current = null;
    };
  }, []);

  const applyStreamPayload = (payload: StreamMessage) => {
    if (payload.text !== undefined) {
      setTranscript(payload.text || "No speech detected yet.");
    }

    if (payload.timings_ms) {
      const sessionRoundtripMs =
        sessionStartedAtRef.current === null ? null : Number((performance.now() - sessionStartedAtRef.current).toFixed(2));
      setMetrics({
        ...payload.timings_ms,
        client_roundtrip_ms: sessionRoundtripMs,
      });
    }

    if (payload.debug) {
      setDebugInfo(payload.debug);
    }
  };

  const startStreaming = async () => {
    if (isConnecting || isRecordingRef.current) {
      return;
    }

    const startAttemptId = startAttemptRef.current + 1;
    startAttemptRef.current = startAttemptId;

    try {
      // Stop any audio still playing from a previous session before starting a new one.
      stopAudioGraph();
      setError(null);
      setMetrics(null);
      setDebugInfo(null);
      setTranscript("");
      setLlmLatencyMs(null);
      setTtsLatencyMs(null);
      activeUserIdRef.current = null;
      activeAssistantIdRef.current = null;
      setIsConnecting(true);
      setIsFinalizing(false);
      isFinalizingRef.current = false;
      receivedFinalRef.current = false;
      normalCloseRef.current = false;
      interruptSentRef.current = false;
      bargeinFrameCountRef.current = 0;

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      if (startAttemptRef.current !== startAttemptId) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      mediaStreamRef.current = stream;
      setIsRecording(true);
      isRecordingRef.current = true;
      startTransition(() => { setMode("listening"); });

      // ── WebRTC path ────────────────────────────────────────────────────────
      if (transportType === "webrtc") {
        const transport = new WebRTCTransport(backendUrl);
        webrtcRef.current = transport;

        // Shared message handler for WebRTC (same protocol as WebSocket).
        // Key differences: no socket.close() on tts_done — the session is
        // persistent; instead we reset to "listening" for the next turn.
        transport.onMessage = (rtcPayload) => {
          const payload = rtcPayload as StreamMessage;
          const updateMsg = (id: string, patch: Partial<ChatMessage>) => {
            setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch } : m)));
          };

          if (payload.type === "ready") {
            streamReadyRef.current = true;
            loadedDocumentIdRef.current = null;
            syncSelectedVoice(selectedTtsVoice);
            syncSelectedDocument(selectedDocumentId ?? null);
            const uid = crypto.randomUUID();
            activeUserIdRef.current = uid;
            activeAssistantIdRef.current = null;
            setMessages((prev) => [
              ...prev,
              { id: uid, role: "user", text: "Listening…", isStreaming: true, isError: false },
            ]);
            if (payload.request_id) {
              setDebugInfo((current) => ({
                request_id: String(payload.request_id ?? current?.request_id ?? "--"),
                filename: current?.filename ?? "stream.wav",
                audio_bytes: current?.audio_bytes ?? 0,
                detected_language: current?.detected_language ?? null,
                segments: current?.segments ?? 0,
                model_size: current?.model_size ?? "--",
                device: current?.device ?? "--",
                compute_type: current?.compute_type ?? "--",
                sample_rate: current?.sample_rate ?? null,
                chunks_received: current?.chunks_received ?? null,
              }));
            }
            return;
          }

          if (payload.type === "partial") {
            const text = typeof payload.text === "string" ? payload.text : undefined;
            if (text !== undefined) setTranscript(text || "No speech detected yet.");
            const timings = payload.timings_ms as Metrics | undefined;
            if (timings) {
              setMetrics({
                ...timings,
                client_roundtrip_ms:
                  sessionStartedAtRef.current === null
                    ? null
                    : Number((performance.now() - sessionStartedAtRef.current).toFixed(2)),
              });
            }
            if (payload.debug) setDebugInfo(payload.debug as DebugInfo);
            let uid = activeUserIdRef.current;
            if (!uid && text?.trim()) {
              uid = crypto.randomUUID();
              activeUserIdRef.current = uid;
              setMessages((prev) => [
                ...prev,
                { id: uid!, role: "user", text: text ?? "", isStreaming: true, isError: false },
              ]);
            } else if (uid && text?.trim()) {
              updateMsg(uid, { text });
            }
            startTransition(() => { setMode("thinking"); });
            return;
          }

          if (payload.type === "final") {
            receivedFinalRef.current = true;
            const text = typeof payload.text === "string" ? payload.text : undefined;
            if (text !== undefined) setTranscript(text || "No speech detected yet.");
            setIsFinalizing(false);
            isFinalizingRef.current = false;
            const uid = activeUserIdRef.current;
            const finalText = text?.trim() ?? "";
            if (uid) updateMsg(uid, { text: finalText || "…", isStreaming: false });
            startTransition(() => { setMode("responding"); });
            return;
          }

          if (payload.type === "llm_start") {
            const uid = activeUserIdRef.current;
            const userText = typeof payload.user_text === "string" ? payload.user_text.trim() : "";
            if (uid) {
              if (userText) {
                updateMsg(uid, { text: userText, isStreaming: false });
              } else {
                setMessages((prev) => prev.filter((m) => m.id !== uid));
              }
            }
            activeUserIdRef.current = null;
            // Finalize any lingering assistant bubble from a previous interrupted turn
            // instead of resetting it — that way the partial response stays visible.
            const prevAid = activeAssistantIdRef.current;
            if (prevAid) {
              const prevText = revealedTextRef.current || pendingAssistantTextRef.current;
              if (prevText) updateMsg(prevAid, { text: prevText, isStreaming: false });
            }
            activeAssistantIdRef.current = null;
            pendingAssistantTextRef.current = "";
            ttsAudioReceivedRef.current = false;
            if (revealRafRef.current !== null) {
              cancelAnimationFrame(revealRafRef.current);
              revealRafRef.current = null;
            }
            // Always open a fresh assistant bubble for this turn.
            const aid = crypto.randomUUID();
            activeAssistantIdRef.current = aid;
            setMessages((prev) => [
              ...prev,
              { id: aid, role: "assistant", text: "", isStreaming: true, isError: false },
            ]);
            return;
          }

          if (payload.type === "llm_partial") {
            pendingAssistantTextRef.current = typeof payload.text === "string" ? payload.text : "";
            startTransition(() => { setMode("responding"); });
            return;
          }

          if (payload.type === "llm_final") {
            pendingAssistantTextRef.current = typeof payload.text === "string" ? payload.text : "";
            if (payload.llm_ms != null) setLlmLatencyMs(Number(payload.llm_ms));
            startTransition(() => { setMode("listening"); });
            return;
          }

          if (payload.type === "llm_error") {
            const aid = activeAssistantIdRef.current;
            if (aid)
              updateMsg(aid, {
                text: "AI unavailable — make sure Ollama is running.",
                isStreaming: false,
                isError: true,
              });
            return;
          }

          if (payload.type === "tts_start") {
            // Stop any currently playing audio before starting the new sequence.
            // Prevents parallel playback when a second response starts before the
            // first finishes (e.g. after a short-pause false LLM trigger).
            clearTtsQueue();
            pendingDecodesRef.current = 0;
            ttsAudioReceivedRef.current = false;
            startTransition(() => { setMode("speaking"); });
            return;
          }

          if (payload.type === "tts_audio") {
            if (payload.tts_ms != null) setTtsLatencyMs(Number(payload.tts_ms));
            const sentenceText = typeof payload.sentence_text === "string" ? payload.sentence_text : (pendingAssistantTextRef.current ?? "");
            const docSentenceIdx =
              typeof payload.sentence_idx === "number"
                ? payload.sentence_idx
                : pendingDocSentenceIdxRef.current ?? undefined;
            pendingDocSentenceIdxRef.current = null;
            const data = typeof payload.data === "string" ? payload.data : "";
            const binaryString = atob(data);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
            const audioCtx = audioContextRef.current ?? new AudioContext();
            if (!audioContextRef.current) audioContextRef.current = audioCtx;
            ttsAudioReceivedRef.current = true;
            pendingDecodesRef.current++;
            void audioCtx.decodeAudioData(bytes.buffer.slice(0), (buffer) => {
              pendingDecodesRef.current--;
              ttsQueueRef.current.push({ buffer, text: sentenceText, docSentenceIdx });
              if (!isTtsPlayingRef.current) playNextTtsChunk();
            });
            return;
          }

          if (payload.type === "tts_done") {
            ttsAllChunksReceivedRef.current = true;
            if (!ttsAudioReceivedRef.current) {
              const aid = activeAssistantIdRef.current;
              if (aid && pendingAssistantTextRef.current) {
                updateMsg(aid, { text: pendingAssistantTextRef.current, isStreaming: false });
              }
              activeAssistantIdRef.current = null;
              pendingAssistantTextRef.current = "";
              revealedTextRef.current = "";
              receivedFinalRef.current = false;
              const freshUid = crypto.randomUUID();
              activeUserIdRef.current = freshUid;
              setMessages((prev) => [
                ...prev,
                { id: freshUid, role: "user", text: "Listening…", isStreaming: true, isError: false },
              ]);
              startTransition(() => { setMode("listening"); });
            } else if (!isTtsPlayingRef.current && ttsQueueRef.current.length === 0 && pendingDecodesRef.current === 0) {
              const aid = activeAssistantIdRef.current;
              if (aid) updateMsg(aid, { text: pendingAssistantTextRef.current, isStreaming: false });
              activeAssistantIdRef.current = null;
              pendingAssistantTextRef.current = "";
              revealedTextRef.current = "";
              receivedFinalRef.current = false;
              const freshUid = crypto.randomUUID();
              activeUserIdRef.current = freshUid;
              setMessages((prev) => [
                ...prev,
                { id: freshUid, role: "user", text: "Listening…", isStreaming: true, isError: false },
              ]);
              startTransition(() => { setMode("listening"); });
            }
            return;
          }

          if (payload.type === "tts_interrupted") {
            isReadingModeRef.current = false;
            clearTtsQueue();
            onDocumentEvent?.({ type: "tts_interrupted" });
            startTransition(() => { setMode("listening"); });
            return;
          }

          if (payload.type === "error") {
            const message = typeof payload.message === "string"
              ? payload.message
              : "Streaming transcription failed.";
            errorRef.current = message;
            setError(message);
            setIsFinalizing(false);
            isFinalizingRef.current = false;
            startTransition(() => { setMode("listening"); });
          }

          // ── Document events ──────────────────────────────────────────────
          if (payload.type === "doc_list" || payload.type === "doc_list_requested") {
            if (payload.documents) onDocumentEvent?.({ type: "doc_list", documents: payload.documents as never });
            return;
          }
          if (payload.type === "doc_read_start") {
            pendingDocSentenceIdxRef.current = null;
            cancelHighlight();
            isReadingModeRef.current = true;
            onDocumentEvent?.({ type: "doc_read_start", doc_id: payload.doc_id ?? "", sentences: (payload.sentences ?? []) as string[], title: payload.title ?? "" });
            return;
          }
          if (payload.type === "doc_opened") {
            onDocumentEvent?.({ type: "doc_opened", doc_id: payload.doc_id ?? "", title: payload.title ?? "", raw_markdown: (payload.raw_markdown ?? "") as string, sentences: (payload.sentences ?? []) as string[], annotations: payload.annotations as never });
            return;
          }
          if (payload.type === "doc_highlight") {
            pendingDocSentenceIdxRef.current = payload.sentence_idx ?? null;
            onDocumentEvent?.({ type: "doc_highlight", sentence_idx: payload.sentence_idx ?? 0, word_count: payload.word_count ?? 0 });
            return;
          }
          if (payload.type === "doc_save_snippet") {
            const lastAssistantText = pendingAssistantTextRef.current;
            onDocumentEvent?.({ type: "doc_save_snippet", term: payload.term ?? "" });
            onSnippetExplanation?.(lastAssistantText);
            return;
          }
          if (payload.type === "doc_note_saved" && payload.snippet) {
            onDocumentEvent?.({ type: "doc_note_saved", snippet: payload.snippet as never });
            return;
          }
          if (payload.type === "doc_highlight_saved") {
            onDocumentEvent?.({ type: "doc_highlight_saved", sentence_idx: payload.sentence_idx ?? 0, color: payload.color });
            return;
          }
          if (payload.type === "doc_search_start") {
            onDocumentEvent?.({ type: "doc_search_start", query: payload.query ?? "" });
            return;
          }
          if (payload.type === "doc_search_result") {
            onDocumentEvent?.({ type: "doc_search_result", query: payload.query ?? "", results: (payload.results ?? []) as never });
            return;
          }
          if (payload.type === "doc_export") {
            onDocumentEvent?.({ type: "doc_export", format: payload.format ?? "", download_url: payload.download_url ?? "" });
            return;
          }
          if (payload.type === "doc_reading_pause") {
            isReadingModeRef.current = false;
            onDocumentEvent?.({ type: "doc_reading_pause" });
            return;
          }
          if (payload.type === "doc_reading_resume") {
            isReadingModeRef.current = true;
            onDocumentEvent?.({ type: "doc_reading_resume" });
            return;
          }
        };

        transport.onClose = () => {
          if (startAttemptRef.current !== startAttemptId) {
            return;
          }
          streamReadyRef.current = false;
          loadedDocumentIdRef.current = null;
          webrtcRef.current = null;
          setIsConnecting(false);
          setIsRecording(false);
          isRecordingRef.current = false;
          if (!errorRef.current) stopAudioGraph();
        };

        try {
          await transport.connect(stream, { voice: selectedTtsVoice });
        } catch (connectErr) {
          if (startAttemptRef.current !== startAttemptId) {
            transport.close();
            return;
          }
          const msg =
            connectErr instanceof Error ? connectErr.message : "WebRTC connection failed.";
          errorRef.current = msg;
          setError(msg);
          setIsConnecting(false);
          setIsRecording(false);
          setIsFinalizing(false);
          stopAudioGraph();
          return;
        }

        if (startAttemptRef.current !== startAttemptId) {
          transport.close();
          return;
        }

        // ScriptProcessor: amplitude visualisation + client-side barge-in detection.
        // Audio is NOT sent as binary — it travels via the RTP track added in connect().
        const rtcAudioCtx = new AudioContext();
        audioContextRef.current = rtcAudioCtx;
        const rtcSource = rtcAudioCtx.createMediaStreamSource(stream);
        const rtcProcessor = rtcAudioCtx.createScriptProcessor(2048, 1, 1);
        const rtcGain = rtcAudioCtx.createGain();
        rtcGain.gain.value = 0;
        rtcSource.connect(rtcProcessor);
        rtcProcessor.connect(rtcGain);
        rtcGain.connect(rtcAudioCtx.destination);
        sourceNodeRef.current = rtcSource;
        processorNodeRef.current = rtcProcessor;
        gainNodeRef.current = rtcGain;

        rtcProcessor.onaudioprocess = (event) => {
          const channelData = event.inputBuffer.getChannelData(0);
          const rms = getRmsAmplitude(channelData);
          const nextAmplitude = Math.min(1, Math.max(0.04, rms * 11.5));
          const smoothed = amplitudeRef.current * 0.58 + nextAmplitude * 0.42;
          amplitudeRef.current = smoothed;
          setAmplitude(smoothed);
          const nextBars = [...waveLevelsRef.current.slice(1), smoothed];
          waveLevelsRef.current = nextBars;
          setWaveLevels(nextBars);

          // Client-side barge-in (complements server-side VAD in session.py)
          if ((ttsSourceRef.current || ttsQueueRef.current.length > 0) && !interruptSentRef.current) {
            if (rms > BARGE_IN_THRESHOLD) {
              bargeinFrameCountRef.current += 1;
              if (bargeinFrameCountRef.current >= BARGE_IN_FRAMES) {
                interruptSentRef.current = true;
                clearTtsQueue();
                transport.send({ type: "interrupt" });
                startTransition(() => { setMode("listening"); });
              }
            } else {
              bargeinFrameCountRef.current = 0;
            }
          }
          // No binary send here — audio travels via the RTP track.
        };

        transport.send({ type: "start", sample_rate: rtcAudioCtx.sampleRate });
        setIsConnecting(false);
        startTransition(() => { setMode("listening"); });
        return;
      }
      // ── End WebRTC path ────────────────────────────────────────────────────

      const socket = new WebSocket(websocketUrl);
      socket.binaryType = "arraybuffer";
      websocketRef.current = socket;
      sessionStartedAtRef.current = performance.now();
      streamReadyRef.current = false;

      socket.onopen = async () => {
        if (startAttemptRef.current !== startAttemptId) {
          normalCloseRef.current = true;
          socket.close();
          return;
        }

        const audioContext = new AudioContext();
        audioContextRef.current = audioContext;

        const sourceNode = audioContext.createMediaStreamSource(stream);
        const processorNode = audioContext.createScriptProcessor(2048, 1, 1);
        const gainNode = audioContext.createGain();
        gainNode.gain.value = 0;

        sourceNode.connect(processorNode);
        processorNode.connect(gainNode);
        gainNode.connect(audioContext.destination);

        sourceNodeRef.current = sourceNode;
        processorNodeRef.current = processorNode;
        gainNodeRef.current = gainNode;

        processorNode.onaudioprocess = (event) => {
          if (!streamReadyRef.current || socket.readyState !== WebSocket.OPEN) {
            return;
          }

          const channelData = event.inputBuffer.getChannelData(0);
          const rms = getRmsAmplitude(channelData);
          const nextAmplitude = Math.min(1, Math.max(0.04, rms * 11.5));
          const smoothedAmplitude = amplitudeRef.current * 0.58 + nextAmplitude * 0.42;
          amplitudeRef.current = smoothedAmplitude;
          setAmplitude(smoothedAmplitude);

          const nextBars = [...waveLevelsRef.current.slice(1), smoothedAmplitude];
          waveLevelsRef.current = nextBars;
          setWaveLevels(nextBars);

          if ((ttsSourceRef.current || ttsQueueRef.current.length > 0) && !interruptSentRef.current) {
            if (rms > BARGE_IN_THRESHOLD) {
              bargeinFrameCountRef.current += 1;
              if (bargeinFrameCountRef.current >= BARGE_IN_FRAMES) {
                interruptSentRef.current = true;
                clearTtsQueue();
                if (socket.readyState === WebSocket.OPEN) {
                  socket.send(JSON.stringify({ type: "interrupt" }));
                }
                startTransition(() => { setMode("listening"); });
              }
            } else {
              bargeinFrameCountRef.current = 0;
            }
          }

          const pcm16 = float32ToInt16(channelData);
          socket.send(pcm16.buffer);
        };

        socket.send(
          JSON.stringify({
            type: "start",
            sample_rate: audioContext.sampleRate,
          }),
        );

        setIsConnecting(false);
        startTransition(() => {
          setMode("listening");
        });
      };

      socket.onmessage = (event) => {
        const payload = JSON.parse(event.data) as StreamMessage;

        if (payload.type === "ready") {
          streamReadyRef.current = true;
          loadedDocumentIdRef.current = null;
          syncSelectedVoice(selectedTtsVoice);
          syncSelectedDocument(selectedDocumentId ?? null);
          // Create the user message bubble for this recording session
          const uid = crypto.randomUUID();
          activeUserIdRef.current = uid;
          activeAssistantIdRef.current = null;
          setMessages((prev) => [...prev, { id: uid, role: "user", text: "Listening…", isStreaming: true, isError: false }]);
          if (payload.request_id) {
            setDebugInfo((current) => ({
              request_id: payload.request_id ?? current?.request_id ?? "--",
              filename: current?.filename ?? "stream.wav",
              audio_bytes: current?.audio_bytes ?? 0,
              detected_language: current?.detected_language ?? null,
              segments: current?.segments ?? 0,
              model_size: current?.model_size ?? "--",
              device: current?.device ?? "--",
              compute_type: current?.compute_type ?? "--",
              sample_rate: current?.sample_rate ?? null,
              chunks_received: current?.chunks_received ?? null,
            }));
          }
          return;
        }

        if (payload.type === "partial") {
          applyStreamPayload(payload);
          let uid = activeUserIdRef.current;
          if (!uid && payload.text?.trim()) {
            uid = crypto.randomUUID();
            activeUserIdRef.current = uid;
            setMessages((prev) => [...prev, { id: uid!, role: "user", text: payload.text ?? "", isStreaming: true, isError: false }]);
          } else if (uid && payload.text?.trim()) {
            updateMsg(uid, { text: payload.text });
          }
          startTransition(() => { setMode("thinking"); });
          return;
        }

        if (payload.type === "final") {
          receivedFinalRef.current = true;
          applyStreamPayload(payload);
          setIsFinalizing(false);
          isFinalizingRef.current = false;
          const uid = activeUserIdRef.current;
          const finalText = payload.text?.trim() ?? "";
          if (uid) updateMsg(uid, { text: finalText || "…", isStreaming: false });
          startTransition(() => { setMode("responding"); });
          if (!finalText) {
            normalCloseRef.current = true;
            socket.close();
          }
          return;
        }

        if (payload.type === "llm_start") {
          const uid = activeUserIdRef.current;
          const userText = payload.user_text?.trim();
          if (uid) {
            if (userText) {
              updateMsg(uid, { text: userText, isStreaming: false });
            } else {
              setMessages((prev) => prev.filter((m) => m.id !== uid));
            }
          }
          activeUserIdRef.current = null;
          // Finalize any lingering assistant bubble rather than resetting it.
          const prevAid = activeAssistantIdRef.current;
          if (prevAid) {
            const prevText = revealedTextRef.current || pendingAssistantTextRef.current;
            if (prevText) updateMsg(prevAid, { text: prevText, isStreaming: false });
          }
          activeAssistantIdRef.current = null;
          pendingAssistantTextRef.current = "";
          ttsAudioReceivedRef.current = false;
          if (revealRafRef.current !== null) {
            cancelAnimationFrame(revealRafRef.current);
            revealRafRef.current = null;
          }
          const aid = crypto.randomUUID();
          activeAssistantIdRef.current = aid;
          setMessages((prev) => [...prev, { id: aid, role: "assistant", text: "", isStreaming: true, isError: false }]);
          return;
        }

        if (payload.type === "llm_partial") {
          // Buffer the text but DON'T render yet — we want the visible text to advance
          // in lockstep with TTS audio playback, not race ahead of it.
          pendingAssistantTextRef.current = payload.text ?? "";
          startTransition(() => { setMode("responding"); });
          return;
        }

        if (payload.type === "llm_final") {
          pendingAssistantTextRef.current = payload.text ?? "";
          if (payload.llm_ms != null) setLlmLatencyMs(payload.llm_ms);
          startTransition(() => { setMode(isRecordingRef.current ? "listening" : "responding"); });
          return;
        }

        if (payload.type === "llm_error") {
          const aid = activeAssistantIdRef.current;
          if (aid) updateMsg(aid, { text: "AI unavailable — make sure Ollama is running.", isStreaming: false, isError: true });
          if (!isRecordingRef.current) {
            normalCloseRef.current = true;
            socket.close();
          }
          return;
        }

        if (payload.type === "tts_start") {
          clearTtsQueue();
          pendingDecodesRef.current = 0;
          ttsAudioReceivedRef.current = false;
          startTransition(() => { setMode("speaking"); });
          return;
        }

        if (payload.type === "tts_audio") {
          if (payload.tts_ms != null) setTtsLatencyMs(payload.tts_ms);
          const sentenceText = payload.sentence_text ?? pendingAssistantTextRef.current;
          const docSentenceIdx =
            typeof payload.sentence_idx === "number"
              ? payload.sentence_idx
              : pendingDocSentenceIdxRef.current ?? undefined;
          pendingDocSentenceIdxRef.current = null;
          const binaryString = atob(payload.data ?? "");
          const bytes = new Uint8Array(binaryString.length);
          for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
          const audioCtx = audioContextRef.current ?? new AudioContext();
          if (!audioContextRef.current) audioContextRef.current = audioCtx;
          ttsAudioReceivedRef.current = true;
          pendingDecodesRef.current++;
          void audioCtx.decodeAudioData(bytes.buffer.slice(0), (buffer) => {
            pendingDecodesRef.current--;
            ttsQueueRef.current.push({ buffer, text: sentenceText, docSentenceIdx });
            if (!isTtsPlayingRef.current) playNextTtsChunk();
          });
          return;
        }

        if (payload.type === "tts_done") {
          ttsAllChunksReceivedRef.current = true;
          if (!ttsAudioReceivedRef.current) {
            const aid = activeAssistantIdRef.current;
            if (aid && pendingAssistantTextRef.current) {
              updateMsg(aid, { text: pendingAssistantTextRef.current, isStreaming: false });
            }
            activeAssistantIdRef.current = null;
            pendingAssistantTextRef.current = "";
            revealedTextRef.current = "";
            startTransition(() => { setMode(isRecordingRef.current ? "listening" : "responding"); });
            if (!isRecordingRef.current && receivedFinalRef.current) {
              normalCloseRef.current = true;
              socket.close();
            }
          } else if (!isTtsPlayingRef.current && ttsQueueRef.current.length === 0 && pendingDecodesRef.current === 0) {
            const aid = activeAssistantIdRef.current;
            if (aid) updateMsg(aid, { text: pendingAssistantTextRef.current, isStreaming: false });
            activeAssistantIdRef.current = null;
            pendingAssistantTextRef.current = "";
            revealedTextRef.current = "";
            startTransition(() => { setMode(isRecordingRef.current ? "listening" : "responding"); });
            if (!isRecordingRef.current && receivedFinalRef.current) {
              normalCloseRef.current = true;
              socket.close();
            }
          }
          return;
        }

        if (payload.type === "tts_interrupted") {
          isReadingModeRef.current = false;
          clearTtsQueue();
          onDocumentEvent?.({ type: "tts_interrupted" });
          startTransition(() => { setMode("listening"); });
          return;
        }

        if (payload.type === "error") {
          const message = payload.message ?? "Streaming transcription failed.";
          errorRef.current = message;
          setError(message);
          setIsFinalizing(false);
          isFinalizingRef.current = false;
          startTransition(() => {
            setMode("listening");
          });
        }

        // ── Document events (WebSocket path) ──────────────────────────────
        if (payload.type === "doc_list" || payload.type === "doc_list_requested") {
          if (payload.documents) onDocumentEvent?.({ type: "doc_list", documents: payload.documents as never });
          return;
        }
        if (payload.type === "doc_read_start") {
          pendingDocSentenceIdxRef.current = null;
          cancelHighlight();
          isReadingModeRef.current = true;
          onDocumentEvent?.({ type: "doc_read_start", doc_id: payload.doc_id ?? "", sentences: (payload.sentences ?? []) as string[], title: payload.title ?? "" });
          return;
        }
        if (payload.type === "doc_opened") {
          onDocumentEvent?.({ type: "doc_opened", doc_id: payload.doc_id ?? "", title: payload.title ?? "", raw_markdown: (payload.raw_markdown ?? "") as string, sentences: (payload.sentences ?? []) as string[], annotations: payload.annotations as never });
          return;
        }
        if (payload.type === "doc_highlight") {
          pendingDocSentenceIdxRef.current = payload.sentence_idx ?? null;
          onDocumentEvent?.({ type: "doc_highlight", sentence_idx: payload.sentence_idx ?? 0, word_count: payload.word_count ?? 0 });
          return;
        }
        if (payload.type === "doc_save_snippet") {
          onDocumentEvent?.({ type: "doc_save_snippet", term: payload.term ?? "" });
          onSnippetExplanation?.(pendingAssistantTextRef.current);
          return;
        }
        if (payload.type === "doc_note_saved" && payload.snippet) {
          onDocumentEvent?.({ type: "doc_note_saved", snippet: payload.snippet as never });
          return;
        }
        if (payload.type === "doc_highlight_saved") {
          onDocumentEvent?.({ type: "doc_highlight_saved", sentence_idx: payload.sentence_idx ?? 0, color: payload.color });
          return;
        }
        if (payload.type === "doc_search_start") {
          onDocumentEvent?.({ type: "doc_search_start", query: payload.query ?? "" });
          return;
        }
        if (payload.type === "doc_search_result") {
          onDocumentEvent?.({ type: "doc_search_result", query: payload.query ?? "", results: (payload.results ?? []) as never });
          return;
        }
        if (payload.type === "doc_export") {
          onDocumentEvent?.({ type: "doc_export", format: payload.format ?? "", download_url: payload.download_url ?? "" });
          return;
        }
        if (payload.type === "doc_reading_pause") {
          isReadingModeRef.current = false;
          onDocumentEvent?.({ type: "doc_reading_pause" });
          return;
        }
        if (payload.type === "doc_reading_resume") {
          isReadingModeRef.current = true;
          onDocumentEvent?.({ type: "doc_reading_resume" });
          return;
        }
      };

      socket.onerror = () => {
        if (startAttemptRef.current !== startAttemptId) {
          return;
        }
        const message = `Could not connect to backend stream at ${websocketUrl}. Run make dev and retry.`;
        errorRef.current = message;
        setError(message);
        setTranscript("Streaming connection failed before transcription could start.");
        setIsConnecting(false);
        setIsRecording(false);
        setIsFinalizing(false);
        stopAudioGraph();
      };

      socket.onclose = (event) => {
        if (startAttemptRef.current !== startAttemptId) {
          return;
        }
        streamReadyRef.current = false;
        loadedDocumentIdRef.current = null;
        websocketRef.current = null;
        setIsConnecting(false);
        setIsRecording(false);

        const wasExpectedClose =
          normalCloseRef.current || receivedFinalRef.current || isFinalizingRef.current || event.code === 1000;

        if (!wasExpectedClose && !errorRef.current) {
          const message = `Streaming connection closed unexpectedly at ${websocketUrl}.`;
          errorRef.current = message;
          setError(message);
        }

        stopAudioGraph();
      };
    } catch (caughtError) {
      if (startAttemptRef.current !== startAttemptId) {
        return;
      }
      const message = caughtError instanceof Error ? caughtError.message : "Microphone access failed.";
      setError(message);
      errorRef.current = message;
      setTranscript("Unable to start live transcription.");
      setIsConnecting(false);
      setIsRecording(false);
      setIsFinalizing(false);
      stopAudioGraph();
    }
  };

  const copyTranscript = useCallback(() => {
    const text = messages
      .map((m) => `${m.role === "user" ? "You" : "AI"}: ${m.text}`)
      .join("\n\n");
    void navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [messages]);

  const sendToBackend = useCallback((msg: object) => {
    const rtc = webrtcRef.current;
    if (rtc) { rtc.send(msg as Parameters<typeof rtc.send>[0]); return; }
    const ws = websocketRef.current;
    if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
  }, []);

  const syncSelectedVoice = useCallback((voice: string) => {
    if (!streamReadyRef.current) {
      return;
    }
    sendToBackend({ type: "tts_voice", voice });
  }, [sendToBackend]);

  useEffect(() => {
    syncSelectedVoice(selectedTtsVoice);
  }, [selectedTtsVoice, syncSelectedVoice]);

  const stopVoicePreview = useCallback(() => {
    previewAudioRef.current?.pause();
    previewAudioRef.current = null;
    if (previewAudioUrlRef.current) {
      URL.revokeObjectURL(previewAudioUrlRef.current);
      previewAudioUrlRef.current = null;
    }
    setPreviewingVoice(null);
  }, []);

  const previewVoice = useCallback(async (voiceId: string) => {
    if (isRecordingRef.current || isConnecting || mode === "speaking") {
      return;
    }
    stopVoicePreview();
    setVoicePreviewError(null);
    setPreviewingVoice(voiceId);

    try {
      const response = await fetch(`${backendUrl}/tts/preview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ voice: voiceId }),
      });
      if (!response.ok) {
        throw new Error("Voice preview failed.");
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      previewAudioUrlRef.current = url;
      previewAudioRef.current = audio;
      audio.onended = () => {
        if (previewAudioRef.current === audio) {
          stopVoicePreview();
        }
      };
      audio.onerror = () => {
        if (previewAudioRef.current === audio) {
          setVoicePreviewError("Preview playback failed.");
          stopVoicePreview();
        }
      };
      await audio.play();
    } catch (err) {
      setVoicePreviewError(err instanceof Error ? err.message : "Voice preview failed.");
      stopVoicePreview();
    }
  }, [isConnecting, mode, stopVoicePreview]);

  const selectVoice = useCallback((voiceId: string) => {
    setSelectedTtsVoice(voiceId);
    void previewVoice(voiceId);
  }, [previewVoice]);

  const syncSelectedDocument = useCallback((docId: string | null) => {
    if (!streamReadyRef.current) {
      return;
    }

    if (docId) {
      if (loadedDocumentIdRef.current === docId) {
        return;
      }
      loadedDocumentIdRef.current = docId;
      sendToBackend({ type: "doc_load", doc_id: docId });
      return;
    }

    if (loadedDocumentIdRef.current !== null) {
      loadedDocumentIdRef.current = null;
      sendToBackend({ type: "doc_unload" });
    }
  }, [sendToBackend]);

  useEffect(() => {
    syncSelectedDocument(selectedDocumentId ?? null);
  }, [selectedDocumentId, syncSelectedDocument]);

  const stopStreaming = () => {
    startAttemptRef.current += 1;
    const wasConnecting = isConnecting;
    setIsConnecting(false);
    setIsRecording(false);
    isRecordingRef.current = false;
    normalCloseRef.current = true;

    // WebRTC: closing the transport tears down the peer connection; the server
    // handles cleanup via connectionstatechange.  No "stop" message is sent
    // because the session is persistent and there is no single "current utterance"
    // to finalise — the server's silence debounce handles turn detection.
    const transport = webrtcRef.current;
    if (transport) {
      stopAudioGraph(); // closes transport + mic + audio graph
      setIsFinalizing(false);
      isFinalizingRef.current = false;
      startTransition(() => { setMode("listening"); });
      return;
    }

    // WebSocket: send "stop" and let the existing WS pipeline finalise.
    const socket = websocketRef.current;
    if (wasConnecting) {
      websocketRef.current = null;
      socket?.close();
      stopAudioGraph();
      setIsFinalizing(false);
      isFinalizingRef.current = false;
      startTransition(() => { setMode("listening"); });
      return;
    }

    setIsFinalizing(true);
    isFinalizingRef.current = true;
    stopAudioGraph();
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "stop" }));
      startTransition(() => {
        setMode("thinking");
      });
      return;
    }

    setIsFinalizing(false);
    isFinalizingRef.current = false;
  };

  const activeMode = modeConfig[mode];
  const controlLabel = isRecording || isConnecting ? "Stop Streaming" : "Start Live Transcription";
  const controlDisabled = isFinalizing;
  const orbScale = (1 + amplitude * 0.42).toFixed(3);
  const orbGlow = (0.45 + amplitude * 1.3).toFixed(3);
  const orbTilt = `${(amplitude * 18).toFixed(2)}deg`;
  const orbCoreScale = (1 + amplitude * 0.34).toFixed(3);
  const orbDriftX = `${(amplitude * 12).toFixed(2)}px`;
  const orbDriftY = `${(amplitude * -10).toFixed(2)}px`;
  const selectedVoice = ttsVoices.find((voice) => voice.id === selectedTtsVoice) ?? ttsVoices[0] ?? FALLBACK_TTS_VOICES[0];
  const voicePreviewDisabled = isRecording || isConnecting || mode === "speaking";
  const selectedVoiceIndex = Math.max(0, ttsVoices.findIndex((voice) => voice.id === selectedVoice.id));
  const selectedVoiceLabel = getVoiceLabel(selectedVoice);
  const visibleVoiceOffsets = [-2, -1, 0, 1, 2];
  const visibleVoiceItems = visibleVoiceOffsets
    .map((offset) => {
      if (!ttsVoices.length) return null;
      const index = (selectedVoiceIndex + offset + ttsVoices.length) % ttsVoices.length;
      const voice = ttsVoices[index];
      return { voice, offset, index, label: getVoiceLabel(voice) };
    })
    .filter(Boolean) as { voice: TtsVoice; offset: number; index: number; label: { displayName: string; detail: string } }[];
  const selectAdjacentVoice = useCallback((direction: -1 | 1) => {
    if (!ttsVoices.length) return;
    const nextIndex = (selectedVoiceIndex + direction + ttsVoices.length) % ttsVoices.length;
    selectVoice(ttsVoices[nextIndex].id);
  }, [selectedVoiceIndex, selectVoice, ttsVoices]);

  return (
    <main className="console-shell">
      <section className="console-frame">

        {/* ── Topbar ──────────────────────────────────────────── */}
        <header className="topbar surface">
          <div>
            <p className="kicker">Document Intelligence Workspace</p>
            <h1>NeuroTalk</h1>
            <p className="topbar-tagline">Document Intelligence Workspace</p>
          </div>
          <nav className="app-tabs" aria-label="Workspace mode">
            <button type="button" className="app-tab is-active" aria-pressed="true">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
                <path d="M4 12v.01M8 7v10M12 4v16M16 8v8M20 11v2" />
              </svg>
              Conversation
            </button>
            <button type="button" className="app-tab" aria-pressed="false">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
                <path d="M4 4.5A2.5 2.5 0 0 1 6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5z" />
              </svg>
              Reading
            </button>
          </nav>
          <div className="topbar-meta">
            <span className="status-pill is-live">Realtime sync</span>
            <span className="status-pill is-ghost">Private session</span>
            <button
              type="button"
              className="voice-lens-button"
              onClick={() => setIsVoiceSettingsOpen(true)}
              aria-label="Open voice settings"
              title={selectedVoice.name}
            >
              <span className="voice-lens-core" aria-hidden="true" />
            </button>
            <button
              type="button"
              className="theme-toggle"
              onClick={toggleTheme}
              aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
            >
              {isDark ? (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="5"/>
                  <line x1="12" y1="1" x2="12" y2="3"/>
                  <line x1="12" y1="21" x2="12" y2="23"/>
                  <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
                  <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
                  <line x1="1" y1="12" x2="3" y2="12"/>
                  <line x1="21" y1="12" x2="23" y2="12"/>
                  <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
                  <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
                </svg>
              ) : (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
                </svg>
              )}
            </button>
          </div>
        </header>

        {isVoiceSettingsOpen && (
          <div className="settings-overlay" role="presentation" onClick={() => setIsVoiceSettingsOpen(false)}>
            <section
              className="voice-settings-panel surface"
              role="dialog"
              aria-modal="true"
              aria-labelledby="voice-settings-title"
              onClick={(event) => event.stopPropagation()}
            >
              <aside className="voice-settings-nav" aria-label="Voice settings sections">
                <button type="button" className="voice-settings-nav-item is-active">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
                    <path d="M4 12v.01M8 7v10M12 4v16M16 8v8M20 11v2" />
                  </svg>
                  Voice
                </button>
                <button type="button" className="voice-settings-nav-item">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M4 13a8 8 0 1 1 16 0" />
                    <path d="M4 13h3l1 5h8l1-5h3" />
                  </svg>
                  Speech speed
                </button>
                <button type="button" className="voice-settings-nav-item">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M11 5 6 9H3v6h3l5 4z" />
                    <path d="M16 9a5 5 0 0 1 0 6" />
                    <path d="M19 6a9 9 0 0 1 0 12" />
                  </svg>
                  Playback
                </button>
                <div className="voice-tip">
                  <strong>Tip</strong>
                  <span>You can change voices anytime during a session.</span>
                </div>
              </aside>

              <div className="voice-settings-stage">
                <button
                  type="button"
                  className="settings-close-button"
                  onClick={() => setIsVoiceSettingsOpen(false)}
                  aria-label="Close voice settings"
                >
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
                    <path d="M6 6l12 12M18 6 6 18" />
                  </svg>
                </button>

                <header className="voice-settings-hero">
                  <div className="voice-settings-mark" aria-hidden="true">
                    <span />
                    <span />
                    <span />
                    <span />
                  </div>
                  <p className="kicker">Voice Settings</p>
                  <h2 id="voice-settings-title">Choose your NeuroTalk voice</h2>
                  <p>Preview and select the voice used for reading and answers.</p>
                </header>

                <div className="voice-carousel">
                  <button
                    type="button"
                    className="voice-carousel-arrow voice-carousel-arrow--left"
                    onClick={() => selectAdjacentVoice(-1)}
                    aria-label="Previous voice"
                  >
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <path d="m15 18-6-6 6-6" />
                    </svg>
                  </button>

                  <div className="voice-carousel-track">
                    {visibleVoiceItems.map((item) => {
                      const isSelected = item.voice.id === selectedTtsVoice;
                      const isPreviewing = previewingVoice === item.voice.id;
                      return (
                        <button
                          key={`${item.voice.id}-${item.offset}`}
                          type="button"
                          className={[
                            "voice-choice",
                            `voice-choice--offset-${item.offset}`,
                            isSelected ? "is-selected" : "",
                            isPreviewing ? "is-previewing" : "",
                          ].filter(Boolean).join(" ")}
                          style={{ "--voice-index": item.index } as CSSProperties}
                          onClick={() => selectVoice(item.voice.id)}
                          aria-pressed={isSelected}
                          aria-label={`Select and preview ${item.voice.name} voice`}
                        >
                          <span className="voice-choice-portrait" aria-hidden="true" />
                          <span className="voice-choice-wave" aria-hidden="true">
                            <span /><span /><span /><span />
                          </span>
                          <strong>{item.label.displayName}</strong>
                          <small>{item.label.detail}</small>
                        </button>
                      );
                    })}
                  </div>

                  <button
                    type="button"
                    className="voice-carousel-arrow voice-carousel-arrow--right"
                    onClick={() => selectAdjacentVoice(1)}
                    aria-label="Next voice"
                  >
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <path d="m9 18 6-6-6-6" />
                    </svg>
                  </button>
                </div>

                <div className="voice-selected-copy">
                  <div className="voice-selected-wave" aria-hidden="true">
                    <span /><span /><span /><span /><span />
                  </div>
                  <h3>{selectedVoiceLabel.displayName}</h3>
                  <p>{selectedVoiceLabel.detail}</p>
                  <div className="voice-mini-wave" aria-hidden="true">
                    {waveformHeights.slice(0, 18).map((height, index) => (
                      <span key={`${height}-${index}`} style={{ "--bar-height": `${8 + (index % 5) * 5}px` } as CSSProperties} />
                    ))}
                  </div>
                  <p className="voice-description">Warm, natural and empathetic. Great for conversations, reading and storytelling.</p>
                </div>

                <div className="voice-settings-actions">
                  <button
                    type="button"
                    className="voice-preview-button"
                    onClick={() => previewVoice(selectedTtsVoice)}
                    disabled={voicePreviewDisabled}
                  >
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <path d="m8 5 11 7-11 7z" />
                    </svg>
                    {previewingVoice === selectedTtsVoice ? "Playing preview" : "Preview this voice"}
                  </button>
                  <button
                    type="button"
                    className="voice-use-button"
                    onClick={() => setIsVoiceSettingsOpen(false)}
                  >
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <path d="m20 6-11 11-5-5" />
                    </svg>
                    Use this voice
                  </button>
                </div>

                <div className="voice-carousel-dots" aria-hidden="true">
                  {[-2, -1, 0, 1, 2].map((dot) => (
                    <span key={dot} className={dot === 0 ? "is-active" : undefined} />
                  ))}
                </div>

                <footer className="voice-settings-footer">
                  <span>{voicePreviewDisabled ? "Stop the live session to preview voices." : "Your voice preference is saved locally and synced to the active session."}</span>
                  {voicePreviewError ? <strong>{voicePreviewError}</strong> : null}
                </footer>
              </div>
            </section>
          </div>
        )}

        <div className="console-body">
          <div className="voice-workspace">
            {/* ── Orb Zone ─────────────────────────────────────────── */}
            <div className="orb-zone surface">
              <div className="orb-zone-header">
                <span className={`mode-chip ${activeMode.accent}`}>
                  {(mode === "listening" || mode === "speaking") && (
                    <span className="mode-chip-dot" aria-hidden="true" />
                  )}
                  {activeMode.eyebrow}
                </span>
                <div className="mode-switcher">
                  {(["listening", "thinking", "responding"] as Mode[]).map((item, index) => (
                    <Fragment key={item}>
                      {index > 0 && (
                        <span className={`mode-step-line${
                          ["listening", "thinking", "responding"].indexOf(mode) >= index ? " is-active" : ""
                        }`} />
                      )}
                      <button
                        type="button"
                        className={item === mode ? "mode-button is-selected" : "mode-button is-static"}
                        disabled
                      >{item}</button>
                    </Fragment>
                  ))}
                </div>
              </div>

              <div className="orb-stage">
                <div className="orb-center-row">
                  <button
                    type="button"
                    className={[
                      "orbital-core",
                      isRecording ? "orbital-core--recording" : "",
                      isConnecting ? "orbital-core--connecting" : "",
                    ].filter(Boolean).join(" ")}
                    disabled={controlDisabled}
                    onClick={isRecording || isConnecting ? stopStreaming : () => void startStreaming()}
                    aria-label={controlLabel}
                    style={
                      {
                        "--orb-scale": orbScale,
                        "--orb-glow": orbGlow,
                        "--orb-tilt": orbTilt,
                        "--orb-core-scale": orbCoreScale,
                        "--orb-drift-x": orbDriftX,
                        "--orb-drift-y": orbDriftY,
                      } as CSSProperties
                    }
                  >
                    <div className="orb-ring orb-ring-1" />
                    <div className="orb-ring orb-ring-2" />
                    <div className="orb-center" />
                    <div className="orb-scanline" />
                  </button>

                  <div className="orb-side">
                    <p className="orb-status-title">
                      {isFinalizing
                        ? "Thinking..."
                        : isConnecting && !isRecording
                          ? "Starting..."
                          : mode === "speaking"
                            ? "Speaking..."
                            : "Listening..."}
                    </p>
                    <p className={`orb-tap-hint${isRecording ? " is-active" : isFinalizing ? " is-muted" : ""}`}>
                      {error
                        ? <span className="is-error">{error}</span>
                        : isRecording
                          ? "Speak naturally, I'm here to help."
                          : activeMode.summary}
                    </p>
                    <div className="wave-grid" aria-hidden="true">
                      {waveformHeights.map((height, index) => (
                        <span
                          className="wave-bar"
                          key={`${height}-${index}`}
                          style={
                            {
                              "--bar-height": `${24 + waveLevels[index] * 90}px`,
                              "--bar-delay": `${index * 0.03}s`,
                              "--bar-opacity": (0.3 + waveLevels[index] * 0.7).toFixed(3),
                              "--bar-scale": (0.82 + waveLevels[index] * 0.48).toFixed(3),
                            } as CSSProperties
                          }
                        />
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* ── Transcript Feed ─────────────────────────────────── */}
            <article className="transcript-panel surface">
              <div className="section-heading">
                <p className="kicker">Live Transcription</p>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <span className={`status-pill${isRecording ? " is-live" : " is-ghost"}`}>
                    {error ? "Attention needed" : isRecording ? "Live" : messages.length ? "Done" : "Ready"}
                  </span>
                  <button
                    type="button"
                    className={`copy-button${copied ? " is-copied" : ""}`}
                    onClick={copyTranscript}
                    disabled={messages.length === 0}
                    aria-label="Copy conversation"
                  >
                    {copied ? (
                      <>
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                        Copied
                      </>
                    ) : (
                      <>
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                        Copy
                      </>
                    )}
                  </button>
                </div>
              </div>

              <div className="chat-thread">
                {messages.length === 0 ? (
                  <div className="transcript-sample" aria-label="Transcript preview">
                    <div className="transcript-line transcript-line--user">
                      <span className="transcript-avatar" aria-hidden="true" />
                      <p><strong>You</strong> <span>00:12</span></p>
                      <p>Can you explain the main idea of this document?</p>
                    </div>
                    <div className="transcript-line transcript-line--assistant">
                      <span className="transcript-avatar" aria-hidden="true" />
                      <p><strong>NeuroTalk</strong> <span>00:15</span></p>
                      <p>The document introduces PrismDocs, an intelligent document generator that transforms complex content into clear, structured, accessible documents...</p>
                    </div>
                  </div>
                ) : (
                  messages.map((msg) => (
                    <div key={msg.id} className={`chat-message chat-message--${msg.role}`}>
                      <span className={`chat-avatar chat-avatar--${msg.role}`}>
                        {msg.role === "user" ? "You" : "AI"}
                      </span>
                      <div
                        className={[
                          "chat-bubble",
                          `chat-bubble--${msg.role}`,
                          msg.isError ? "chat-bubble--error" : "",
                          msg.isStreaming && msg.role === "assistant" ? "is-streaming" : "",
                        ].filter(Boolean).join(" ")}
                      >
                        {msg.isError ? (
                          <p className="chat-text chat-text--error">{msg.text}</p>
                        ) : msg.text ? (
                          <p className="chat-text">{msg.text}</p>
                        ) : msg.isStreaming ? (
                          msg.role === "assistant" ? (
                            <div className="chat-typing-indicator" aria-label="AI is thinking">
                              <span /><span /><span />
                            </div>
                          ) : (
                            <p className="chat-text chat-text--placeholder">Listening…</p>
                          )
                        ) : (
                          <p className="chat-text chat-text--placeholder">…</p>
                        )}
                        {msg.isStreaming && msg.role === "user" && msg.text && (
                          <span className="chat-typing-dot" aria-hidden="true" />
                        )}
                      </div>
                    </div>
                  ))
                )}
                <div ref={chatEndRef} />
              </div>

              <div className="transcript-footer">
                <span className="transcript-meta">{error ?? `Ref: ${debugInfo?.request_id ?? "--"}`}</span>
                <span className="transcript-meta">Lang: {debugInfo?.detected_language ?? "--"}</span>
              </div>
            </article>
          </div>

          <div className="document-workspace">
            {children}
          </div>
        </div>

      </section>

      <footer className="console-footer">
        <div className="console-footer-inner">
          <span className="console-footer-brand">NeuroTalk</span>
          <span className="console-footer-sep" aria-hidden="true">·</span>
          <span className="console-footer-tagline">Customer-ready document voice interface</span>
          <span className="console-footer-sep" aria-hidden="true">·</span>
          <a
            className="console-footer-link"
            href="https://github.com/nitishkmr005/neuroTalk"
            target="_blank"
            rel="noopener noreferrer"
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" style={{ display: "inline", verticalAlign: "middle", marginRight: 5 }}>
              <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
            </svg>
            GitHub
          </a>
        </div>
      </footer>
    </main>
  );
}
