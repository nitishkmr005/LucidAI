/**
 * AudioWorklet processor for real-time PCM16 capture.
 *
 * Accumulates Float32 samples in BATCH_SIZE chunks, converts to Int16 PCM,
 * computes RMS amplitude for barge-in detection, then posts both to the main thread.
 *
 * Runs off the main thread — no UI work here.
 */

const BATCH_SIZE = 2048; // ~46ms at 44100 Hz, ~128ms at 16000 Hz

class AudioCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = new Float32Array(BATCH_SIZE);
    this._pos = 0;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;

    for (let i = 0; i < channel.length; i++) {
      this._buf[this._pos++] = channel[i];

      if (this._pos === BATCH_SIZE) {
        // RMS for barge-in
        let sum = 0;
        for (let j = 0; j < BATCH_SIZE; j++) sum += this._buf[j] ** 2;
        const rms = Math.sqrt(sum / BATCH_SIZE);

        // Float32 → Int16
        const pcm16 = new Int16Array(BATCH_SIZE);
        for (let j = 0; j < BATCH_SIZE; j++) {
          const s = Math.max(-1, Math.min(1, this._buf[j]));
          pcm16[j] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }

        this.port.postMessage({ pcm16: pcm16.buffer, rms }, [pcm16.buffer]);
        this._pos = 0;
      }
    }

    return true; // keep processor alive
  }
}

registerProcessor("audio-capture-processor", AudioCaptureProcessor);
