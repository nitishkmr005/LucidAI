# TTS Comparison Report

Text: `Happy to help. Tell me what you need, and we will sort it out together.`

## Summary

| model                      | status | latency             | official_latency                            | params | disk    | output                        |
|----------------------------|--------|---------------------|---------------------------------------------|--------|---------|-------------------------------|
| Chatterbox Turbo           | ok     | 3897.5 ms (3.90s)   | sub-200ms service / local runtime-dependent | 350M   | 4.04 GB | scripts/speech/chatterbox.wav |
| Qwen3-TTS 0.6B CustomVoice | ok     | 7153.1 ms (7.15s)   | as low as 97 ms                             | 0.6B   | 2.5 GB  | scripts/speech/qwen.wav       |
| VibeVoice Realtime 0.5B    | ok     | 15960.1 ms (15.96s) | ~200 ms first audio                         | 0.5B   | 2.04 GB | scripts/speech/vibevoice.wav  |
| Kokoro 82M MLX             | ok     | 221.1 ms (0.22s)    | fast local MLX runtime                      | 82M    | 355 MB  | scripts/speech/kokoro.wav     |

## Details

## Chatterbox Turbo

- Status: `ok`
- Latency: `3897.5 ms`
- Output: `scripts/speech/chatterbox.wav`
- Voice: `built-in default`

Notes:
```text
Turbo tags via inline prompt. Fell back to built-in default voice because /tmp/voice_agent.wav was missing.
```

## Qwen3-TTS 0.6B CustomVoice

- Status: `ok`
- Latency: `7153.1 ms`
- Output: `scripts/speech/qwen.wav`
- Voice: `speaker=Ryan`

Notes:
```text
instruct='Very happy and energetic.'; device=cpu
```

## VibeVoice Realtime 0.5B

- Status: `ok`
- Latency: `15960.1 ms`
- Output: `scripts/speech/vibevoice.wav`
- Voice: `speaker=Emma`

Notes:
```text
Ran via official realtime demo script on cpu.
```

## Kokoro 82M MLX

- Status: `ok`
- Latency: `221.1 ms`
- Output: `scripts/speech/kokoro.wav`
- Voice: `af_heart`

Notes:
```text
MLX Kokoro with lang_code=a speed=1.0.
```
