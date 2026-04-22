# Voice-Powered Interactive Document Reader — Implementation Plan

## Context

The existing project (NeuroTalk) is a production-grade real-time voice agent using WebRTC, faster-whisper STT, Ollama LLM, and Kokoro TTS. The user wants to extend it into an interactive document reading assistant: a two-panel UI where the left panel holds the voice agent and the right panel displays a markdown document being read aloud — with live word highlighting, Q&A, web search, annotation saving, and PDF/DOCX export.

---

## Architecture Overview

The voice pipeline (audio → STT → LLM → TTS) remains intact. We add:
1. A **document action tag system** extending the existing emotion tag pattern (`emotion.py`)
2. **REST endpoints** for document upload, list, annotate, and export
3. A **two-panel React layout** with a new document viewer component
4. **Word-level highlight sync** via audio duration estimation in the frontend
5. **DuckDuckGo web search** running as a background async task during LLM streaming

---

## New Files Created

| Path | Responsibility |
|------|---------------|
| `backend/app/utils/action_tags.py` | Parse/strip `[DOC_ACTION:...]` tags, convert to WS message dicts |
| `backend/app/services/document_store.py` | Markdown file I/O, sentence parsing, annotation JSON CRUD |
| `backend/app/services/search.py` | Async DuckDuckGo search (5s timeout, executor-based) |
| `backend/app/services/exporter.py` | Generate PDF (reportlab) and DOCX (python-docx) exports |
| `backend/app/routers/documents.py` | REST router: upload, list, get, annotate, export endpoints |
| `frontend/components/document-panel.tsx` | Right-panel: markdown render, word spans, highlights, tooltips |
| `frontend/components/document-uploader.tsx` | Drag-drop file upload + document list UI |
| `frontend/components/search-result-popup.tsx` | Floating popup for web search results |
| `frontend/hooks/use-document-highlight.ts` | Word-tick timing algorithm (rAF + setTimeout chain) |

---

## Existing Files Modified

| Path | Change |
|------|--------|
| `backend/app/main.py` | Extract doc action tags per LLM token batch; dispatch WS messages; kick off search task |
| `backend/app/webrtc/session.py` | Mirror all main.py changes for WebRTC path; store `active_document_context` |
| `backend/app/services/llm.py` | Add `document_context: str | None` param injected as second system message |
| `backend/app/prompts/system.py` | Add `DOCUMENT_READING_PROMPT` constant with action tag instructions + few-shot examples |
| `backend/config/settings.py` | Add `docs_dir`, `annotations_dir`, `exports_dir`, `web_search_enabled`, `web_search_max_results` |
| `backend/pyproject.toml` | Add `duckduckgo-search>=6.0`, `reportlab>=4.0`, `python-docx>=1.1`, `markdown>=3.6` |
| `frontend/app/page.tsx` | Two-column grid layout; lift document state; render both panels |
| `frontend/components/voice-agent-console.tsx` | Handle new WS message types; call word-highlight hook from `playNextTtsChunk` |
| `frontend/app/globals.css` | Add layout vars, document panel styles, highlight/annotation CSS |

---

## Action Tag Design (extends `emotion.py` pattern)

LLM embeds tags at the **start** of its response. A new `action_tags.py` module handles them — `emotion.py` is unchanged.

```
[DOC_ACTION:list_docs]              → send doc_list to frontend
[DOC_ACTION:read:{doc_id}]          → open document in right panel, start reading
[DOC_ACTION:highlight:{sent_idx}:{word_count}]  → mark sentence being read
[DOC_ACTION:search:{query}]         → trigger background web search
[DOC_ACTION:save_snippet:{term}]    → save current agent answer as annotation
[DOC_ACTION:export:pdf]             → trigger browser PDF download
[DOC_ACTION:export:docx]            → trigger browser DOCX download
[DOC_ACTION:reading_pause]          → pause reading
[DOC_ACTION:reading_resume]         → resume reading
```

`extract_doc_actions(text) -> (cleaned_text, list[DocAction])` — strips tags, returns parsed actions. Called on each accumulated LLM response in `main.py`. A `doc_actions_seen: set[str]` per-LLM-call prevents re-dispatching the same tag on subsequent partial tokens.

---

## New WebSocket Message Types

**Server → Client (additions):**
```json
{"type": "doc_list", "documents": [...]}
{"type": "doc_read_start", "doc_id": "...", "sentences": [...], "title": "..."}
{"type": "doc_highlight", "sentence_idx": 3, "word_count": 12}
{"type": "doc_search_start", "query": "..."}
{"type": "doc_search_result", "query": "...", "results": [{"title","snippet","url"}]}
{"type": "doc_save_snippet", "term": "...", "sentence_idx": 7}
{"type": "doc_export", "format": "pdf"}
{"type": "doc_reading_pause"}
{"type": "doc_reading_resume"}
```

**Existing `tts_audio` extended with optional field:**
```json
{"type": "tts_audio", "data": "base64", "sentence_text": "...", "doc_sentence_idx": 3}
```

**Client → Server (additions):**
```json
{"type": "doc_load", "doc_id": "..."}
{"type": "doc_unload"}
{"type": "doc_save_highlight_text", "doc_id": "...", "sentence_idx": 4}
```

---

## Word Highlight Sync Algorithm (`use-document-highlight.ts`)

```
When tts_audio arrives with doc_sentence_idx:
  words = sentence_text.split(/\s+/)
  msPerWord = max(80ms,  audioDuration * 0.85 / words.length)
                         ↑ 15% buffer for trailing silence

  tick():
    emit onWordTick(sentenceIdx, wordIdx)
    wordIdx++
    if more words: requestAnimationFrame(() => setTimeout(tick, msPerWord))

On tts_interrupted or source.onended:
  cancelAnimationFrame + clearTimeout
  activeWordIdx = null
```

`audioDuration` comes from `chunk.buffer.duration * 1000` — already available in the existing `playNextTtsChunk` function, no backend change needed.

---

## Data Storage Schema

```
backend/data/documents/{doc_id}.md           # raw markdown
backend/data/documents/index.json            # [{doc_id, filename, title, word_count, uploaded_at}]
backend/data/annotations/{doc_id}_annotations.json
```

Annotation JSON:
```json
{
  "doc_id": "...",
  "highlights": [{"id","sentence_idx","color","created_at","source"}],
  "snippets":   [{"id","term","explanation","sentence_idx","word_idx","search_results","created_at"}],
  "reading_position": {"last_sentence_idx": 87, "saved_at": "..."}
}
```

---

## REST Endpoints (`/documents` router)

```
POST   /documents/upload              → {doc_id, filename, title, word_count}
GET    /documents/                    → [{doc_id, filename, title, ...}]
GET    /documents/{doc_id}            → {doc_id, filename, title, raw_markdown, sentences, annotations}
DELETE /documents/{doc_id}
POST   /documents/{doc_id}/annotations/highlight  → body: {sentence_idx, color}
POST   /documents/{doc_id}/annotations/snippet    → body: {term, explanation, sentence_idx, word_idx}
GET    /documents/{doc_id}/export/{pdf|docx}      → FileResponse (auto-deletes temp file)
```

---

## Export Implementation

**PDF** (`reportlab`): Sentences as `Paragraph` objects. Highlighted sentences get a yellow `Rect` drawn on the canvas behind them. Snippets rendered as indented italic callout paragraphs.

**DOCX** (`python-docx`): Sentences as paragraphs. Highlighted sentences: each word added as a separate `Run` with `font.highlight_color = WD_COLOR_INDEX.YELLOW`. Snippets as italic paragraphs prefixed with `[Note: ...]`.

Both exporters run in `loop.run_in_executor(None, ...)` to avoid blocking the async server.

---

## Implementation Phases

| Phase | Days | Deliverables |
|-------|------|-------------|
| 1 — Foundation | 1-2 | `action_tags.py`, `document_store.py`, settings + deps, system prompt |
| 2 — Backend Pipeline | 3-4 | `llm.py` update, `main.py`+`session.py` tag dispatch, `routers/documents.py`, `search.py` |
| 3 — Frontend Layout | 5-6 | `page.tsx` two-column, `document-uploader.tsx`, `document-panel.tsx` (static) |
| 4 — Real-Time Sync | 7-8 | Highlight hook, `voice-agent-console.tsx` wiring, text-selection flow, search popup |
| 5 — Export | 9 | `exporter.py`, export endpoint, frontend download trigger |
| 6 — Polish | 10 | Scroll-to-active, reading position persistence, error states, responsive CSS |

---

## Critical Files Reference

- `backend/app/utils/emotion.py` — pattern to replicate for action tags
- `backend/app/main.py` — LLM token loop to modify for tag extraction
- `backend/app/webrtc/session.py` — mirror of main.py for WebRTC path
- `backend/app/services/llm.py` — add `document_context` param
- `backend/app/prompts/system.py` — add `DOCUMENT_READING_PROMPT`
- `backend/config/settings.py` — add storage paths + search settings
- `backend/pyproject.toml` — add 4 new deps
- `frontend/components/voice-agent-console.tsx` — handle new message types, word-highlight hook
- `frontend/app/page.tsx` — two-column layout
- `frontend/app/globals.css` — document panel + highlight styles

---

## Verification Plan

1. Upload a `.md` file via drag-drop → confirm it appears in document list
2. Say "Can you see my documents?" → agent speaks names and asks which to read
3. Say "Read [document name]" → document appears in right panel, agent reads aloud, words highlight sentence by sentence
4. Interrupt mid-reading with "What does X mean?" → agent pauses, answers or shows search popup
5. Say "Save that" → snippet annotation saved; hovering the word in the document shows the snippet
6. Select a sentence with mouse → "Highlight" button appears; click it → permanent yellow highlight
7. Say "Save as PDF" → PDF downloads with all highlights and snippets included
8. Say "Save as DOCX" → DOCX downloads with yellow-highlighted sentences and italic snippet notes
