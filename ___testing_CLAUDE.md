# AGENTS.md

## 🧠 Core Features Contract (Do Not Break)

- Agent must always know:
  - all uploaded documents
  - currently selected document
  - and confirm before reading

- Document reading must:
  - use stored content only (never LLM-generated)
  - stream sequentially (sentence/token order)
  - support pause and exact resume

- TTS must be streaming with:
  - real-time word-level highlighting
  - perfect sync between speech and UI
  - resume continuing from last position

- Interruptions must:
  - immediately stop TTS
  - preserve reading state
  - switch to Q&A without losing context

- Q&A must:
  - use recent document context (last N sentences)
  - avoid generic answers if context exists
  - optionally use web search if needed

- After Q&A:
  - "continue" must resume reading exactly where paused
  - highlighting + TTS must stay in sync

- Notes must:
  - be saved with linked document span
  - persist across sessions
  - open via clickable text in document

- Highlighting must support:
  - live reading highlight (moving)
  - user highlight (static, different color)
  - persistence

- Agent behavior:
  - must act like a teacher explaining the document
  - support word, sentence, and concept explanations

- Export must support:
  - PDF and DOCX
  - include document + highlights + notes + annotations

- System state must always track:
  - current_document_id
  - reading_position (sentence, word)
  - is_reading
  - notes
  - highlights
  - and must never reset on interruption

- LLM must:
  - handle only Q&A and decisions
  - not generate full document reading
  - not control playback or highlighting

- Session must:
  - remain conversational and continuous
  - retain context across interactions

- UI must always maintain:
  - left panel → voice agent
  - right panel → document viewer
  - real-time highlighting + clickable notes

- System must:
  - never lose reading position, notes, or highlights
  - degrade gracefully on failure
  - allow resume without restart

---

## 🚨 Critical Regression Rule
Any change that breaks:
- reading continuity
- word-level sync
- interruption handling
- context-aware Q&A

→ must be rejected

## Repository Rules
- Keep a single `README.md` in the repository root.
- Keep project documentation inside `docs/`.
- `docs/` must contain:
  - `blog.md` written in blog-style format explaining the project
  - `demo_narrative.md` for stakeholder demo narration
- Keep all learnable standalone scripts inside `scripts/`.
- `scripts/` must contain educational scripts for each core important module used in the codebase.
- Maintain modular project structure so components are easy to replace without large refactors.

## Run and Developer Experience
- Use `make` for one-line developer commands.
- Starting, checking, and other common workflows must be available through `make`.
- Prefer simple, predictable developer workflows.

## Python and Environment
- Use `uv` for Python dependency and environment management.
- Do not use `pip`, `poetry`, or other package managers.
- Use `.env` only for secrets such as passwords and API keys.
- Do not store non-secret configuration in `.env`.
- `.venv` and `.env` must be ignored in `.gitignore`.

## Configuration Rules
- Keep application configuration in `settings.py` using Pydantic settings.
- Centralize configuration access through settings objects.
- Do not scatter configuration across files.
- Do not read configuration directly from environment variables in application code.

## Logging Rules
- Use `loguru` for all logging.
- Terminal logs must support development and debugging.
- Persist structured JSON logs in `logs/`.
- Keep a maximum of 5 log files through rotation/retention.
- Logs must support LLM tracing and include:
  - input
  - output
  - latency
- Do not use `print()` for application logging.

## Architecture Rules
- Preserve modular design across the codebase.
- Separate orchestration, integrations, business logic, configuration, and utilities cleanly.
- Design the LLM layer so provider swaps are easy.
- It must be easy to replace Ollama-based LLM calls with Anthropic, OpenAI, or Gemini-based implementations.
- Avoid provider-specific logic leaking across the codebase.

## Documentation and Maintainability
- Keep code concise, readable, and production-oriented.
- Prefer replaceable modules over tightly coupled implementations.
- Write code so future agents can understand and modify it safely.
- Any new feature must fit the existing modular structure rather than introducing shortcuts or duplication.