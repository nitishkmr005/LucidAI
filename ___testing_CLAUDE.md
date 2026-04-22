# Claude Code — Project Conventions

Generic conventions for AI agent projects. Copy to any new project and adjust names/values.

---

## Project Layout

```
project/
├── backend/          Python backend (FastAPI + uv)
├── frontend/         Next.js frontend (TypeScript)
├── scripts/          Standalone learnable .py demos
├── docs/
│   └── blog.md       Narrative explanation of the project
├── README.md         Minimal setup guide
├── Makefile          dev / setup / check targets
├── CLAUDE.md         This file
└── AGENTS.md         Codex / other agent conventions
```

---

## Backend Conventions

**Package manager:** uv (`uv add <pkg>`, `uv run python ...`). Never `pip` or `poetry`.

**Python:** 3.11+. Use modern syntax: `X | Y` unions, `match`, `list[str]`.

**Folder roles inside `backend/app/`:**

| Folder | Purpose |
|--------|---------|
| `services/` | External integrations (STT, LLM, TTS, DB). One class or function per file. |
| `agents/` | Orchestration — chains services together. |
| `prompts/` | System prompt constants. Never inline prompts in business logic. |
| `tools/` | LLM tool/function definitions. |
| `modules/` | Reusable domain modules. |
| `utils/` | Pure utility functions, no side effects. |
| `models.py` | Pydantic request/response schemas. |
| `main.py` | FastAPI app, routes, WebSocket handlers only. |

**Config pattern:** `pydantic-settings` + `.env` via `config/settings.py`. Never use bare `os.getenv()` in application code.

**Never commit `.env`.** Always maintain `.env.example` with safe placeholder values.

---

## Logging

Two sinks — distinct purposes, both always set up at startup.

**Terminal sink** (runtime/human): colorized Loguru output to stdout. Used for live debugging during development. Not persisted.

```python
logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan> | <level>{message}</level>")
```

**JSON file sink** (`backend/logs/`): structured, persistent output for performance analysis, debugging, and LLM call tracing. These are the durable records — not for human reading during dev.

```python
logger.add("logs/app_{time}.json", serialize=True, rotation="10 MB", retention=5)
```

Rules:
- `backend/logs/` — gitignored, `.gitkeep` commits the folder
- Keep only latest 5 files (`retention=5`)
- Never use `print()` — always `logger.info / debug / warning / error`
- Structured key=value pairs: `logger.info("event=llm_done latency_ms={}", ms)`
- LLM calls must log: model, token count, latency, request ID

---

## Frontend Conventions

**Stack:** Next.js 15 · TypeScript · CSS custom properties (no Tailwind unless project uses it).

**Theme system** (when implemented):
- Tokens in `:root` (light) and `[data-theme="dark"]` blocks in `globals.css`
- `data-theme` attribute set on `<html>` element
- Persist preference in localStorage (key: `nt-theme`, values: `"dark"` | `"light"`)
- Initialize theme with `useState(defaultValue)` — always a stable SSR-safe value — then sync localStorage in `useEffect`. Never use `typeof window` in `useState` initializer (causes hydration mismatch in Next.js App Router).

---

## Scripts Folder

`scripts/` — standalone, runnable `.py` files. One per module. Teach how each component works in isolation.

Rules:
- Runs with `uv run --project backend python scripts/<name>.py`
- No shared imports between scripts — self-contained
- Docstring at top: what it teaches and how to run it
- `if __name__ == "__main__":` block with working demo
- Prints timing output

---

## Deployment

| Layer | Platform | Notes |
|-------|----------|-------|
| Frontend | Vercel | Connect GitHub → set root to `frontend/` → add `NEXT_PUBLIC_BACKEND_URL` env var |
| Backend | Railway | Connect GitHub → set root to `backend/` → set all `.env.example` vars → expose port |

---

## Authentication

Use Supabase for user authentication.

- Frontend: `@supabase/supabase-js`, env vars `NEXT_PUBLIC_SUPABASE_URL` + `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- Backend: verify JWT from `Authorization: Bearer <token>` using Supabase admin client with `SUPABASE_SERVICE_ROLE_KEY`
- Never expose `SUPABASE_SERVICE_ROLE_KEY` to frontend or commit it

---

## Makefile Targets

```makefile
make setup    # install all deps
make dev      # start backend + frontend with hot-reload
make backend  # backend only
make frontend # frontend only
make check    # lint + type check
```

---

## Code Style

- Type-annotate all function signatures
- `async def` for all I/O-bound functions
- No bare `except:` — always catch specific exceptions
- Use `loguru`, not `logging`
- Import order: stdlib → third-party → local (enforced by ruff)
- No TODO comments in committed code — open a GitHub issue instead
- Write acceptance criteria in claude.md file 

---

## .gitignore Essentials

Ignore: `.env`, `.venv`, `__pycache__`, `*.pyc`, `.cache`, `logs/`, `node_modules/`, `.next/`, `.DS_Store`

Track: `.env.example`, `logs/.gitkeep`
