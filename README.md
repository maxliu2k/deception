# Agent Arena Prototype

Six agents participate in a Socratic seminar, taking turns through a shared turn-request queue and writing messages to a shared SQLite database.

Each seminar runs for 12 minutes. When time expires, the seminar pauses automatically and the judge grades all participants.

## Requirements

- Python 3.10+
- `OPENAI_API_KEY` (or `keys/gptkey.txt`) for GPT participants
- `ANTHROPIC_API_KEY` (or `keys/claudekey.txt`) for Claude participants
- Default seminar participants are `GPT 4o`, `GPT 5.2`, `Claude Haiku 4.5`, `Claude Sonnet 4.6`, and `Claude Opus 4.6`
- Optional: `SEMINAR_TOPIC` to override the default seminar text loaded from `app/prelude.txt` (fallback: `prelude.txt`) and injected into each model's prompt

Install dependencies:

```bash
pip install fastapi uvicorn aiosqlite openai anthropic
```

## Run

```bash
uvicorn app.server:app --reload
```

Open:

```text
http://127.0.0.1:8000/
```

Reloading the page starts a fresh seminar automatically.

## Files

- `app/server.py` FastAPI server + websocket
- `app/db.py` async SQLite helpers (`arena.sqlite` in repo root)
- `app/agents.py` 5 seminar agents + referee loop
- `app/static/index.html` UI

## Reset

Use the `Reset Seminar` button in the UI or:

```bash
curl -X POST http://127.0.0.1:8000/api/reset
```
