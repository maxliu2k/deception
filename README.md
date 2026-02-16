# Agent Arena Prototype

10 OpenAI-powered agents share a common text space, take turns via a turn-request queue, and write messages to a shared SQLite database.

## Requirements

- Python 3.10+
- `OPENAI_API_KEY` set in your environment
- Optional: `OPENAI_MODEL` (defaults to `gpt-5.2`)

Install dependencies:

```bash
pip install fastapi uvicorn aiosqlite openai
```

## Run

```bash
uvicorn app.server:app --reload
```

Open:

```
http://127.0.0.1:8000/
```

## Files

- `app/server.py` FastAPI server + websocket
- `app/db.py` async SQLite helpers (`arena.sqlite` in repo root)
- `app/agents.py` 10 agents + referee loop
- `app/static/index.html` UI

## Reset

Use the “Reset DB” button in the UI or:

```bash
curl -X POST http://127.0.0.1:8000/api/reset
```
