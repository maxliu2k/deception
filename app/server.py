import asyncio
import json
import os
import signal
import time
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import agents
from . import db


app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")

_clients: List[WebSocket] = []
_broadcast_lock = asyncio.Lock()
_grading_lock = asyncio.Lock()
PAUSED = False
PAUSED_AT = 0.0
TOTAL_PAUSED_SECONDS = 0.0
DEFAULT_AGENTS = list(agents.AGENT_NAMES)
DEFAULT_TOPICS = list(agents.AVAILABLE_TOPICS)
CURRENT_AGENTS = list(DEFAULT_AGENTS)
CURRENT_TOPICS = list(DEFAULT_TOPICS)
CURRENT_VOICE_PREFS = {}
SEMINAR_DURATION_SECONDS = 12 * 60
SEMINAR_STARTED_AT = 0.0
SEMINAR_RUN_ID = 0
LATEST_GRADES = None
SEMINAR_ACTIVE = False


def _seconds_remaining() -> int:
    if not SEMINAR_STARTED_AT:
        return SEMINAR_DURATION_SECONDS
    paused_total = TOTAL_PAUSED_SECONDS + (time.time() - PAUSED_AT if PAUSED and PAUSED_AT else 0.0)
    effective_elapsed = (time.time() - SEMINAR_STARTED_AT) - paused_total
    return max(0, int(SEMINAR_DURATION_SECONDS - effective_elapsed))


async def start_default_seminar(selected_agents: List[str] | None = None, selected_topics: List[str] | None = None) -> None:
    global SEMINAR_STARTED_AT, SEMINAR_RUN_ID, LATEST_GRADES, PAUSED, PAUSED_AT, TOTAL_PAUSED_SECONDS
    global CURRENT_AGENTS, CURRENT_TOPICS, SEMINAR_ACTIVE
    valid_agents = [name for name in DEFAULT_AGENTS if name in (selected_agents or DEFAULT_AGENTS)]
    if not valid_agents:
        valid_agents = list(DEFAULT_AGENTS)
    valid_topics = [name for name in DEFAULT_TOPICS if name in (selected_topics or DEFAULT_TOPICS)]
    if not valid_topics:
        valid_topics = list(DEFAULT_TOPICS)
    CURRENT_AGENTS = valid_agents
    CURRENT_TOPICS = valid_topics
    agents.set_topic_focus(CURRENT_TOPICS)
    SEMINAR_STARTED_AT = time.time()
    SEMINAR_RUN_ID += 1
    LATEST_GRADES = None
    SEMINAR_ACTIVE = True
    PAUSED = False
    PAUSED_AT = 0.0
    TOTAL_PAUSED_SECONDS = 0.0
    agents.set_paused(False)
    await db.set_room_agents("main", ",".join(CURRENT_AGENTS))
    await db.insert_message(
        "System",
        f"{len(CURRENT_AGENTS)} seminar participants will discuss `prelude.txt` collaboratively. "
        f"Focus areas: {', '.join(CURRENT_TOPICS)}. "
        "Speakers are called in a randomized rotating order. On your turn, contribute if you have something meaningful; otherwise you may pass so another speaker can continue the point. "
        "The text has been preloaded for everyone, so ground claims in it, cite or paraphrase specifics, challenge assumptions, and ask follow-up questions. "
        "The seminar lasts 12 minutes, then the judge grades everyone.",
        room="main",
    )
    asyncio.create_task(seminar_timer_loop(SEMINAR_RUN_ID))


async def complete_and_grade_seminar(auto: bool = False) -> dict:
    global PAUSED, LATEST_GRADES, SEMINAR_ACTIVE
    async with _grading_lock:
        if LATEST_GRADES is not None:
            return LATEST_GRADES

        PAUSED = True
        SEMINAR_ACTIVE = False
        agents.set_paused(True)
        await db.cancel_active_turn(room="main")
        await db.cancel_pending_requests(room="main")
        if auto:
            await db.insert_message(
                "System",
                "Time is up. The seminar is now closed, and the judge is grading each participant.",
                room="main",
            )
        LATEST_GRADES = await agents.grade_seminar()
        await broadcast_state()
        return LATEST_GRADES


async def seminar_timer_loop(run_id: int) -> None:
    while True:
        await asyncio.sleep(1.0)
        if run_id != SEMINAR_RUN_ID:
            return
        if LATEST_GRADES is not None:
            return
        if not PAUSED and _seconds_remaining() <= 0:
            await complete_and_grade_seminar(auto=True)
            return


async def broadcast_state() -> None:
    async with _broadcast_lock:
        state = await db.get_state(room="main")
        state["paused"] = PAUSED
        state["grades"] = LATEST_GRADES
        state["seminar_started_at"] = SEMINAR_STARTED_AT
        state["seminar_duration_seconds"] = SEMINAR_DURATION_SECONDS
        state["seconds_remaining"] = _seconds_remaining()
        state["seminar_active"] = SEMINAR_ACTIVE
        state["selected_agents"] = list(CURRENT_AGENTS)
        state["selected_topics"] = list(CURRENT_TOPICS)
        state["voice_prefs"] = dict(CURRENT_VOICE_PREFS)
        data = json.dumps(state)

        dead = []
        for ws in _clients:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)

        for ws in dead:
            if ws in _clients:
                _clients.remove(ws)


@app.on_event("startup")
async def on_startup() -> None:
    global PAUSED, PAUSED_AT, TOTAL_PAUSED_SECONDS, SEMINAR_ACTIVE, SEMINAR_STARTED_AT, LATEST_GRADES, CURRENT_VOICE_PREFS
    await db.init_db()
    await db.reset_db()
    PAUSED = True
    PAUSED_AT = 0.0
    TOTAL_PAUSED_SECONDS = 0.0
    SEMINAR_ACTIVE = False
    SEMINAR_STARTED_AT = 0.0
    LATEST_GRADES = None
    CURRENT_VOICE_PREFS = await db.get_voice_prefs()
    agents.set_paused(True)
    await db.set_room_agents("main", ",".join(DEFAULT_AGENTS))
    agents.set_topic_focus(DEFAULT_TOPICS)
    await agents.start_agents(broadcast_state)


signal.signal(signal.SIGINT, lambda *_: os._exit(0))


@app.get("/")
async def root():
    return FileResponse("app/static/index.html")


@app.get("/api/state")
async def api_state():
    state = await db.get_state(room="main")
    state["paused"] = PAUSED
    state["grades"] = LATEST_GRADES
    state["seminar_started_at"] = SEMINAR_STARTED_AT
    state["seminar_duration_seconds"] = SEMINAR_DURATION_SECONDS
    state["seconds_remaining"] = _seconds_remaining()
    state["seminar_active"] = SEMINAR_ACTIVE
    state["selected_agents"] = list(CURRENT_AGENTS)
    state["selected_topics"] = list(CURRENT_TOPICS)
    state["voice_prefs"] = dict(CURRENT_VOICE_PREFS)
    return JSONResponse(state)


@app.get("/api/options")
async def api_options():
    return JSONResponse(
        {
            "agents": list(DEFAULT_AGENTS),
            "topics": list(DEFAULT_TOPICS),
            "selected_agents": list(CURRENT_AGENTS),
            "selected_topics": list(CURRENT_TOPICS),
            "seminar_active": SEMINAR_ACTIVE,
            "voice_prefs": dict(CURRENT_VOICE_PREFS),
        }
    )


@app.post("/api/start")
async def api_start(payload: dict):
    global CURRENT_VOICE_PREFS
    selected_agents = payload.get("agents") or []
    selected_topics = payload.get("topics") or []
    incoming_voice_prefs = payload.get("voice_prefs") or {}
    CURRENT_VOICE_PREFS = {
        name: str(voice)
        for name, voice in incoming_voice_prefs.items()
        if name in DEFAULT_AGENTS and isinstance(voice, str) and voice.strip()
    }
    await db.set_voice_prefs(CURRENT_VOICE_PREFS)
    await db.reset_db()
    await start_default_seminar(selected_agents=selected_agents, selected_topics=selected_topics)
    await broadcast_state()
    state = await db.get_state(room="main")
    state["paused"] = PAUSED
    state["grades"] = LATEST_GRADES
    state["seminar_started_at"] = SEMINAR_STARTED_AT
    state["seminar_duration_seconds"] = SEMINAR_DURATION_SECONDS
    state["seconds_remaining"] = SEMINAR_DURATION_SECONDS
    state["seminar_active"] = SEMINAR_ACTIVE
    state["selected_agents"] = list(CURRENT_AGENTS)
    state["selected_topics"] = list(CURRENT_TOPICS)
    state["voice_prefs"] = dict(CURRENT_VOICE_PREFS)
    return JSONResponse(state)


@app.post("/api/reset")
async def api_reset():
    await db.reset_db()
    await start_default_seminar(selected_agents=CURRENT_AGENTS, selected_topics=CURRENT_TOPICS)
    await broadcast_state()
    state = await db.get_state(room="main")
    state["paused"] = PAUSED
    state["grades"] = LATEST_GRADES
    state["seminar_started_at"] = SEMINAR_STARTED_AT
    state["seminar_duration_seconds"] = SEMINAR_DURATION_SECONDS
    state["seconds_remaining"] = SEMINAR_DURATION_SECONDS
    state["seminar_active"] = SEMINAR_ACTIVE
    state["selected_agents"] = list(CURRENT_AGENTS)
    state["selected_topics"] = list(CURRENT_TOPICS)
    state["voice_prefs"] = dict(CURRENT_VOICE_PREFS)
    return JSONResponse(state)


@app.post("/api/end")
async def api_end():
    global PAUSED, PAUSED_AT, TOTAL_PAUSED_SECONDS, SEMINAR_ACTIVE, LATEST_GRADES, SEMINAR_STARTED_AT
    PAUSED = True
    PAUSED_AT = 0.0
    TOTAL_PAUSED_SECONDS = 0.0
    SEMINAR_ACTIVE = False
    LATEST_GRADES = None
    SEMINAR_STARTED_AT = 0.0
    agents.set_paused(True)
    await db.cancel_active_turn(room="main")
    await db.cancel_pending_requests(room="main")
    await db.reset_db()
    await db.set_room_agents("main", ",".join(CURRENT_AGENTS))
    await broadcast_state()
    state = await db.get_state(room="main")
    state["paused"] = PAUSED
    state["grades"] = LATEST_GRADES
    state["seminar_started_at"] = SEMINAR_STARTED_AT
    state["seminar_duration_seconds"] = SEMINAR_DURATION_SECONDS
    state["seconds_remaining"] = SEMINAR_DURATION_SECONDS
    state["seminar_active"] = SEMINAR_ACTIVE
    state["selected_agents"] = list(CURRENT_AGENTS)
    state["selected_topics"] = list(CURRENT_TOPICS)
    state["voice_prefs"] = dict(CURRENT_VOICE_PREFS)
    return JSONResponse(state)


@app.post("/api/system_message")
async def api_system_message(payload: dict):
    content = (payload.get("content") or "").strip()
    if not content:
        return JSONResponse({"ok": False, "error": "empty"}, status_code=400)
    await db.cancel_active_turn(room="main")
    await db.insert_message("System", content, room="main")
    await broadcast_state()
    return JSONResponse({"ok": True})


@app.post("/api/pause")
async def api_pause(payload: dict):
    global PAUSED, PAUSED_AT, TOTAL_PAUSED_SECONDS
    value = payload.get("paused")
    new_paused = not PAUSED if value is None else bool(value)
    if new_paused and not PAUSED:
        PAUSED_AT = time.time()
    elif not new_paused and PAUSED and PAUSED_AT:
        TOTAL_PAUSED_SECONDS += time.time() - PAUSED_AT
        PAUSED_AT = 0.0
    PAUSED = new_paused
    agents.set_paused(PAUSED)
    await broadcast_state()
    return JSONResponse({"paused": PAUSED})


@app.post("/api/grade")
async def api_grade():
    grades = await complete_and_grade_seminar(auto=False)
    return JSONResponse({"grades": grades})


@app.post("/api/shutdown")
async def api_shutdown():
    os._exit(0)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.append(ws)
    state = await db.get_state(room="main")
    state["paused"] = PAUSED
    state["grades"] = LATEST_GRADES
    state["seminar_started_at"] = SEMINAR_STARTED_AT
    state["seminar_duration_seconds"] = SEMINAR_DURATION_SECONDS
    state["seconds_remaining"] = _seconds_remaining()
    state["seminar_active"] = SEMINAR_ACTIVE
    state["selected_agents"] = list(CURRENT_AGENTS)
    state["selected_topics"] = list(CURRENT_TOPICS)
    state["voice_prefs"] = dict(CURRENT_VOICE_PREFS)
    await ws.send_text(json.dumps(state))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if ws in _clients:
            _clients.remove(ws)
