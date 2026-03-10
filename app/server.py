import asyncio
from datetime import datetime
import json
import logging
import os
import signal
import time
import uuid
import re
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi import Request
from openai import AsyncOpenAI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

from . import agents
from . import db


app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
LOGGER = logging.getLogger("agent_arena.server")

_clients: List[WebSocket] = []
_broadcast_lock = asyncio.Lock()
_grading_lock = asyncio.Lock()
PAUSED = False
PAUSED_AT = 0.0
TOTAL_PAUSED_SECONDS = 0.0
DEFAULT_AGENTS = list(agents.AGENT_NAMES)
DEFAULT_TOPICS = list(agents.AVAILABLE_TOPICS)
DEFAULT_CHAPTERS = [c["id"] for c in agents.get_available_chapters()]
CURRENT_AGENTS = list(DEFAULT_AGENTS)
CURRENT_TOPICS = list(DEFAULT_TOPICS)
CURRENT_CHAPTERS = list(DEFAULT_CHAPTERS)
CURRENT_VOICE_PREFS = {}
SEMINAR_DURATION_SECONDS = 20 * 60
SEMINAR_STARTED_AT = 0.0
SEMINAR_RUN_ID = 0
LATEST_GRADES = None
SEMINAR_ACTIVE = False
SEMINAR_REMAINING_SECONDS = float(SEMINAR_DURATION_SECONDS)
LAST_TIMER_TICK_TS = 0.0
SPEAKING_UNTIL_TS = 0.0
LAST_SPOKEN_MESSAGE_ID = 0
TIMER_EXPIRED = False
REFLECTION_STATUS: dict[str, str] = {}
GRADER_NAME = "Sonnet"
GRADER_STATUS = "idle"  # idle | grading | done
REFLECTION_MESSAGE_IDS: list[int] = []
GRADER_MESSAGE_IDS: list[int] = []
LAST_API_ACTIVITY_TS = time.time()
AUTO_SHUTDOWN_IDLE_SECONDS = 60 * 60
OPENAI_TTS_VOICES = {"alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin", "cedar", "nova"}
DEFAULT_TTS_VOICE = {
    "4o": "nova",
    "5.4": "ash",
    "Flash": "echo",
    "Pro": "sage",
    "Haiku": "marin",
    "Sonnet": "cedar",
    "Opus": "onyx",
    "System": "alloy",
}
TTS_MANIFESTS: dict[int, dict] = {}
TTS_CHUNKS: dict[str, bytes] = {}


def _seconds_remaining() -> int:
    return max(0, int(SEMINAR_REMAINING_SECONDS))


def _touch_activity() -> None:
    global LAST_API_ACTIVITY_TS
    LAST_API_ACTIVITY_TS = time.time()


def _estimate_speaking_seconds(text: str) -> float:
    words = max(1, len((text or "").split()))
    wpm = 130.0
    speed = 1.1
    return max(1.0, (words / (wpm * speed)) * 60.0)


def _refresh_speaking_window(state: dict) -> None:
    global LAST_SPOKEN_MESSAGE_ID, SPEAKING_UNTIL_TS
    messages = state.get("messages") or []
    if not messages:
        return
    last = messages[-1]
    msg_id = int(last.get("id") or 0)
    if msg_id <= LAST_SPOKEN_MESSAGE_ID:
        return
    LAST_SPOKEN_MESSAGE_ID = msg_id
    sender = last.get("sender")
    content = (last.get("content") or "").strip()
    if sender and sender != "System" and content:
        SPEAKING_UNTIL_TS = max(SPEAKING_UNTIL_TS, time.time() + _estimate_speaking_seconds(content))


def _advance_timer_tick(state: dict) -> None:
    global LAST_TIMER_TICK_TS, SEMINAR_REMAINING_SECONDS
    now = time.time()
    if LAST_TIMER_TICK_TS <= 0:
        LAST_TIMER_TICK_TS = now
        return
    delta = max(0.0, now - LAST_TIMER_TICK_TS)
    LAST_TIMER_TICK_TS = now
    if delta <= 0:
        return
    active = state.get("active") or {}
    is_thinking = bool(active.get("agent"))
    is_speaking = now < SPEAKING_UNTIL_TS
    if SEMINAR_ACTIVE and (not PAUSED) and (not is_thinking) and is_speaking:
        SEMINAR_REMAINING_SECONDS = max(0.0, SEMINAR_REMAINING_SECONDS - delta)


def _get_openai_key() -> str:
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    key_path = "keys/gptkey.txt"
    if os.path.exists(key_path):
        return open(key_path, "r", encoding="utf-8").read().strip()
    return ""


def _split_sentences_with_offsets(text: str) -> list[dict]:
    chunks: list[dict] = []
    for m in re.finditer(r"[^.!?]*[.!?]+", text or ""):
        raw = m.group(0)
        stripped = raw.strip()
        if stripped:
            start = m.start() + (len(raw) - len(raw.lstrip()))
            chunks.append({"text": stripped, "start": start, "end": start + len(stripped)})
    if not chunks and (text or "").strip():
        stripped = text.strip()
        start = (text or "").find(stripped)
        chunks.append({"text": stripped, "start": max(0, start), "end": max(0, start) + len(stripped)})
    return chunks


async def _prebake_tts_for_message(message_id: int, speaker: str, text: str) -> dict:
    key = _get_openai_key()
    chunks = _split_sentences_with_offsets(text)
    manifest = {"message_id": message_id, "speaker": speaker, "chunks": []}
    if not key or not chunks:
        TTS_MANIFESTS[message_id] = manifest
        return manifest

    preferred_voice = CURRENT_VOICE_PREFS.get(speaker) or DEFAULT_TTS_VOICE.get(speaker) or "alloy"
    voice = preferred_voice if preferred_voice in OPENAI_TTS_VOICES else "alloy"
    model = os.environ.get("OPENAI_TTS_MODEL", "").strip() or "gpt-4o-mini-tts"
    try:
        speed = float(os.environ.get("OPENAI_TTS_SPEED", "1.1"))
    except Exception:
        speed = 1.1

    client = AsyncOpenAI(api_key=key)

    async def synth(idx: int, chunk: dict) -> dict:
        chunk_id = f"{message_id}_{idx}_{uuid.uuid4().hex[:8]}"
        item = {
            "chunk_id": chunk_id,
            "index": idx,
            "text": chunk["text"],
            "start": chunk["start"],
            "end": chunk["end"],
            "ready": False,
        }
        try:
            audio = await client.audio.speech.create(
                model=model,
                voice=voice,
                input=chunk["text"],
                response_format="mp3",
                speed=speed,
            )
            data = await audio.aread()
            TTS_CHUNKS[chunk_id] = data
            item["ready"] = True
        except Exception as exc:
            item["error"] = str(exc)
        return item

    results = await asyncio.gather(*(synth(i, c) for i, c in enumerate(chunks)))
    manifest["chunks"] = results
    TTS_MANIFESTS[message_id] = manifest
    return manifest


async def _on_agent_message(msg: dict) -> None:
    sender = msg.get("sender")
    if not sender or sender == "System":
        return
    text = (msg.get("content") or "").strip()
    if not text:
        return
    message_id = int(msg.get("id") or 0)
    if message_id <= 0:
        return
    await _prebake_tts_for_message(message_id=message_id, speaker=sender, text=text)


def _augment_state(state: dict) -> dict:
    _refresh_speaking_window(state)
    _advance_timer_tick(state)
    state["paused"] = PAUSED
    state["grades"] = LATEST_GRADES
    state["seminar_started_at"] = SEMINAR_STARTED_AT
    state["seminar_duration_seconds"] = SEMINAR_DURATION_SECONDS
    state["seconds_remaining"] = _seconds_remaining()
    state["seminar_active"] = SEMINAR_ACTIVE
    state["timer_expired"] = TIMER_EXPIRED
    state["selected_agents"] = list(CURRENT_AGENTS)
    state["selected_topics"] = list(CURRENT_TOPICS)
    state["selected_chapters"] = list(CURRENT_CHAPTERS)
    state["voice_prefs"] = dict(CURRENT_VOICE_PREFS)
    state["candidate"] = agents.get_candidate_state()
    state["reflection_status"] = dict(REFLECTION_STATUS)
    state["grader_name"] = GRADER_NAME
    state["grader_status"] = GRADER_STATUS
    state["reflection_message_ids"] = list(REFLECTION_MESSAGE_IDS)
    state["grader_message_ids"] = list(GRADER_MESSAGE_IDS)
    state["prefetch_draft"] = agents.get_prefetch_draft()
    active = state.get("active") or {}
    state["turn_queue_order"] = agents.get_turn_queue_order(state.get("agents") or [], active.get("agent"))
    return state


async def start_default_seminar(
    selected_agents: List[str] | None = None,
    selected_topics: List[str] | None = None,
    selected_chapters: List[str] | None = None,
) -> None:
    global SEMINAR_STARTED_AT, SEMINAR_RUN_ID, LATEST_GRADES, PAUSED, PAUSED_AT, TOTAL_PAUSED_SECONDS
    global SEMINAR_REMAINING_SECONDS, LAST_TIMER_TICK_TS, SPEAKING_UNTIL_TS, LAST_SPOKEN_MESSAGE_ID, TIMER_EXPIRED
    global CURRENT_AGENTS, CURRENT_TOPICS, CURRENT_CHAPTERS, SEMINAR_ACTIVE, REFLECTION_STATUS, GRADER_STATUS
    global REFLECTION_MESSAGE_IDS, GRADER_MESSAGE_IDS
    valid_agents = [name for name in DEFAULT_AGENTS if name in (selected_agents or DEFAULT_AGENTS)]
    if not valid_agents:
        valid_agents = list(DEFAULT_AGENTS)
    valid_topics = [name for name in DEFAULT_TOPICS if name in (selected_topics or DEFAULT_TOPICS)]
    if not valid_topics:
        valid_topics = list(DEFAULT_TOPICS)
    valid_chapters = [cid for cid in DEFAULT_CHAPTERS if cid in (selected_chapters or DEFAULT_CHAPTERS)]
    if not valid_chapters:
        valid_chapters = list(DEFAULT_CHAPTERS)
    CURRENT_AGENTS = valid_agents
    CURRENT_TOPICS = valid_topics
    CURRENT_CHAPTERS = valid_chapters
    agents.reset_turn_order()
    agents.reset_pipeline_state()
    TTS_MANIFESTS.clear()
    TTS_CHUNKS.clear()
    agents.set_topic_focus(CURRENT_TOPICS)
    agents.set_selected_chapters(CURRENT_CHAPTERS)
    SEMINAR_STARTED_AT = time.time()
    SEMINAR_RUN_ID += 1
    LATEST_GRADES = None
    SEMINAR_ACTIVE = True
    SEMINAR_REMAINING_SECONDS = float(SEMINAR_DURATION_SECONDS)
    LAST_TIMER_TICK_TS = time.time()
    SPEAKING_UNTIL_TS = 0.0
    LAST_SPOKEN_MESSAGE_ID = 0
    TIMER_EXPIRED = False
    REFLECTION_STATUS = {}
    GRADER_STATUS = "idle"
    REFLECTION_MESSAGE_IDS = []
    GRADER_MESSAGE_IDS = []
    PAUSED = False
    PAUSED_AT = 0.0
    TOTAL_PAUSED_SECONDS = 0.0
    agents.set_paused(False)
    agents.set_turn_intake_enabled(True)
    await db.set_room_agents("main", ",".join(CURRENT_AGENTS))
    await db.insert_message(
        "System",
        f"{len(CURRENT_AGENTS)} seminar participants will discuss the selected text materials collaboratively. "
        f"Selected sections: {', '.join(agents.get_selected_chapter_labels())}. "
        f"Focus areas: {', '.join(CURRENT_TOPICS)}. "
        "Speakers are called in a randomized rotating order. The queue display reflects the actual speaking order for this session. "
        "On your turn, contribute if you have something meaningful; otherwise you may pass so another speaker can continue the point. "
        "The text has been preloaded for everyone, so ground claims in it, cite or paraphrase specifics, challenge assumptions, and ask follow-up questions. "
        "The 20-minute timer runs during speaking and pauses while models are thinking. When time expires, the judge grades everyone.",
        room="main",
    )
    asyncio.create_task(seminar_timer_loop(SEMINAR_RUN_ID))


async def complete_and_grade_seminar(auto: bool = False) -> dict:
    global PAUSED, LATEST_GRADES, SEMINAR_ACTIVE, REFLECTION_STATUS, GRADER_STATUS
    global REFLECTION_MESSAGE_IDS, GRADER_MESSAGE_IDS
    async with _grading_lock:
        if LATEST_GRADES is not None:
            return LATEST_GRADES

        PAUSED = True
        SEMINAR_ACTIVE = False
        agents.set_paused(True)
        agents.set_turn_intake_enabled(False)
        agents.reset_pipeline_state()
        await db.cancel_active_turn(room="main")
        await db.cancel_pending_requests(room="main")
        if auto:
            await db.insert_message(
                "System",
                "The discussion phase is closed. Sending the full transcript to the grader now.",
                room="main",
            )
            REFLECTION_STATUS = {}
            await broadcast_state()
        GRADER_STATUS = "grading"
        LOGGER.info("Grading started with %s", GRADER_NAME)
        await broadcast_state()
        LATEST_GRADES = await agents.grade_seminar()
        GRADER_STATUS = "done"
        LOGGER.info("Grading complete with %s", GRADER_NAME)
        if isinstance(LATEST_GRADES, dict) and LATEST_GRADES:
            lines = ["Grader feedback:"]
            for name, grade in LATEST_GRADES.items():
                score = grade.get("score", "?") if isinstance(grade, dict) else "?"
                feedback = grade.get("feedback", "") if isinstance(grade, dict) else ""
                lines.append(f"{name} ({score}/100): {feedback}")
            await db.insert_message("System", "\n".join(lines), room="main")
            msg = await db.get_last_message(room="main")
            if msg and msg.get("id"):
                GRADER_MESSAGE_IDS.append(int(msg["id"]))
            LOGGER.info("Grader feedback summary: %s", " | ".join(lines[1:]))
        await broadcast_state()
        return LATEST_GRADES


async def seminar_timer_loop(run_id: int) -> None:
    global TIMER_EXPIRED
    while True:
        await asyncio.sleep(1.0)
        if run_id != SEMINAR_RUN_ID:
            return
        if LATEST_GRADES is not None:
            return
        state = await db.get_state(room="main")
        _refresh_speaking_window(state)
        _advance_timer_tick(state)
        if (not PAUSED) and _seconds_remaining() <= 0:
            if not TIMER_EXPIRED:
                TIMER_EXPIRED = True
                # Stop scheduling new turns, but let the current turn finish and be spoken.
                agents.set_turn_intake_enabled(False)
                agents.abort_candidate_thinking()
                await db.insert_message(
                    "System",
                    "Time is up. No new discussion turns will be scheduled. The current speaker may finish, then we move to reflections and grading.",
                    room="main",
                )
                await broadcast_state()
            active = state.get("active") or {}
            if active.get("agent"):
                continue
            if time.time() < SPEAKING_UNTIL_TS:
                continue
            await complete_and_grade_seminar(auto=True)
            return


async def inactivity_shutdown_loop() -> None:
    while True:
        await asyncio.sleep(30.0)
        idle = time.time() - LAST_API_ACTIVITY_TS
        if idle >= AUTO_SHUTDOWN_IDLE_SECONDS:
            os._exit(0)


async def broadcast_state() -> None:
    async with _broadcast_lock:
        state = await db.get_state(room="main")
        state = _augment_state(state)
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
    global PAUSED, PAUSED_AT, TOTAL_PAUSED_SECONDS, SEMINAR_ACTIVE, SEMINAR_STARTED_AT, LATEST_GRADES, CURRENT_VOICE_PREFS, CURRENT_CHAPTERS
    global SEMINAR_REMAINING_SECONDS, LAST_TIMER_TICK_TS, SPEAKING_UNTIL_TS, LAST_SPOKEN_MESSAGE_ID, TIMER_EXPIRED, REFLECTION_STATUS, GRADER_STATUS
    global REFLECTION_MESSAGE_IDS, GRADER_MESSAGE_IDS
    await db.init_db()
    await db.reset_db()
    PAUSED = True
    PAUSED_AT = 0.0
    TOTAL_PAUSED_SECONDS = 0.0
    SEMINAR_ACTIVE = False
    SEMINAR_STARTED_AT = 0.0
    SEMINAR_REMAINING_SECONDS = float(SEMINAR_DURATION_SECONDS)
    LAST_TIMER_TICK_TS = time.time()
    SPEAKING_UNTIL_TS = 0.0
    LAST_SPOKEN_MESSAGE_ID = 0
    TIMER_EXPIRED = False
    REFLECTION_STATUS = {}
    GRADER_STATUS = "idle"
    REFLECTION_MESSAGE_IDS = []
    GRADER_MESSAGE_IDS = []
    LATEST_GRADES = None
    CURRENT_VOICE_PREFS = await db.get_voice_prefs()
    CURRENT_CHAPTERS = list(DEFAULT_CHAPTERS)
    agents.set_selected_chapters(CURRENT_CHAPTERS)
    agents.set_paused(True)
    agents.set_turn_intake_enabled(False)
    agents.set_post_message_hook(_on_agent_message)
    await db.set_room_agents("main", ",".join(DEFAULT_AGENTS))
    agents.set_topic_focus(DEFAULT_TOPICS)
    await agents.start_agents(broadcast_state)
    asyncio.create_task(inactivity_shutdown_loop())


@app.middleware("http")
async def track_http_activity(request: Request, call_next):
    _touch_activity()
    response = await call_next(request)
    # Prevent stale frontend assets/state after code reloads; each refresh gets fresh files.
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


signal.signal(signal.SIGINT, lambda *_: os._exit(0))


@app.get("/")
async def root():
    return FileResponse("app/static/index.html")


@app.get("/api/state")
async def api_state():
    state = await db.get_state(room="main")
    state = _augment_state(state)
    return JSONResponse(state)


@app.get("/api/options")
async def api_options():
    return JSONResponse(
        {
            "agents": list(DEFAULT_AGENTS),
            "topics": list(DEFAULT_TOPICS),
            "chapters": agents.get_available_chapters(),
            "selected_agents": list(CURRENT_AGENTS),
            "selected_topics": list(CURRENT_TOPICS),
            "selected_chapters": list(CURRENT_CHAPTERS),
            "seminar_active": SEMINAR_ACTIVE,
            "voice_prefs": dict(CURRENT_VOICE_PREFS),
        }
    )


@app.post("/api/start")
async def api_start(payload: dict):
    selected_agents = payload.get("agents") or []
    selected_topics = payload.get("topics") or []
    selected_chapters = payload.get("chapters") or []
    await db.reset_db()
    await start_default_seminar(
        selected_agents=selected_agents,
        selected_topics=selected_topics,
        selected_chapters=selected_chapters,
    )
    await broadcast_state()
    state = await db.get_state(room="main")
    state = _augment_state(state)
    state["seconds_remaining"] = SEMINAR_DURATION_SECONDS
    return JSONResponse(state)


@app.post("/api/reset")
async def api_reset():
    await db.reset_db()
    await start_default_seminar(
        selected_agents=CURRENT_AGENTS,
        selected_topics=CURRENT_TOPICS,
        selected_chapters=CURRENT_CHAPTERS,
    )
    await broadcast_state()
    state = await db.get_state(room="main")
    state = _augment_state(state)
    state["seconds_remaining"] = SEMINAR_DURATION_SECONDS
    return JSONResponse(state)


@app.post("/api/end")
async def api_end():
    global PAUSED, PAUSED_AT, TOTAL_PAUSED_SECONDS, SEMINAR_ACTIVE, LATEST_GRADES, SEMINAR_STARTED_AT
    global SEMINAR_REMAINING_SECONDS, LAST_TIMER_TICK_TS, SPEAKING_UNTIL_TS, LAST_SPOKEN_MESSAGE_ID, TIMER_EXPIRED, REFLECTION_STATUS, GRADER_STATUS
    global REFLECTION_MESSAGE_IDS, GRADER_MESSAGE_IDS
    PAUSED = True
    PAUSED_AT = 0.0
    TOTAL_PAUSED_SECONDS = 0.0
    SEMINAR_ACTIVE = False
    LATEST_GRADES = None
    SEMINAR_STARTED_AT = 0.0
    SEMINAR_REMAINING_SECONDS = float(SEMINAR_DURATION_SECONDS)
    LAST_TIMER_TICK_TS = time.time()
    SPEAKING_UNTIL_TS = 0.0
    LAST_SPOKEN_MESSAGE_ID = 0
    TIMER_EXPIRED = False
    REFLECTION_STATUS = {}
    GRADER_STATUS = "idle"
    REFLECTION_MESSAGE_IDS = []
    GRADER_MESSAGE_IDS = []
    agents.set_paused(True)
    agents.set_turn_intake_enabled(False)
    agents.reset_pipeline_state()
    agents.clear_runtime_caches()
    TTS_MANIFESTS.clear()
    TTS_CHUNKS.clear()
    await db.cancel_active_turn(room="main")
    await db.cancel_pending_requests(room="main")
    await db.reset_db()
    await db.set_room_agents("main", ",".join(CURRENT_AGENTS))
    await broadcast_state()
    state = await db.get_state(room="main")
    state = _augment_state(state)
    state["seconds_remaining"] = SEMINAR_DURATION_SECONDS
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


@app.post("/api/timeup")
async def api_timeup():
    global SEMINAR_REMAINING_SECONDS, TIMER_EXPIRED
    SEMINAR_REMAINING_SECONDS = 30.0
    TIMER_EXPIRED = False
    if SEMINAR_ACTIVE and (not PAUSED):
        agents.set_turn_intake_enabled(True)
    await broadcast_state()
    return JSONResponse({"ok": True, "seconds_remaining": 30})


@app.post("/api/tts/prebake")
async def api_tts_prebake(payload: dict):
    message_id = int(payload.get("message_id") or 0)
    speaker = (payload.get("speaker") or "").strip()
    text = (payload.get("text") or "").strip()
    if message_id <= 0 or not speaker or not text:
        return JSONResponse({"error": "missing message_id/speaker/text"}, status_code=400)
    manifest = await _prebake_tts_for_message(message_id=message_id, speaker=speaker, text=text)
    return JSONResponse(manifest)


@app.get("/api/tts/manifest/{message_id}")
async def api_tts_manifest(message_id: int):
    manifest = TTS_MANIFESTS.get(int(message_id))
    if not manifest:
        return JSONResponse({"error": "manifest not found"}, status_code=404)
    return JSONResponse(manifest)


@app.get("/api/tts/chunk/{chunk_id}")
async def api_tts_chunk(chunk_id: str):
    data = TTS_CHUNKS.get(chunk_id)
    if not data:
        return JSONResponse({"error": "chunk not found"}, status_code=404)
    return Response(content=data, media_type="audio/mpeg")


@app.post("/api/tts")
async def api_tts(payload: dict):
    text = (payload.get("text") or "").strip()
    speaker = (payload.get("speaker") or "").strip()
    if not SEMINAR_ACTIVE:
        return JSONResponse({"error": "seminar not active"}, status_code=400)
    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)

    key = _get_openai_key()
    if not key:
        return JSONResponse({"error": "OPENAI_API_KEY missing"}, status_code=500)

    preferred_voice = CURRENT_VOICE_PREFS.get(speaker) or DEFAULT_TTS_VOICE.get(speaker) or "alloy"
    voice = preferred_voice if preferred_voice in OPENAI_TTS_VOICES else "alloy"
    model = os.environ.get("OPENAI_TTS_MODEL", "").strip() or "gpt-4o-mini-tts"
    try:
        speed = float(os.environ.get("OPENAI_TTS_SPEED", "1.1"))
    except Exception:
        speed = 1.3

    client = AsyncOpenAI(api_key=key)
    try:
        audio = await client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format="mp3",
            speed=speed,
        )
        content = await audio.aread()
        return Response(content=content, media_type="audio/mpeg")
    except Exception as exc:
        return JSONResponse({"error": f"TTS failed: {exc}"}, status_code=502)


@app.get("/api/export_txt")
async def api_export_txt():
    state = await db.get_state(room="main")
    grades = LATEST_GRADES or {}
    lines = []
    lines.append("Socratic Seminar Export")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Participants: {', '.join(state.get('agents') or [])}")
    lines.append(f"Focus Topics: {', '.join(CURRENT_TOPICS)}")
    lines.append(f"Selected Chapters: {', '.join(agents.get_selected_chapter_labels())}")
    lines.append("")
    lines.append("Transcript")
    lines.append("-" * 80)
    for msg in state.get("messages") or []:
        ts = datetime.fromtimestamp(msg.get("ts") or 0).strftime("%H:%M:%S")
        sender = msg.get("sender") or "Unknown"
        content = (msg.get("content") or "").strip()
        lines.append(f"[{ts}] {sender}:")
        lines.append(content)
        lines.append("")
    lines.append("Grades")
    lines.append("-" * 80)
    if grades:
        for name, g in grades.items():
            score = g.get("score", "?") if isinstance(g, dict) else "?"
            feedback = g.get("feedback", "") if isinstance(g, dict) else str(g)
            lines.append(f"{name}: {score}/100")
            lines.append(f"Feedback: {feedback}")
            lines.append("")
    else:
        lines.append("No grades available yet. Click 'Grade Seminar' first.")
        lines.append("")
    text = "\n".join(lines)
    filename = f"seminar_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    headers = {"Content-Disposition": f'attachment; filename=\"{filename}\"'}
    return Response(content=text.encode("utf-8"), media_type="text/plain; charset=utf-8", headers=headers)


@app.post("/api/shutdown")
async def api_shutdown():
    os._exit(0)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _touch_activity()
    _clients.append(ws)
    state = await db.get_state(room="main")
    state = _augment_state(state)
    await ws.send_text(json.dumps(state))
    try:
        while True:
            message = await ws.receive_text()
            stripped = message.strip().lower()
            if stripped == "done_speaking":
                agents.signal_speaking_done()
            elif stripped != "ping":
                _touch_activity()
    except WebSocketDisconnect:
        pass
    finally:
        if ws in _clients:
            _clients.remove(ws)
