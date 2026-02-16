import asyncio
import json
import os
import random
import signal
from typing import Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import agents
from . import db


app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")

_clients: List[WebSocket] = []
_ws_rooms: Dict[WebSocket, str] = {}
_broadcast_lock = asyncio.Lock()
START_PROMPT = ""
PAUSED = False
START_MAFIA_ON_STARTUP = True


async def start_default_mafia() -> None:
    players = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    roles = ["mafia"] * 3 + ["doctor"] + ["sheriff"] + ["town"] * (len(players) - 5)
    random.shuffle(players)
    random.shuffle(roles)
    assignments = [{"agent": p, "role": r, "alive": True, "revealed": None} for p, r in zip(players, roles)]

    await db.set_mafia_players(assignments)
    await db.set_mafia_game(status="running", phase="night", day=1, paused=0, last_doctor_target=None)

    mafia_agents = [a["agent"] for a in assignments if a["role"] == "mafia"]
    doctor_agent = next(a["agent"] for a in assignments if a["role"] == "doctor")
    sheriff_agent = next(a["agent"] for a in assignments if a["role"] == "sheriff")

    await db.set_room_agents("main", ",".join([a["agent"] for a in assignments]))
    await db.set_room_agents("mafia", ",".join(mafia_agents))
    await db.set_room_agents("doctor", doctor_agent)
    await db.set_room_agents("sheriff", sheriff_agent)

    await db.insert_message("System", "A new game of Mafia has started. This is a game. Roles have been assigned.", room="main")
    await db.insert_message("System", f"You are Mafia. Mafia team: {', '.join(mafia_agents)}. This is a game.", room="mafia")
    await db.insert_message("System", "You are the Doctor. This is a game. Choose who to protect each night. You cannot protect the same person twice in a row.", room="doctor")
    await db.insert_message("System", "You are the Sheriff. This is a game. Choose one player to investigate each night.", room="sheriff")


async def broadcast_state() -> None:
    async with _broadcast_lock:
        dead = []
        for ws in _clients:
            room = _ws_rooms.get(ws, "main")
            state = await db.get_state(room=room)
            state["paused"] = PAUSED
            data = json.dumps(state)
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in _clients:
                _clients.remove(ws)
                _ws_rooms.pop(ws, None)


@app.on_event("startup")
async def on_startup() -> None:
    await db.init_db()
    await db.reset_db()
    await db.insert_message("System", START_PROMPT, room="main")
    if START_MAFIA_ON_STARTUP:
        await start_default_mafia()
    await agents.start_agents(broadcast_state)


signal.signal(signal.SIGINT, lambda *_: os._exit(0))


@app.get("/")
async def root():
    return FileResponse("app/static/index.html")


@app.get("/api/state")
async def api_state(room: str = "main"):
    state = await db.get_state(room=room)
    state["paused"] = PAUSED
    return JSONResponse(state)


@app.post("/api/reset")
async def api_reset():
    await db.reset_db()
    await db.insert_message("System", START_PROMPT, room="main")
    await broadcast_state()
    return JSONResponse({"ok": True})


@app.post("/api/mafia/start")
async def api_mafia_start():
    await start_default_mafia()
    await broadcast_state()
    return JSONResponse({"ok": True})


@app.get("/api/mafia/state")
async def api_mafia_state():
    return JSONResponse({"game": await db.get_mafia_game(), "players": await db.get_mafia_players()})


@app.post("/api/system_message")
async def api_system_message(payload: dict):
    content = (payload.get("content") or "").strip()
    room = (payload.get("room") or "main").strip()
    if not content:
        return JSONResponse({"ok": False, "error": "empty"}, status_code=400)
    await db.cancel_active_turn(room=room)
    await db.insert_message("System", content, room=room)
    await broadcast_state()
    return JSONResponse({"ok": True})


@app.get("/api/rooms")
async def api_rooms():
    return JSONResponse({"rooms": await db.get_rooms()})


@app.post("/api/rooms")
async def api_create_room(payload: dict):
    name = (payload.get("name") or "").strip()
    agents_list = payload.get("agents") or []
    agents_csv = ",".join([a.strip() for a in agents_list if a.strip()])
    if not name:
        return JSONResponse({"ok": False, "error": "empty name"}, status_code=400)
    await db.create_room(name, agents_csv)
    await broadcast_state()
    return JSONResponse({"ok": True})


@app.post("/api/pause")
async def api_pause(payload: dict):
    global PAUSED
    value = payload.get("paused")
    if value is None:
        PAUSED = not PAUSED
    else:
        PAUSED = bool(value)
    agents.set_paused(PAUSED)
    await broadcast_state()
    return JSONResponse({"paused": PAUSED})


@app.post("/api/shutdown")
async def api_shutdown():
    os._exit(0)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.append(ws)
    _ws_rooms[ws] = "main"
    state = await db.get_state(room="main")
    state["paused"] = PAUSED
    await ws.send_text(json.dumps(state))
    try:
        while True:
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
                if data.get("type") == "subscribe":
                    _ws_rooms[ws] = data.get("room") or "main"
            except Exception:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        if ws in _clients:
            _clients.remove(ws)
        _ws_rooms.pop(ws, None)
