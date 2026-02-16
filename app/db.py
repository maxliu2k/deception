import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite


DB_PATH = Path(__file__).resolve().parents[1] / "arena.sqlite"


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL,
    sender TEXT,
    content TEXT,
    room TEXT DEFAULT 'main',
    visibility TEXT DEFAULT 'all'
);

CREATE TABLE IF NOT EXISTS turn_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL,
    agent TEXT,
    reason TEXT,
    status TEXT,
    priority INTEGER,
    room TEXT DEFAULT 'main'
);

CREATE TABLE IF NOT EXISTS active_turns (
    room TEXT PRIMARY KEY,
    agent TEXT,
    turn_request_id INTEGER,
    ts REAL
);

CREATE TABLE IF NOT EXISTS rooms (
    name TEXT PRIMARY KEY,
    agents TEXT
);

CREATE TABLE IF NOT EXISTS mafia_game (
    id INTEGER PRIMARY KEY CHECK(id=1),
    status TEXT,
    phase TEXT,
    day INTEGER,
    last_update REAL,
    paused INTEGER,
    last_doctor_target TEXT
);

CREATE TABLE IF NOT EXISTS mafia_players (
    agent TEXT PRIMARY KEY,
    role TEXT,
    alive INTEGER,
    revealed TEXT
);

CREATE TABLE IF NOT EXISTS mafia_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    day INTEGER,
    phase TEXT,
    actor TEXT,
    action TEXT,
    target TEXT,
    ts REAL
);
"""


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(SCHEMA_SQL)
        await _ensure_column(db, "messages", "room", "TEXT", "'main'")
        await _ensure_column(db, "messages", "visibility", "TEXT", "'all'")
        await _ensure_column(db, "turn_requests", "room", "TEXT", "'main'")
        await db.execute(
            "INSERT OR IGNORE INTO active_turns (room, agent, turn_request_id, ts) VALUES ('main', NULL, NULL, NULL)"
        )
        await db.execute(
            "INSERT OR IGNORE INTO rooms (name, agents) VALUES ('main', 'A,B,C,D,E,F,G,H,I,J')"
        )
        await db.execute(
            "INSERT OR IGNORE INTO mafia_game (id, status, phase, day, last_update, paused, last_doctor_target) "
            "VALUES (1, 'idle', 'setup', 0, NULL, 0, NULL)"
        )
        await db.commit()


async def _ensure_column(db, table: str, column: str, col_type: str, default_sql: str) -> None:
    async with db.execute(f"PRAGMA table_info({table})") as cursor:
        cols = [row[1] async for row in cursor]
    if column not in cols:
        await db.execute(
            f"ALTER TABLE {table} ADD COLUMN {column} {col_type} DEFAULT {default_sql}"
        )


async def reset_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM messages")
        await db.execute("DELETE FROM turn_requests")
        await db.execute("DELETE FROM active_turns")
        await db.execute("DELETE FROM rooms WHERE name != 'main'")
        await db.execute("DELETE FROM mafia_players")
        await db.execute("DELETE FROM mafia_actions")
        await db.execute("UPDATE mafia_game SET status='idle', phase='setup', day=0, last_update=NULL, paused=0, last_doctor_target=NULL WHERE id=1")
        await db.execute(
            "INSERT OR REPLACE INTO active_turns (room, agent, turn_request_id, ts) VALUES ('main', NULL, NULL, NULL)"
        )
        await db.commit()


async def insert_message(sender: str, content: str, room: str = "main", visibility: str = "all") -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO messages (ts, sender, content, room, visibility) VALUES (?, ?, ?, ?, ?)",
            (time.time(), sender, content, room, visibility),
        )
        await db.commit()


async def insert_turn_request(agent: str, reason: str, priority: int = 0, room: str = "main") -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT COUNT(*) FROM turn_requests WHERE agent=? AND status IN ('pending','active') AND room=?",
            (agent, room),
        ) as cursor:
            row = await cursor.fetchone()
            if row and int(row[0]) > 0:
                return False
        await db.execute(
            "INSERT INTO turn_requests (ts, agent, reason, status, priority, room) VALUES (?, ?, ?, 'pending', ?, ?)",
            (time.time(), agent, reason, priority, room),
        )
        await db.commit()
        return True


async def get_last_message(room: str = "main") -> Optional[Dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, ts, sender, content, room, visibility FROM messages WHERE room=? ORDER BY id DESC LIMIT 1",
            (room,),
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


async def get_pending_requests(room: str = "main") -> List[Dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, ts, agent, reason, status, priority FROM turn_requests "
            "WHERE status='pending' AND room=? ORDER BY priority DESC, ts ASC",
            (room,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]


async def set_active_turn(agent: Optional[str], turn_request_id: Optional[int], room: str = "main") -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO active_turns (room, agent, turn_request_id, ts) VALUES (?, ?, ?, ?)",
            (room, agent, turn_request_id, time.time() if agent else None),
        )
        await db.commit()


async def mark_turn_request_status(request_id: int, status: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE turn_requests SET status=? WHERE id=?",
            (status, request_id),
        )
        await db.commit()


async def get_active_turn(room: str = "main") -> Dict[str, Any]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT room, agent, turn_request_id, ts FROM active_turns WHERE room=?",
            (room,),
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else {"room": room, "agent": None, "turn_request_id": None, "ts": None}


async def cancel_active_turn(room: str = "main") -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT agent, turn_request_id FROM active_turns WHERE room=?",
            (room,),
        ) as cursor:
            row = await cursor.fetchone()
            if row and row["turn_request_id"]:
                await db.execute(
                    "UPDATE turn_requests SET status='cancelled' WHERE id=?",
                    (row["turn_request_id"],),
                )
        await db.execute(
            "UPDATE active_turns SET agent=NULL, turn_request_id=NULL, ts=NULL WHERE room=?",
            (room,),
        )
        await db.commit()


async def get_messages(limit: int = 200, room: str = "main", allowed_visibilities: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if allowed_visibilities is None:
            async with db.execute(
                "SELECT id, ts, sender, content, room, visibility FROM messages WHERE room=? ORDER BY id DESC LIMIT ?",
                (room, limit),
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(r) for r in reversed(rows)]

        placeholders = ",".join(["?"] * len(allowed_visibilities))
        query = (
            f"SELECT id, ts, sender, content, room, visibility FROM messages "
            f"WHERE room=? AND visibility IN ({placeholders}) ORDER BY id DESC LIMIT ?"
        )
        params = [room, *allowed_visibilities, limit]
        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in reversed(rows)]


async def get_message_count(room: str = "main") -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT COUNT(*) FROM messages WHERE room=?", (room,)) as cursor:
            row = await cursor.fetchone()
            return int(row[0]) if row else 0


async def get_state(room: str = "main") -> Dict[str, Any]:
    messages = await get_messages(room=room)
    queue = await get_pending_requests(room=room)
    active = await get_active_turn(room=room)
    return {"messages": messages, "queue": queue, "active": active, "room": room}


async def get_rooms() -> List[Dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT name, agents FROM rooms ORDER BY name ASC") as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]


async def create_room(name: str, agents: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR IGNORE INTO rooms (name, agents) VALUES (?, ?)",
            (name, agents),
        )
        await db.execute(
            "INSERT OR IGNORE INTO active_turns (room, agent, turn_request_id, ts) VALUES (?, NULL, NULL, NULL)",
            (name,),
        )
        await db.commit()


async def set_room_agents(name: str, agents: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO rooms (name, agents) VALUES (?, ?)",
            (name, agents),
        )
        await db.execute(
            "INSERT OR IGNORE INTO active_turns (room, agent, turn_request_id, ts) VALUES (?, NULL, NULL, NULL)",
            (name,),
        )
        await db.commit()


async def get_room_agents(name: str) -> List[str]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT agents FROM rooms WHERE name=?", (name,)) as cursor:
            row = await cursor.fetchone()
            if not row or not row[0]:
                return []
            return [a.strip() for a in row[0].split(",") if a.strip()]


async def get_mafia_game() -> Dict[str, Any]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, status, phase, day, last_update, paused, last_doctor_target FROM mafia_game WHERE id=1"
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else {}


async def set_mafia_game(**fields) -> None:
    if not fields:
        return
    keys = ", ".join([f"{k}=?" for k in fields.keys()])
    values = list(fields.values())
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            f"UPDATE mafia_game SET {keys}, last_update=? WHERE id=1",
            (*values, time.time()),
        )
        await db.commit()


async def set_mafia_players(players: List[Dict[str, Any]]) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM mafia_players")
        for p in players:
            await db.execute(
                "INSERT INTO mafia_players (agent, role, alive, revealed) VALUES (?, ?, ?, ?)",
                (p["agent"], p["role"], int(p["alive"]), p.get("revealed")),
            )
        await db.commit()


async def get_mafia_players() -> List[Dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT agent, role, alive, revealed FROM mafia_players") as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]


async def set_player_alive(agent: str, alive: bool) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE mafia_players SET alive=? WHERE agent=?",
            (1 if alive else 0, agent),
        )
        await db.commit()


async def set_player_revealed(agent: str, revealed: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE mafia_players SET revealed=? WHERE agent=?",
            (revealed, agent),
        )
        await db.commit()


async def log_mafia_action(day: int, phase: str, actor: str, action: str, target: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO mafia_actions (day, phase, actor, action, target, ts) VALUES (?, ?, ?, ?, ?, ?)",
            (day, phase, actor, action, target, time.time()),
        )
        await db.commit()


async def get_mafia_actions(day: int, phase: str, action: Optional[str] = None) -> List[Dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if action:
            async with db.execute(
                "SELECT id, day, phase, actor, action, target, ts FROM mafia_actions WHERE day=? AND phase=? AND action=?",
                (day, phase, action),
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(r) for r in rows]
        async with db.execute(
            "SELECT id, day, phase, actor, action, target, ts FROM mafia_actions WHERE day=? AND phase=?",
            (day, phase),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]


async def clear_mafia_actions(day: int, phase: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "DELETE FROM mafia_actions WHERE day=? AND phase=?",
            (day, phase),
        )
        await db.commit()
