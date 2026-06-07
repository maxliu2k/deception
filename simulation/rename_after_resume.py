"""Rename slots to match the loadout that kick_existing.py actually ran.

The parallel slot creation race made the original slot names inaccurate.
The TRUE mapping (loadout actually persisted via /api/reset in kick_existing.py)
is by slot_id, in the order below. This script aligns the names to that truth.
"""
from __future__ import annotations

import json
import urllib.request

BASE = "http://localhost:8010"

# (slot_id, tier, replaced_llm) -> the loadout that was actually run.
TRUE_MAPPING = [
    ("save_slot_1164", "T1", "GPT-5.4"),
    ("save_slot_1165", "T1", "Grok"),
    ("save_slot_1166", "T1", "Opus"),
    ("save_slot_1167", "T1", "Pro"),
    ("save_slot_1168", "T1", "Llama"),
    ("save_slot_1169", "T2", "GPT-5.4"),
    ("save_slot_1172", "T2", "Grok"),
    ("save_slot_1173", "T2", "Opus"),
    ("save_slot_1175", "T2", "Pro"),
    ("save_slot_1179", "T2", "Llama"),
    ("save_slot_1174", "T3", "GPT-5.4"),
    ("save_slot_1177", "T3", "Grok"),
    ("save_slot_1180", "T3", "Opus"),
    ("save_slot_1181", "T3", "Pro"),
    ("save_slot_1182", "T3", "Llama"),
    ("save_slot_1170", "T4", "GPT-5.4"),
    ("save_slot_1171", "T4", "Grok"),
    ("save_slot_1176", "T4", "Opus"),
    ("save_slot_1178", "T4", "Pro"),
    ("save_slot_1183", "T4", "Llama"),
]


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def main() -> int:
    n = 0
    for slot_id, tier, replaced in TRUE_MAPPING:
        new_name = f"{tier}-replaces-{replaced} 1"
        r = post("/api/save_slot_rename", {"slot_id": slot_id, "name": new_name})
        ok = r.get("ok")
        print(f"  {slot_id} -> {new_name!r:30}  ok={ok}")
        n += int(bool(ok))
    print(f"Renamed {n}/{len(TRUE_MAPPING)} slots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
