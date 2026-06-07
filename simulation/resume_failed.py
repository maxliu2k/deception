"""Resume crashed T1-T4 auction slots without losing progress.

Skips /api/reset (which would wipe state) and just calls /api/step. The
step_worker loads the persisted env from runtime.pkl and continues the
`while not env.done` loop from wherever the previous crash left it.

Concurrency capped at 5 to avoid burning OpenRouter credits in a burst.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "http://localhost:8010"

# All T1-T4 slots from the parallel batch except save_slot_1175 which already
# completed successfully.
SLOTS_TO_RESUME = [
    "save_slot_1164",  # T1 replaces GPT-5.4
    "save_slot_1165",  # T1 replaces Grok
    "save_slot_1166",  # T1 replaces Opus
    "save_slot_1167",  # T1 replaces Pro
    "save_slot_1168",  # T1 replaces Llama
    "save_slot_1169",  # T2 replaces GPT-5.4
    "save_slot_1172",  # T2 replaces Grok
    "save_slot_1173",  # T2 replaces Opus
    "save_slot_1179",  # T2 replaces Llama
    "save_slot_1174",  # T3 replaces GPT-5.4
    "save_slot_1177",  # T3 replaces Grok
    "save_slot_1180",  # T3 replaces Opus
    "save_slot_1181",  # T3 replaces Pro
    "save_slot_1182",  # T3 replaces Llama
    "save_slot_1170",  # T4 replaces GPT-5.4
    "save_slot_1171",  # T4 replaces Grok
    "save_slot_1176",  # T4 replaces Opus
    "save_slot_1178",  # T4 replaces Pro
    "save_slot_1183",  # T4 replaces Llama
]


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            return {"_status": e.code, **json.loads(e.read())}
        except Exception:
            return {"_status": e.code, "detail": str(e)}


def get(path: str) -> dict:
    req = urllib.request.Request(BASE + path)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


lock = threading.Lock()
state = {"done_n": 0}


def resume_one(slot_id: str, timeout_s: float = 1200) -> tuple[str, bool, str | None, float]:
    t0 = time.time()
    try:
        step_resp = post("/api/step", {"use_models": True, "save_slot": slot_id})
        if step_resp.get("_status") == 400 and "complete" in str(step_resp.get("detail", "")).lower():
            return slot_id, True, "already_complete", 0.0
        if step_resp.get("ok") is not True:
            return slot_id, False, f"step launch failed: {step_resp}", time.time() - t0
        while time.time() - t0 < timeout_s:
            time.sleep(2.0)
            st = get(f"/api/step_status?session_id={slot_id}")
            status = st.get("status") or st
            if status.get("error"):
                return slot_id, False, status["error"], time.time() - t0
            if status.get("done"):
                return slot_id, True, None, time.time() - t0
        return slot_id, False, "timeout", time.time() - t0
    except Exception as e:
        return slot_id, False, f"{type(e).__name__}: {e}", time.time() - t0


def main() -> int:
    print(f"Resuming {len(SLOTS_TO_RESUME)} slots, parallel=5...")
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(resume_one, sid): sid for sid in SLOTS_TO_RESUME}
        for fut in as_completed(futures):
            sid, ok, err, dur = fut.result()
            with lock:
                state["done_n"] += 1
                tag = "OK  " if ok else "FAIL"
                extra = "" if ok else f"  error={err!r:.180s}"
                print(f"  [{state['done_n']}/{len(SLOTS_TO_RESUME)}] {tag} {sid}  {dur:.0f}s{extra}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
