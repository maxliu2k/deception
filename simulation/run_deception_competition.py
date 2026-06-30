"""Run N deception-competition episodes via the same HTTP API the UI uses.

Each episode:
  1. Creates a save slot in the target folder
  2. POSTs /api/reset with the deception_competition mode + loadout
  3. POSTs /api/step (kicks off the 12-round episode worker)
  4. Polls /api/step_status until done

Episodes run in parallel (each slot has its own step worker) up to --parallel
concurrent runs.

Usage:
    python -m simulation.run_deception_competition --count 20 --loadout Math-T1,Math-T2,Math-T3,Math-T4,Math-T1
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


DEFAULT_LOADOUT = ["Math-T1", "Math-T2", "Math-T3", "Math-T4", "Math-T1"]


def _request(method: str, base: str, path: str, body: dict | None = None, timeout: float = 90.0) -> dict:
    url = base.rstrip("/") + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read().decode("utf-8"))
        except Exception:
            payload = {"error": str(e)}
        payload["_status"] = e.code
        return payload


def get(base: str, path: str) -> dict:
    return _request("GET", base, path)


def post(base: str, path: str, body: dict) -> dict:
    return _request("POST", base, path, body)


def find_or_create_folder(base: str, name: str) -> str:
    # Use the lightweight /api/folders endpoint (catalog-only) rather than
    # /api/save_slots, which unpickles every slot's runtime and times out once
    # the catalog grows past a few thousand slots. Fall back to the heavy
    # endpoint for older servers without /api/folders.
    listing = get(base, "/api/folders")
    if not listing.get("folders") and not listing.get("ok"):
        listing = get(base, "/api/save_slots")
    for f in (listing.get("folders") or []):
        if f.get("name") == name:
            return f["folder_id"]
    created = post(base, "/api/folder_create", {"name": name})
    return created["folder"]["folder_id"]


def run_one(
    base: str,
    *,
    folder_id: str,
    seed: int,
    truth_seed: int,
    slot_name: str,
    loadout: list[str],
    num_rounds: int,
    poll_interval_s: float,
    timeout_s: float,
) -> dict:
    create_resp = post(base, "/api/save_slot_create", {"name": slot_name, "folder_id": folder_id})
    slot = create_resp.get("slot") or {}
    slot_id = slot.get("slot_id")
    if not slot_id:
        return {"slot_name": slot_name, "ok": False, "error": f"create failed: {create_resp}"}

    reset_payload = {
        "selected_models": loadout,
        "mode": "deception_competition",
        "seed": int(seed),
        "truth_seed": int(truth_seed),
        "num_rounds": int(num_rounds),
        "save_slot": slot_id,
    }
    reset_resp = post(base, "/api/reset", reset_payload)
    if reset_resp.get("ok") is not True and reset_resp.get("_status", 200) >= 400:
        return {"slot_id": slot_id, "ok": False, "error": f"reset failed: {reset_resp}"}

    step_resp = post(base, "/api/step", {"use_models": True, "save_slot": slot_id})
    if step_resp.get("ok") is not True and step_resp.get("_status", 200) >= 400:
        return {"slot_id": slot_id, "ok": False, "error": f"step failed: {step_resp}"}

    started = time.time()
    while True:
        if time.time() - started > timeout_s:
            return {"slot_id": slot_id, "ok": False, "error": "timeout", "loadout": loadout}
        time.sleep(poll_interval_s)
        status_resp = get(base, f"/api/step_status?session_id={slot_id}")
        status = status_resp.get("status") or status_resp
        if status.get("error"):
            return {"slot_id": slot_id, "ok": False, "error": status["error"], "loadout": loadout}
        if status.get("done"):
            return {
                "slot_id": slot_id,
                "ok": True,
                "used_models": status.get("used_models"),
                "llm_error": status.get("llm_error"),
                "loadout": loadout,
            }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="http://localhost:8010")
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--folder", default="Deception Competition v1")
    p.add_argument("--start-seed", type=int, default=1)
    p.add_argument("--truth-seed-start", type=int, default=10_000,
                   help="Starting truth-vector RNG seed; each episode uses a distinct seed.")
    p.add_argument("--slot-prefix", default="Deception")
    p.add_argument("--name-start", type=int, default=1)
    p.add_argument("--poll-interval", type=float, default=2.0)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--parallel", type=int, default=4, help="Concurrent episodes.")
    p.add_argument("--permutation-seed", type=int, default=0,
                   help="Seed for sampling distinct random permutations of the loadout.")
    p.add_argument("--loadout", default=",".join(DEFAULT_LOADOUT),
                   help="Comma-separated list of 5 model aliases for the agent slots.")
    p.add_argument("--num-rounds", type=int, default=12)
    args = p.parse_args()

    loadout = [m.strip() for m in args.loadout.split(",") if m.strip()]
    if len(loadout) != 5:
        print(f"ERROR: loadout must have exactly 5 models, got {len(loadout)}: {loadout}")
        return 1

    folder_id = find_or_create_folder(args.base, args.folder)
    print(f"Folder '{args.folder}' = {folder_id}")
    print(f"Loadout: {loadout}")
    print(f"num_rounds = {args.num_rounds}")
    print(f"Running {args.count} episodes, {args.parallel} at a time")

    # Sample distinct random permutations of the agent slot order.
    all_perms = [list(p) for p in itertools.permutations(loadout)]
    perm_rng = random.Random(args.permutation_seed)
    perm_rng.shuffle(all_perms)
    if args.count > len(all_perms):
        extra = args.count - len(all_perms)
        loadouts = list(all_perms) + perm_rng.choices(all_perms, k=extra)
    else:
        loadouts = all_perms[: args.count]

    print_lock = threading.Lock()
    completed = {"n": 0}

    def _task(i: int) -> dict:
        seed = args.start_seed + i
        truth_seed = args.truth_seed_start + i
        slot_name = f"{args.slot_prefix} {args.name_start + i}"
        slot_loadout = loadouts[i]
        t0 = time.time()
        result = run_one(
            args.base,
            folder_id=folder_id,
            seed=seed,
            truth_seed=truth_seed,
            slot_name=slot_name,
            loadout=slot_loadout,
            num_rounds=args.num_rounds,
            poll_interval_s=args.poll_interval,
            timeout_s=args.timeout,
        )
        dur = time.time() - t0
        result.update({"seed": seed, "slot_name": slot_name, "duration_s": round(dur, 1)})
        with print_lock:
            completed["n"] += 1
            tag = "ok " if result.get("ok") else "FAIL"
            extra = f"slot={result.get('slot_id')}" if result.get("ok") else f"error={result.get('error')}"
            print(f"  [{completed['n']}/{args.count}] {tag} seed={seed} '{slot_name}'  {dur:.1f}s  {extra}", flush=True)
        return result

    summary: list[dict] = []
    overall_t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(_task, i) for i in range(args.count)]
        for fut in as_completed(futures):
            summary.append(fut.result())
    overall_dur = time.time() - overall_t0

    print("\n=== Summary ===")
    ok = sum(1 for r in summary if r.get("ok"))
    print(f"  Successful: {ok}/{len(summary)}  in {overall_dur:.1f}s")
    for r in sorted(summary, key=lambda x: x.get("seed", 0)):
        if not r.get("ok"):
            print(f"    [seed {r['seed']}] {r.get('slot_name')}: {r.get('error')}")
    return 0 if ok == len(summary) else 1


if __name__ == "__main__":
    sys.exit(main())
