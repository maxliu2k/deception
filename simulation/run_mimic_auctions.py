"""Run N mimic-loadout auctions via the same HTTP API the UI uses.

Each auction:
  1. Creates a save slot in the target folder
  2. POSTs /api/reset with the mimic loadout (same as clicking the preset)
  3. POSTs /api/step (same as clicking "Run Full Auction")
  4. Polls /api/step_status until done

Auctions run in parallel (each slot has its own step worker subprocess) up
to --parallel concurrent runs.

Usage:
    python -m simulation.run_mimic_auctions --count 30 --parallel 6 --folder "Mimic Auctions"
"""
from __future__ import annotations

import argparse
import random
import sys
import time
import urllib.request
import urllib.error
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


MIMIC_LOADOUT = ["Mimic-Grok", "Mimic-Opus", "Mimic-GPT-5.4", "Mimic-Pro", "Mimic-Llama"]


def _request(method: str, base: str, path: str, body: dict | None = None, timeout: float = 30.0) -> dict:
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
    slot_name: str,
    loadout: list[str],
    poll_interval_s: float,
    timeout_s: float,
) -> dict:
    # Create slot in the target folder
    create_resp = post(base, "/api/save_slot_create", {"name": slot_name, "folder_id": folder_id})
    slot = create_resp.get("slot") or {}
    slot_id = slot.get("slot_id")
    if not slot_id:
        return {"slot_name": slot_name, "ok": False, "error": f"create failed: {create_resp}"}

    # Reset with provided mimic loadout ordering
    reset_payload = {
        "selected_models": loadout,
        "mode": "open_painting_auction",
        "seed": int(seed),
        "max_rounds": 20,
        "save_slot": slot_id,
    }
    reset_resp = post(base, "/api/reset", reset_payload)
    if reset_resp.get("ok") is not True and reset_resp.get("_status", 200) >= 400:
        return {"slot_id": slot_id, "ok": False, "error": f"reset failed: {reset_resp}"}

    # Kick off the auction
    step_resp = post(base, "/api/step", {"use_models": True, "save_slot": slot_id})
    if step_resp.get("ok") is not True and step_resp.get("_status", 200) >= 400:
        return {"slot_id": slot_id, "ok": False, "error": f"step failed: {step_resp}"}

    # Poll for completion
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
    p.add_argument("--count", type=int, default=30)
    p.add_argument("--folder", default="Mimic Auctions")
    p.add_argument("--start-seed", type=int, default=1)
    p.add_argument("--slot-prefix", default="Mimic Auction")
    p.add_argument("--name-start", type=int, default=1,
                   help="Starting index for slot display names. e.g. --name-start 11 produces 'Auction 11', 'Auction 12', ...")
    p.add_argument("--poll-interval", type=float, default=1.0)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--parallel", type=int, default=6, help="Concurrent auctions.")
    p.add_argument("--permutation-seed", type=int, default=0,
                   help="Seed used to randomly sample N distinct permutations of the loadout.")
    p.add_argument("--loadout", default=",".join(MIMIC_LOADOUT),
                   help="Comma-separated list of model aliases to use as the 5-bidder loadout.")
    args = p.parse_args()

    loadout = [m.strip() for m in args.loadout.split(",") if m.strip()]
    if len(loadout) != 5:
        print(f"ERROR: loadout must have exactly 5 models, got {len(loadout)}: {loadout}")
        return 1

    folder_id = find_or_create_folder(args.base, args.folder)
    print(f"Folder '{args.folder}' = {folder_id}")
    print(f"Loadout: {loadout}")
    print(f"Running {args.count} auctions, {args.parallel} at a time")

    # Sample distinct random permutations of the loadout. With 5 models there
    # are 120 possible orderings; we sample without replacement up to that cap.
    import itertools
    all_perms = [list(p) for p in itertools.permutations(loadout)]
    perm_rng = random.Random(args.permutation_seed)
    perm_rng.shuffle(all_perms)
    if args.count > len(all_perms):
        # If they ask for more than 120, allow repeats after exhausting unique perms.
        extra = args.count - len(all_perms)
        loadouts = list(all_perms) + perm_rng.choices(all_perms, k=extra)
    else:
        loadouts = all_perms[: args.count]

    print_lock = threading.Lock()
    completed = {"n": 0}

    def _task(i: int) -> dict:
        seed = args.start_seed + i
        slot_name = f"{args.slot_prefix} {args.name_start + i}"
        loadout = loadouts[i]
        t0 = time.time()
        result = run_one(
            args.base,
            folder_id=folder_id,
            seed=seed,
            slot_name=slot_name,
            loadout=loadout,
            poll_interval_s=args.poll_interval,
            timeout_s=args.timeout,
        )
        dur = time.time() - t0
        result.update({"seed": seed, "slot_name": slot_name, "duration_s": round(dur, 1)})
        with print_lock:
            completed["n"] += 1
            tag = "ok " if result.get("ok") else "FAIL"
            extra = f"slot={result.get('slot_id')}" if result.get("ok") else f"error={result.get('error')}"
            print(f"  [{completed['n']}/{args.count}] {tag} seed={seed} '{slot_name}'  {dur:.1f}s  {extra}")
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
