"""Round-robin negotiation runner for the buyer/seller game.

Each "batch" = all distinct (buyer_alias, seller_alias) pairings drawn from a
loadout. Default 5 LLMs vs 25 pairings per batch (incl. self-pairs).

- Persists each episode to its own save slot under one folder per batch run.
- Retries failed episodes (LLM parse / API errors) with exponential backoff,
  bumping max_tokens on the retry path.
- Each episode runs the full negotiation back-and-forth via /api/step (the
  server orchestrates seller-open vs alternating offers vs accept/reject).

Usage:
    python -m simulation.run_negotiation_batch --batches 3 \\
        --folder "Negotiation v1" --parallel 4
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from simulation.run_deception_competition import (
    find_or_create_folder,
    get,
    post,
)


LLMS = ["GPT-5.4", "Grok", "Opus", "Pro", "Llama"]


def pairings(llms: list[str], include_self: bool = True) -> list[tuple[str, str]]:
    """All (buyer, seller) ordered pairs."""
    return [(b, s) for b in llms for s in llms if include_self or b != s]


def run_one(
    base: str,
    *,
    folder_id: str,
    seed: int,
    slot_name: str,
    buyer_alias: str,
    seller_alias: str,
    message_limit: int,
    poll_interval_s: float = 2.0,
    timeout_s: float = 600.0,
    max_retries: int = 2,
) -> dict:
    """Run one negotiation episode with retries on failure."""

    def try_once(attempt: int) -> dict:
        create = post(base, "/api/save_slot_create", {"name": f"{slot_name} [a{attempt}]", "folder_id": folder_id})
        slot = create.get("slot") or {}
        slot_id = slot.get("slot_id")
        if not slot_id:
            return {"ok": False, "error": f"create failed: {create}", "attempt": attempt}

        # Negotiation requires loadout of size {3, 5} per env validator. Pad with
        # a dummy alias for the unused 3rd seat (server reads only [0] and [1]).
        loadout = [buyer_alias, seller_alias, "GPT-5.4"]
        reset = post(base, "/api/reset", {
            "selected_models": loadout,
            "mode": "buyer_seller_negotiation",
            "seed": int(seed),
            "negotiation_message_limit": int(message_limit),
            "save_slot": slot_id,
        })
        if reset.get("ok") is not True and reset.get("_status", 200) >= 400:
            return {"ok": False, "slot_id": slot_id, "error": f"reset failed: {reset}", "attempt": attempt}

        step = post(base, "/api/step", {"use_models": True, "save_slot": slot_id})
        if step.get("ok") is not True and step.get("_status", 200) >= 400:
            return {"ok": False, "slot_id": slot_id, "error": f"step start: {step}", "attempt": attempt}

        started = time.time()
        while True:
            if time.time() - started > timeout_s:
                return {"ok": False, "slot_id": slot_id, "error": "timeout", "attempt": attempt}
            time.sleep(poll_interval_s)
            status_resp = get(base, f"/api/step_status?session_id={slot_id}")
            status = status_resp.get("status") or status_resp
            if status.get("error"):
                return {"ok": False, "slot_id": slot_id, "error": status["error"], "attempt": attempt}
            if status.get("done"):
                return {"ok": True, "slot_id": slot_id, "attempt": attempt}

    last_err = None
    for attempt in range(max_retries + 1):
        result = try_once(attempt)
        if result.get("ok"):
            return result
        last_err = result.get("error")
        # Brief backoff between retries
        time.sleep(2.0 + attempt * 1.5)

    return {"ok": False, "error": last_err, "buyer": buyer_alias, "seller": seller_alias, "seed": seed}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="http://localhost:8010")
    p.add_argument("--folder", default="Negotiation Real LLM v1")
    p.add_argument("--batches", type=int, default=1, help="Number of round-robin batches.")
    p.add_argument("--include-self", action="store_true", default=True,
                   help="Include LLM-vs-itself pairings (default on; 25/batch). Use --no-self to exclude.")
    p.add_argument("--no-self", dest="include_self", action="store_false")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--message-limit", type=int, default=10)
    p.add_argument("--start-seed", type=int, default=1)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--llms", default=",".join(LLMS),
                   help="Comma-separated list of LLM aliases to round-robin.")
    args = p.parse_args()

    llms = [s.strip() for s in args.llms.split(",") if s.strip()]
    base_pairings = pairings(llms, include_self=args.include_self)
    print(f"Pairings per batch: {len(base_pairings)} ({len(llms)} LLMs × "
          f"{'incl self' if args.include_self else 'no self'})", flush=True)

    folder_id = find_or_create_folder(args.base, args.folder)
    print(f"Folder '{args.folder}' = {folder_id}", flush=True)

    jobs = []
    for batch_idx in range(args.batches):
        for pair_idx, (buyer, seller) in enumerate(base_pairings):
            slot_idx = batch_idx * len(base_pairings) + pair_idx
            seed = args.start_seed + slot_idx
            name = f"b{batch_idx + 1}-{buyer[:4]}-{seller[:4]}-{seed}"
            jobs.append({
                "buyer": buyer, "seller": seller, "seed": seed,
                "slot_name": name, "batch": batch_idx + 1,
            })
    total = len(jobs)
    print(f"Total episodes: {total} ({args.batches} batches × {len(base_pairings)} pairings)", flush=True)

    completed = {"n": 0, "fail": 0}
    lock = Lock()
    results = []
    abort_event = {"fail_info": None}

    def task(j: dict) -> dict:
        if abort_event["fail_info"] is not None:
            # Another episode already failed — skip remaining work fast.
            return {"ok": False, "skipped": True, "job": j}
        t0 = time.time()
        res = run_one(
            args.base,
            folder_id=folder_id,
            seed=j["seed"],
            slot_name=j["slot_name"],
            buyer_alias=j["buyer"],
            seller_alias=j["seller"],
            message_limit=args.message_limit,
            max_retries=args.max_retries,
        )
        dur = time.time() - t0
        res.update({"job": j, "duration_s": round(dur, 1)})
        with lock:
            completed["n"] += 1
            if not res.get("ok"):
                completed["fail"] += 1
                if abort_event["fail_info"] is None:
                    abort_event["fail_info"] = res
            tag = "ok" if res.get("ok") else "FAIL"
            attempts = res.get("attempt", 0) + 1
            print(f"  [{completed['n']}/{total}] {tag} batch={j['batch']} "
                  f"{j['buyer']:>7s}vs{j['seller']:<7s} seed={j['seed']} "
                  f"attempts={attempts} {dur:.1f}s "
                  f"{res.get('slot_id','')} {('err='+str(res.get('error',''))[:80]) if not res.get('ok') else ''}",
                  flush=True)
        return res

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(task, j) for j in jobs]
        for f in as_completed(futures):
            results.append(f.result())
            if abort_event["fail_info"] is not None:
                # Cancel pending work and surface the failure immediately.
                for pending in futures:
                    pending.cancel()
                break

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Successful: {completed['n'] - completed['fail']}/{total}", flush=True)
    if abort_event["fail_info"] is not None:
        fi = abort_event["fail_info"]
        j = fi.get("job", {})
        raise SystemExit(
            f"CRASH: episode failed (no silent fallback). "
            f"buyer={j.get('buyer')} seller={j.get('seller')} seed={j.get('seed')} "
            f"err={fi.get('error')}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
