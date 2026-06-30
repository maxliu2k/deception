"""Math-tier vs Real-LLM verification runner for negotiation.

For each math tier × each LLM × each role (math as buyer / math as seller),
run N episodes. Persists each to its own save slot.

Default: 4 math tiers × 5 LLMs × 2 roles × 2 eps = 80 episodes.
"""
from __future__ import annotations

import argparse
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from simulation.run_deception_competition import find_or_create_folder, post, get


LLMS = ["GPT-5.4", "Grok", "Opus", "Pro", "Llama"]
MATH_TIERS = ["Math-Trivial-Open", "Math-Truth-Anchored", "Math-Reactive", "Math-Deadline-Aware", "Math-RL"]


def run_one(base: str, *, folder_id: str, seed: int, slot_name: str,
            buyer_alias: str, seller_alias: str,
            message_limit: int = 10, poll: float = 2.0, timeout: float = 600.0,
            max_retries: int = 2) -> dict:
    def try_once(attempt: int):
        create = post(base, "/api/save_slot_create", {"name": f"{slot_name} [a{attempt}]", "folder_id": folder_id})
        slot = create.get("slot") or {}
        sid = slot.get("slot_id")
        if not sid:
            return {"ok": False, "error": f"create: {create}", "attempt": attempt}
        reset = post(base, "/api/reset", {
            "selected_models": [buyer_alias, seller_alias, "GPT-5.4"],
            "mode": "buyer_seller_negotiation",
            "seed": int(seed),
            "negotiation_message_limit": int(message_limit),
            "save_slot": sid,
        })
        if reset.get("ok") is not True and reset.get("_status", 200) >= 400:
            return {"ok": False, "slot_id": sid, "error": f"reset: {reset}", "attempt": attempt}
        step = post(base, "/api/step", {"use_models": True, "save_slot": sid})
        if step.get("ok") is not True and step.get("_status", 200) >= 400:
            return {"ok": False, "slot_id": sid, "error": f"step: {step}", "attempt": attempt}
        t0 = time.time()
        while True:
            if time.time() - t0 > timeout:
                return {"ok": False, "slot_id": sid, "error": "timeout", "attempt": attempt}
            time.sleep(poll)
            st = (get(base, f"/api/step_status?session_id={sid}").get("status") or {})
            if st.get("error"):
                return {"ok": False, "slot_id": sid, "error": st["error"], "attempt": attempt}
            if st.get("done"):
                return {"ok": True, "slot_id": sid, "attempt": attempt}

    last_err = None
    for attempt in range(max_retries + 1):
        res = try_once(attempt)
        if res.get("ok"):
            return res
        last_err = res.get("error")
        time.sleep(2.0 + attempt * 1.5)
    return {"ok": False, "error": last_err}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="http://localhost:8010")
    p.add_argument("--folder", default="Negotiation Math vs Real LLM v1")
    p.add_argument("--llms", default=",".join(LLMS),
                   help="Comma-separated opponent aliases (replace default LLMs with e.g. mimic aliases)")
    p.add_argument("--episodes-per-pair", type=int, default=2,
                   help="Episodes per (math_tier, LLM, role) combo (default 2 → 80 total)")
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument("--message-limit", type=int, default=10)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--start-seed", type=int, default=10000)
    args = p.parse_args()

    folder_id = find_or_create_folder(args.base, args.folder)
    print(f"Folder = {folder_id}", flush=True)

    opponents = [s.strip() for s in args.llms.split(",") if s.strip()]
    print(f"Opponents: {opponents}", flush=True)
    # Canonical eval-pool seed allocation: every cell uses the SAME first N
    # seeds (seed = start_seed + ep_idx, no cell offset). Each ep_idx maps to
    # one canonical scenario shared across all (tier, llm, role) cells AND
    # across runs that draw from the same pool (math-vs-mimic, math-vs-LLM,
    # mimic-vs-mimic, LLM-vs-LLM holdout). Enables paired analysis on every
    # comparison dimension.
    jobs = []
    for tier in MATH_TIERS:
        for llm in opponents:
            for role in ("buyer", "seller"):
                for ep_idx in range(args.episodes_per_pair):
                    seed = args.start_seed + ep_idx
                    buyer = tier if role == "buyer" else llm
                    seller = tier if role == "seller" else llm
                    name = f"{tier[:8]}-{role[:1]}-{llm[:4]}-{seed}"
                    jobs.append({"buyer": buyer, "seller": seller, "seed": seed,
                                 "tier": tier, "llm": llm, "tier_role": role,
                                 "ep_idx": ep_idx, "name": name})
    print(f"Total: {len(jobs)} episodes ({len(MATH_TIERS)} tiers × {len(opponents)} opps × 2 roles × {args.episodes_per_pair} eps)", flush=True)

    completed = {"n": 0, "fail": 0}
    lock = threading.Lock()
    results = []

    def task(j):
        t0 = time.time()
        res = run_one(args.base, folder_id=folder_id, seed=j["seed"], slot_name=j["name"],
                      buyer_alias=j["buyer"], seller_alias=j["seller"],
                      message_limit=args.message_limit, max_retries=args.max_retries)
        dur = time.time() - t0
        res.update({"job": j, "duration_s": round(dur, 1)})
        with lock:
            completed["n"] += 1
            if not res.get("ok"):
                completed["fail"] += 1
            tag = "ok" if res.get("ok") else "FAIL"
            print(f"  [{completed['n']}/{len(jobs)}] {tag} {j['tier'][:14]:<14} {j['tier_role']:<6} "
                  f"vs {j['llm']:<8} seed={j['seed']} {dur:.0f}s "
                  f"{res.get('slot_id','')} {('err='+str(res.get('error',''))[:80]) if not res.get('ok') else ''}",
                  flush=True)
        return res

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(task, j) for j in jobs]
        for f in as_completed(futures):
            results.append(f.result())
    print(f"\nDone in {time.time()-t0:.0f}s. {completed['n']-completed['fail']}/{len(jobs)} successful", flush=True)
    return 0 if completed["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
