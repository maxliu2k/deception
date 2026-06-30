"""Surgical in-place refresh of the Trivial-Open-vs-LLM negotiation slots.

Only the Math-Trivial-Open policy changed, so we re-run ONLY the slots in the
"Negotiation v9 Math-vs-LLM" folder whose pairing involves Math-Trivial-Open,
overwriting each existing slot in place (same slot_id, same name, same folder).
Every other slot (Truth/Reactive/Deadline/RL vs LLM) is left untouched — no
wasted API spend on duels whose policy never changed.

Target pairings are read from each slot's STORED config (buyer/seller/seed), not
parsed from the slot name (names are inconsistent across generation runs).

OpenRouter credit usage is sampled before/after so the run reports exact cost.

    python -m simulation.update_negotiation_t1_vs_llm --probe 5     # measure first
    python -m simulation.update_negotiation_t1_vs_llm --all         # full 250
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
import threading
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from simulation.run_deception_competition import post, get

BASE_DEFAULT = "http://localhost:8000"
SIM = Path(__file__).parent
RUNTIME = SIM / ".runtime" / "save_slots"
KEY_PATH = SIM.parent / "keys" / "openkey.txt"
FOLDER_NAME = "Negotiation v9 Math-vs-LLM"
T1_ALIAS = "Math-Trivial-Open"


def _credits() -> float | None:
    """OpenRouter cumulative usage in $ (total_usage). None on failure."""
    try:
        key = KEY_PATH.read_text(encoding="utf-8").strip()
        req = urllib.request.Request("https://openrouter.ai/api/v1/credits",
                                     headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=20) as r:
            import json
            return float(json.loads(r.read())["data"]["total_usage"])
    except Exception as e:
        print(f"  (credit read failed: {e})", flush=True)
        return None


def _folder_id(base: str) -> str:
    import json
    with urllib.request.urlopen(f"{base}/api/folders", timeout=15) as r:
        folders = json.loads(r.read())["folders"]
    for f in folders:
        if f["name"] == FOLDER_NAME:
            return f["folder_id"]
    raise SystemExit(f"Folder {FOLDER_NAME!r} not found")


def _slots_in_folder(base: str, fid: str) -> list[dict]:
    import json
    with urllib.request.urlopen(f"{base}/api/save_slots", timeout=120) as r:
        return [s for s in json.loads(r.read())["slots"] if s.get("folder_id") == fid]


def _slot_config(sid: str) -> dict | None:
    """Read (buyer, seller, seed) from a slot's stored runtime."""
    pkl = RUNTIME / sid / "runtime.pkl"
    if not pkl.exists():
        return None
    try:
        with pkl.open("rb") as f:
            runtime = pickle.load(f)
        env = runtime.get("env")
        cfg = env.config if env is not None else {}
        sel = list(cfg.get("selected_models") or [])
        if len(sel) < 2:
            return None
        return {"buyer": sel[0], "seller": sel[1], "seed": int(cfg.get("seed"))}
    except Exception:
        return None


def _rerun_in_place(base: str, sid: str, buyer: str, seller: str, seed: int,
                    message_limit: int = 10, poll: float = 2.0,
                    timeout: float = 600.0, max_retries: int = 2) -> dict:
    last_err = None
    for attempt in range(max_retries + 1):
        reset = post(base, "/api/reset", {
            "selected_models": [buyer, seller, "GPT-5.4"],
            "mode": "buyer_seller_negotiation",
            "seed": int(seed),
            "negotiation_message_limit": int(message_limit),
            "save_slot": sid,
        })
        if reset.get("ok") is not True and reset.get("_status", 200) >= 400:
            last_err = f"reset: {reset}"; time.sleep(2.0 + attempt); continue
        step = post(base, "/api/step", {"use_models": True, "save_slot": sid})
        if step.get("ok") is not True and step.get("_status", 200) >= 400:
            last_err = f"step: {step}"; time.sleep(2.0 + attempt); continue
        t0 = time.time()
        while time.time() - t0 <= timeout:
            time.sleep(poll)
            st = (get(base, f"/api/step_status?session_id={sid}").get("status") or {})
            if st.get("error"):
                last_err = st["error"]; break
            if st.get("done"):
                return {"ok": True, "slot_id": sid}
        else:
            last_err = "timeout"
        time.sleep(2.0 + attempt)
    return {"ok": False, "slot_id": sid, "error": last_err}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default=BASE_DEFAULT)
    p.add_argument("--probe", type=int, default=0,
                   help="Run only N targets (spread across LLMs) to measure cost.")
    p.add_argument("--all", action="store_true", help="Run all Trivial-Open targets.")
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument("--message-limit", type=int, default=10)
    args = p.parse_args()
    if not args.probe and not args.all:
        raise SystemExit("Pass --probe N or --all")

    fid = _folder_id(args.base)
    slots = _slots_in_folder(args.base, fid)
    print(f"Folder {FOLDER_NAME} = {fid}  ({len(slots)} slots)", flush=True)

    # Build the target list from stored configs (robust to slot naming).
    targets = []
    for s in slots:
        cfg = _slot_config(s["slot_id"])
        if not cfg:
            continue
        if T1_ALIAS not in (cfg["buyer"], cfg["seller"]):
            continue
        llm = cfg["seller"] if cfg["buyer"] == T1_ALIAS else cfg["buyer"]
        targets.append({"sid": s["slot_id"], "name": s["name"],
                        "buyer": cfg["buyer"], "seller": cfg["seller"],
                        "seed": cfg["seed"], "llm": llm})
    print(f"Trivial-Open targets: {len(targets)}", flush=True)

    if args.probe:
        # One target per distinct LLM first (covers all model prices), then fill.
        by_llm = defaultdict(list)
        for t in targets:
            by_llm[t["llm"]].append(t)
        probe = []
        for llm in sorted(by_llm):
            probe.append(by_llm[llm][0])
        for t in targets:
            if len(probe) >= args.probe:
                break
            if t not in probe:
                probe.append(t)
        run_list = probe[:args.probe]
    else:
        run_list = targets
    print(f"Running {len(run_list)} episodes "
          f"(LLMs: {sorted({t['llm'] for t in run_list})})", flush=True)

    usage_before = _credits()
    completed = {"n": 0, "fail": 0}
    lock = threading.Lock()

    def task(t):
        t0 = time.time()
        res = _rerun_in_place(args.base, t["sid"], t["buyer"], t["seller"], t["seed"],
                              message_limit=args.message_limit)
        dur = time.time() - t0
        with lock:
            completed["n"] += 1
            if not res.get("ok"):
                completed["fail"] += 1
            tag = "ok" if res.get("ok") else "FAIL"
            print(f"  [{completed['n']}/{len(run_list)}] {tag} {t['name'][:26]:<26} "
                  f"vs {t['llm']:<8} seed={t['seed']} {dur:.0f}s "
                  f"{('err='+str(res.get('error',''))[:70]) if not res.get('ok') else ''}",
                  flush=True)
        return res

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(task, t) for t in run_list]
        for f in as_completed(futures):
            f.result()
    elapsed = time.time() - t0
    usage_after = _credits()

    n_ok = completed["n"] - completed["fail"]
    print(f"\nDone in {elapsed:.0f}s. {n_ok}/{len(run_list)} successful.", flush=True)
    if usage_before is not None and usage_after is not None:
        spent = usage_after - usage_before
        per_ep = spent / max(1, n_ok)
        print(f"OpenRouter spend: ${spent:.4f} over {n_ok} episodes "
              f"= ${per_ep:.4f}/episode", flush=True)
        print(f"  Projected for all 250 Trivial-Open slots: ${per_ep*250:.2f}", flush=True)
        # Remaining-budget readout.
        try:
            import json as _json
            key = KEY_PATH.read_text(encoding="utf-8").strip()
            req = urllib.request.Request("https://openrouter.ai/api/v1/credits",
                                         headers={"Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=20) as r:
                d = _json.loads(r.read())["data"]
            print(f"  OpenRouter remaining: ${d['total_credits']-d['total_usage']:.2f}", flush=True)
        except Exception:
            pass
    return 0 if completed["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
