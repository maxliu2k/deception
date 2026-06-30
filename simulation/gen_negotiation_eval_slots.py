"""Fast in-process generator for negotiation eval save-slots on canonical seeds.

Runs mimic-vs-mimic and math-tier-vs-mimic duels IN PROCESS (no server
subprocess), driving env.step so each slot carries a real EpisodeResult, then
persists the slot files + catalog entries directly. ~100x faster than routing
every episode through the server's per-episode step_worker.

Run with the server STOPPED (this writes the catalog directly).

    python -m simulation.gen_negotiation_eval_slots
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

from simulation.eval_negotiation_pairings import run_one_env, MIMICS, MATH_TIERS, MATH_RL_ALIAS

SIM = Path(__file__).parent
RUNTIME = SIM / ".runtime" / "save_slots"
CATALOG = SIM / ".runtime" / "save_slots.json"
START_SEED = 2000
N_SEEDS = 25            # canonical eval scenarios 2000..2024
MESSAGE_LIMIT = 10


def _load_catalog() -> dict:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def _save_catalog(cat: dict) -> None:
    CATALOG.write_text(json.dumps(cat, indent=2))


def _reset_folder(cat: dict, name: str) -> str:
    """Drop any existing folder of this name (and its slot dirs), create fresh."""
    old = [f["folder_id"] for f in cat["folders"] if f["name"] == name]
    if old:
        oset = set(old)
        for s in cat["slots"]:
            if s.get("folder_id") in oset:
                d = RUNTIME / s["slot_id"]
                if d.exists():
                    import shutil
                    shutil.rmtree(d, ignore_errors=True)
        cat["slots"] = [s for s in cat["slots"] if s.get("folder_id") not in oset]
        cat["folders"] = [f for f in cat["folders"] if f["folder_id"] not in oset]
    fid = f"folder_{cat.get('next_folder_index', 1)}"
    cat["next_folder_index"] = cat.get("next_folder_index", 1) + 1
    cat["folders"].append({"folder_id": fid, "name": name,
                           "created_at": time.time(), "parent_folder_id": None})
    return fid


def _persist(cat: dict, folder_id: str, name: str, env) -> None:
    idx = cat.get("next_index", 1)
    cat["next_index"] = idx + 1
    sid = f"save_slot_{idx}"
    d = RUNTIME / sid
    d.mkdir(parents=True, exist_ok=True)
    runtime = {
        "env": env,
        "last_reset": None,
        "last_result": env.result,
        "conversation_log": [],
        "step_status": {"running": False, "done": True, "error": None,
                        "used_models": True, "llm_error": None},
        "last_batch_export_text": None,
        "last_mega_batch_export_text": None,
        "batch_status": {},
        "mega_batch_status": {},
        "updated_at": time.time(),
    }
    with (d / "runtime.pkl").open("wb") as f:
        pickle.dump(runtime, f)
    # Full lightweight meta so /api/save_slots can render this slot without
    # unpickling the runtime (mode + done are the fast-path keys server-side).
    (d / "runtime_meta.json").write_text(json.dumps({
        "updated_at": runtime["updated_at"],
        "mode": "buyer_seller_negotiation",
        "phase": "negotiation",
        "done": True,
    }))
    cat["slots"].append({"slot_id": sid, "name": name,
                         "created_at": time.time(), "folder_id": folder_id})


def main() -> int:
    import sys
    # Optional section filter so a policy tweak can refresh just the affected
    # folder without redoing identical work: `--only math-vs-mimic`.
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    do_mm = only in (None, "mimic-vs-mimic")
    do_mt = only in (None, "math-vs-mimic")

    cat = _load_catalog()
    tiers = list(MATH_TIERS) + [MATH_RL_ALIAS]
    seeds = list(range(START_SEED, START_SEED + N_SEEDS))

    n = 0
    # ---- 1) Mimic-vs-mimic: 25 seeds x 20 cross-pairings (no self) ----
    if do_mm:
        mm_fid = _reset_folder(cat, "Negotiation v9 Mimic-vs-Mimic")
        t0 = time.time()
        for seed in seeds:
            for b in MIMICS:           # MIMICS already carry the "Mimic-" prefix
                for s in MIMICS:
                    if b == s:
                        continue
                    env = run_one_env(b, s, seed=seed, message_limit=MESSAGE_LIMIT)
                    bn, sn = b.replace("Mimic-", "")[:4], s.replace("Mimic-", "")[:4]
                    _persist(cat, mm_fid, f"mim-{bn}-{sn}-s{seed}", env)
                    n += 1
            if (seed - START_SEED + 1) % 5 == 0:
                print(f"  mimic-vs-mimic: seed {seed} done ({n} eps, {time.time()-t0:.0f}s)", flush=True)
        print(f"Mimic-vs-mimic: {n} episodes ({time.time()-t0:.0f}s)", flush=True)

    # ---- 2) Math tier vs mimic: tiers x mimics x both roles x 25 seeds ----
    n2 = 0
    if not do_mt:
        _save_catalog(cat)
        print(f"\nDone (mimic-vs-mimic only). Persisted {n} slots. "
              f"Catalog now {len(cat['slots'])} slots.", flush=True)
        return 0
    mt_fid = _reset_folder(cat, "Negotiation v9 Math-vs-Mimic")
    t1 = time.time()
    for tier in tiers:
        for mimic in MIMICS:       # already "Mimic-X"
            for role in ("buyer", "seller"):
                b = tier if role == "buyer" else mimic
                s = tier if role == "seller" else mimic
                mn = mimic.replace("Mimic-", "")[:4]
                for seed in seeds:
                    env = run_one_env(b, s, seed=seed, message_limit=MESSAGE_LIMIT)
                    _persist(cat, mt_fid, f"{tier[:8]}-{role[:1]}-{mn}-s{seed}", env)
                    n2 += 1
        print(f"  math-vs-mimic: {tier} done ({n2} eps, {time.time()-t1:.0f}s)", flush=True)
    print(f"Math-vs-mimic: {n2} episodes ({time.time()-t1:.0f}s)", flush=True)

    _save_catalog(cat)
    print(f"\nDone. Persisted {n + n2} slots. Catalog now {len(cat['slots'])} slots.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
