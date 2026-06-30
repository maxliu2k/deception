"""One-time backfill: enrich runtime_meta.json for every save-slot so the UI
endpoints can render from lightweight sidecars without unpickling each slot's
full torch runtime.

Two things are written into each slot's runtime_meta.json:
  * {updated_at, mode, phase, done}  — fast path for /api/save_slots.
  * "neg" summary {surplus, welfare, deal_capacity} for negotiation slots —
    fast path for /api/folder_distributions (which otherwise unpickles every
    negotiation runtime.pkl to compute surplus shares).

Slots whose sidecar already carries everything needed are skipped without an
unpickle. Run with the server STOPPED:

    python -m simulation.backfill_slot_meta
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

# Reuse the server's own summary derivation so the cached values can never drift
# from what the live endpoint would compute.
from simulation.server import _compute_slot_negotiation_summary

SIM = Path(__file__).parent
RUNTIME = SIM / ".runtime" / "save_slots"
CATALOG = SIM / ".runtime" / "save_slots.json"


def main() -> int:
    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    slots = cat.get("slots", [])
    total = len(slots)
    enriched = neg_added = skipped = missing = errors = 0
    t0 = time.time()

    for i, s in enumerate(slots):
        sid = s["slot_id"]
        d = RUNTIME / sid
        meta_path = d / "runtime_meta.json"
        pkl_path = d / "runtime.pkl"

        meta: dict = {}
        if meta_path.exists():
            try:
                loaded = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    meta = loaded
            except Exception:
                meta = {}

        mode = meta.get("mode")
        has_core = bool(mode) and meta.get("done") is True
        is_neg = mode == "buyer_seller_negotiation"
        needs_neg = is_neg and not isinstance(meta.get("neg"), dict)

        # Fully populated already (core fields present; neg present iff negotiation).
        if has_core and not needs_neg:
            skipped += 1
            continue

        if not pkl_path.exists():
            missing += 1
            continue

        try:
            with pkl_path.open("rb") as f:
                runtime = pickle.load(f)
            env = runtime.get("env")
            mode = getattr(env, "mode", None) or (env.config.get("mode") if env else None) or ""
            step_status = runtime.get("step_status") or {}
            done = bool(step_status.get("done", True))
            updated_at = runtime.get("updated_at") or meta.get("updated_at") or time.time()
            phase = meta.get("phase") or getattr(env, "phase", None) \
                or ("negotiation" if "negotiation" in mode else None)
            new_meta = {
                "updated_at": updated_at,
                "mode": mode,
                "phase": phase,
                "done": done,
            }
            # Preserve any already-cached neg, else compute for negotiation slots.
            if isinstance(meta.get("neg"), dict):
                new_meta["neg"] = meta["neg"]
            elif mode == "buyer_seller_negotiation":
                neg = _compute_slot_negotiation_summary(sid)
                if neg is not None:
                    new_meta["neg"] = neg
                    neg_added += 1
            meta_path.write_text(json.dumps(new_meta))
            enriched += 1
        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  ERROR {sid}: {e}", flush=True)

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{total}  enriched={enriched} neg={neg_added} "
                  f"skipped={skipped} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\nDone. total={total} enriched={enriched} neg_added={neg_added} "
          f"skipped={skipped} missing_pkl={missing} errors={errors} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
