"""Phase 4, step 2 — analyze the real-LLM transfer-check episodes.

Scans save_slots for transfer episodes (one Math-T* focal vs the 4 real LLMs
GPT-5.4/Opus/Pro/Grok), computes the focal's reward/win/trust per episode, and
prints it side-by-side with the vs-mimic-field result. If the ordering matches,
the mimic-field ladder TRANSFERS to the real models; if it reshuffles, that's the
honest finding (cf. the auction's RL-doesn't-transfer result).

Usage:  python -m simulation.analyze_transfer
"""
from __future__ import annotations
import glob
import json
from pathlib import Path

import numpy as np

LLM_FIELD = {"GPT-5.4", "Opus", "Pro", "Grok"}
ROOT = Path(__file__).parent / ".runtime" / "save_slots"
VSMIMIC = Path(__file__).parent / "datasets" / "deception_vsmimic_report_R.json"
TIERS = ["Math-T1", "Math-T2", "Math-T3", "Math-T4"]
LABEL = {"Math-T1": "T1 honest", "Math-T2": "T2 naive", "Math-T3": "T3 conditional", "Math-T4": "T4 backloaded"}


def boot(vals, seed=0):
    a = np.asarray(vals, float)
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(2000, len(a)))
    bs = a[idx].mean(axis=1)
    return float(a.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> int:
    by_focal: dict[str, list[dict]] = {}
    for f in glob.glob(str(ROOT / "**" / "deception_episode" / "episode_log.json"), recursive=True):
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        if not d.get("complete"):
            continue
        agents = d.get("agents", [])
        focals = [a for a in agents if str(a.get("alias", "")).startswith("Math-T")]
        if len(focals) != 1:
            continue
        others = {a["alias"] for a in agents} - {focals[0]["alias"]}
        if others != LLM_FIELD:                      # only transfer episodes (focal vs the 4 real LLMs)
            continue
        fa = focals[0]
        nr = int(d.get("num_rounds", 12)) or 12
        by_focal.setdefault(fa["alias"], []).append({
            "reward": float(fa.get("total_reward", 0.0)) / nr,
            "win": float(fa.get("win_count", 0)) / nr,
            "trust": float(fa.get("final_trust", 0.0)),
        })

    vm = json.loads(VSMIMIC.read_text(encoding="utf-8")).get("per_focal", {}) if VSMIMIC.exists() else {}
    print(f"Transfer check: math tiers vs the 4 REAL LLMs (GPT-5.4/Opus/Pro/Grok)\n")
    print(f"{'focal':16s} {'n':>2s} | {'vs-REAL reward/rnd (95% CI)':>30s} {'win':>6s} {'trust':>6s} | {'vs-MIMIC reward':>15s}")
    for t in TIERS:
        recs = by_focal.get(t, [])
        r = boot([x["reward"] for x in recs], seed=1)
        w = boot([x["win"] for x in recs], seed=2)
        tr = float(np.mean([x["trust"] for x in recs])) if recs else float("nan")
        vmr = vm.get(t, {}).get("reward_per_round", [None])[0]
        vmr_s = f"{vmr:.3f}" if isinstance(vmr, (int, float)) else "n/a"
        print(f"{LABEL.get(t, t):16s} {len(recs):2d} | "
              f"{r[0]:8.3f} [{r[1]:.3f}, {r[2]:.3f}]      {w[0]:6.3f} {tr:6.3f} | {vmr_s:>15s}")
    print("\nRead: if vs-REAL ordering matches vs-MIMIC (T2 worst, T3/T4 top), the ladder transfers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
