"""Auction-level bootstrap CIs for the baseline win rates (Table III).

Point estimate = mean of per-auction focal win fractions (== pooled proportion,
since every auction has exactly 12 paintings). CI = percentile bootstrap over
auctions (the independent replicate), 10,000 resamples. We also print the naive
painting-level binomial CI for comparison, to show which way the correction goes.
"""
import sys, json, pickle
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, ".")
sys.path.insert(0, "simulation")
import numpy as np

HERE = Path("simulation")
cat = json.load(open(HERE / ".runtime" / "save_slots.json", encoding="utf-8"))
folders = {f["folder_id"]: f.get("name", "") for f in cat.get("folders", [])}
slots = cat.get("slots", {})
if isinstance(slots, list):
    slots = {s["slot_id"]: s for s in slots}
byfolder = defaultdict(list)
for sid, m in slots.items():
    byfolder[m.get("folder_id")].append(sid)


def load(sid):
    return pickle.load(open(HERE / ".runtime" / "save_slots" / sid / "runtime.pkl", "rb"))


# tier -> (policy name, mimic folder, real folder)
TIERS = [
    ("T1", "Trivial",         "folder_21", "folder_31"),
    ("T2", "Fair-Share",      "folder_22", "folder_32"),
    ("T3", "Reactive",        "folder_23", "folder_39"),
    ("T4", "Market-Clearing", "folder_24", "folder_33"),
    ("T5", "RL",              "folder_25", "folder_30"),
]

rng = np.random.default_rng(12345)


def focal_fractions(fid, verbose_first=False):
    """Return list of per-auction focal win fractions (k/12) for a folder."""
    fracs = []
    skipped = 0
    first = True
    for sid in byfolder.get(fid, []):
        try:
            obj = load(sid)
        except Exception:
            skipped += 1
            continue
        reset = obj.get("last_reset", {})
        models = reset.get("selected_models")
        der = obj.get("last_result", {}).get("derived", {})
        cp = der.get("completed_paintings", [])
        if not models or len(cp) != 12:
            skipped += 1
            continue
        seat = {f"bidder_{i+1}": models[i] for i in range(len(models))}
        n_focal_seats = sum(1 for m in models if str(m).startswith("Math-"))
        k = sum(1 for p in cp if str(seat.get(p["winner_id"], "")).startswith("Math-"))
        fracs.append(k / 12.0)
        if verbose_first and first:
            print(f"      first slot {sid}: models={models}  focal_seats={n_focal_seats}  k={k}")
            first = False
    return fracs, skipped


def boot_ci(fracs, B=10000):
    a = np.array(fracs, dtype=float)
    n = len(a)
    means = np.array([rng.choice(a, size=n, replace=True).mean() for _ in range(B)])
    return a.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5), a.std(ddof=1) if n > 1 else 0.0


def naive_binom_ci(fracs):
    """Naive painting-level Wald CI treating all 12*n paintings as iid Bernoulli."""
    a = np.array(fracs, dtype=float)
    n_auc = len(a)
    N = 12 * n_auc
    p = a.mean()
    se = np.sqrt(p * (1 - p) / N) if N > 0 else 0.0
    return p, p - 1.96 * se, p + 1.96 * se


for cond_name, key in [("VS MIMICS (n=50)", 2), ("VS REAL LLMs (n=5)", 3)]:
    print("=" * 72)
    print(cond_name)
    print("=" * 72)
    for tier, pol, mimic_fid, real_fid in TIERS:
        fid = (mimic_fid, real_fid)[key - 2]
        fracs, skipped = focal_fractions(fid, verbose_first=True)
        if not fracs:
            print(f"  {tier} {pol:<16} NO DATA (skipped {skipped})")
            continue
        mean, lo, hi, sd = boot_ci(fracs)
        pn, nlo, nhi = naive_binom_ci(fracs)
        print(f"  {tier} {pol:<16} n={len(fracs):>2} (skip {skipped})")
        print(f"      per-auction fracs (x12 = wins): "
              f"{[round(f*12) for f in fracs]}")
        print(f"      mean win rate = {mean*100:5.1f}%   SD(frac)={sd*100:4.1f}pp")
        print(f"      auction-level bootstrap 95% CI = [{lo*100:5.1f}, {hi*100:5.1f}]  "
              f"(width {100*(hi-lo):.1f}pp)")
        print(f"      naive painting-level 95% CI    = [{nlo*100:5.1f}, {nhi*100:5.1f}]  "
              f"(width {100*(nhi-nlo):.1f}pp)")
    print()
