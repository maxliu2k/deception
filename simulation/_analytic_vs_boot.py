"""Analytic (deterministic) CI vs bootstrap, side by side.
Analytic = mean +/- t_{.975,df} * s/sqrt(n)  (and the z version for contrast)."""
import sys, json, pickle
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, "."); sys.path.insert(0, "simulation")
import numpy as np

HERE = Path("simulation")
cat = json.load(open(HERE / ".runtime" / "save_slots.json", encoding="utf-8"))
slots = cat.get("slots", {})
if isinstance(slots, list):
    slots = {s["slot_id"]: s for s in slots}
byfolder = defaultdict(list)
for sid, m in slots.items():
    byfolder[m.get("folder_id")].append(sid)
norm = lambda m: str(m).replace("Mimic-", "").replace("GPT-5.4", "GPT")
load = lambda sid: pickle.load(open(HERE / ".runtime" / "save_slots" / sid / "runtime.pkl", "rb"))
T975 = {4: 2.776, 9: 2.262, 29: 2.045, 49: 2.010}  # t for df


def focal_fracs(fid):
    out = []
    for sid in byfolder.get(fid, []):
        try: obj = load(sid)
        except Exception: continue
        models = obj.get("last_reset", {}).get("selected_models")
        cp = obj.get("last_result", {}).get("derived", {}).get("completed_paintings", [])
        if not models or len(cp) != 12: continue
        seat = {f"bidder_{i+1}": norm(models[i]) for i in range(len(models))}
        k = sum(1 for p in cp if str(seat.get(p["winner_id"], "")).startswith("Math-"))
        out.append(k / 12.0)
    return out


def show(label, fracs):
    a = np.array(fracs, float); n = len(a); m = a.mean(); s = a.std(ddof=1); se = s / np.sqrt(n)
    df = n - 1; t = T975.get(df, 2.0)
    z_lo, z_hi = m - 1.96 * se, m + 1.96 * se
    t_lo, t_hi = m - t * se, m + t * se
    g = np.random.default_rng(7)
    bm = np.array([g.choice(a, n, True).mean() for _ in range(10000)])
    b_lo, b_hi = np.percentile(bm, 2.5), np.percentile(bm, 97.5)
    print(f"{label:<26} n={n:<3} mean={m*100:5.1f}  s={s*100:4.1f}  SE={se*100:4.2f}")
    print(f"      z-CI(1.96) = [{z_lo*100:5.1f},{z_hi*100:5.1f}]   "
          f"t-CI({t:.2f})  = [{t_lo*100:5.1f},{t_hi*100:5.1f}]   "
          f"boot = [{b_lo*100:5.1f},{b_hi*100:5.1f}]")


print("BASELINES vs mimics (n=50):")
for tier, pol, fid in [("T1","Trivial","folder_21"),("T2","Fair-Share","folder_22"),
                       ("T3","Reactive","folder_23"),("T4","Market-Clearing","folder_24"),
                       ("T5","RL","folder_25")]:
    show(f"  {pol}", focal_fracs(fid))

print("\nBASELINES vs real LLMs (n=5):")
for tier, pol, fid in [("T1","Trivial","folder_31"),("T2","Fair-Share","folder_32"),
                       ("T3","Reactive","folder_39"),("T4","Market-Clearing","folder_33"),
                       ("T5","RL","folder_30")]:
    show(f"  {pol}", focal_fracs(fid))
