"""Deterministic Welch t CIs for the fidelity Delta (mimic-real), plus per-group
t CIs. n_real=10, n_mimic=30."""
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
ORDER = ["GPT", "Grok", "Llama", "Opus", "Pro"]
norm = lambda m: str(m).replace("Mimic-", "").replace("GPT-5.4", "GPT")
load = lambda sid: pickle.load(open(HERE / ".runtime" / "save_slots" / sid / "runtime.pkl", "rb"))


def per_model(fid):
    out = {m: [] for m in ORDER}
    for sid in byfolder.get(fid, []):
        try: obj = load(sid)
        except Exception: continue
        models = obj.get("last_reset", {}).get("selected_models")
        cp = obj.get("last_result", {}).get("derived", {}).get("completed_paintings", [])
        if not models or len(cp) != 12: continue
        seat = {f"bidder_{i+1}": norm(models[i]) for i in range(len(models))}
        cnt = {m: 0 for m in ORDER}
        for p in cp:
            w = seat.get(p["winner_id"], "")
            if w in cnt: cnt[w] += 1
        for m in ORDER: out[m].append(cnt[m] / 12.0)
    return {m: np.array(v, float) for m, v in out.items()}


def t_ppf(df):
    # rough t_{.975}; good enough for reporting
    table = {9: 2.262, 29: 2.045}
    if df in table: return table[df]
    # Welch df is fractional; interpolate from a small grid
    grid = [(5,2.571),(6,2.447),(8,2.306),(10,2.228),(12,2.179),(15,2.131),
            (20,2.086),(25,2.060),(30,2.042),(40,2.021),(60,2.000)]
    df = max(1, df)
    for (d,t) in grid:
        if df <= d: return t
    return 1.96


real = per_model("folder_19"); mimic = per_model("folder_20")
print("Fidelity Delta (mimic - real), deterministic Welch t 95% CI:")
print(f"{'Model':<7}{'Real%':>7}{'Mimic%':>8}{'D(pp)':>7}{'Welch 95% CI':>18}")
for m in ORDER:
    r, mm = real[m], mimic[m]
    nr, nm = len(r), len(mm)
    d = mm.mean() - r.mean()
    se = np.sqrt(r.var(ddof=1)/nr + mm.var(ddof=1)/nm)
    num = (r.var(ddof=1)/nr + mm.var(ddof=1)/nm)**2
    den = (r.var(ddof=1)/nr)**2/(nr-1) + (mm.var(ddof=1)/nm)**2/(nm-1)
    df = num/den if den > 0 else nr+nm-2
    t = t_ppf(df)
    print(f"{m:<7}{r.mean()*100:7.1f}{mm.mean()*100:8.1f}{d*100:+7.1f}"
          f"   [{(d-t*se)*100:+.1f}, {(d+t*se)*100:+.1f}]  (df~{df:.0f})")
