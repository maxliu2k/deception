"""Prove the fidelity numbers come from the raw pkls, not from echoing the table.
(1) dump the per-auction win-count matrix (NOT in any table) for the 10 real
    auctions, with row sums (=12) and column sums (=the totals).
(2) show the Delta bootstrap CI shifting with the RNG seed (proof it's resampled,
    not copied)."""
import sys, json, pickle
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, ".")
sys.path.insert(0, "simulation")
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


def matrix(fid):
    rows, ids = [], []
    for sid in byfolder.get(fid, []):
        try:
            obj = load(sid)
        except Exception:
            continue
        models = obj.get("last_reset", {}).get("selected_models")
        cp = obj.get("last_result", {}).get("derived", {}).get("completed_paintings", [])
        if not models or len(cp) != 12:
            continue
        seat = {f"bidder_{i+1}": norm(models[i]) for i in range(len(models))}
        cnt = {m: 0 for m in ORDER}
        for p in cp:
            w = seat.get(p["winner_id"], "")
            if w in cnt:
                cnt[w] += 1
        rows.append([cnt[m] for m in ORDER]); ids.append(sid)
    return np.array(rows), ids


R, rids = matrix("folder_19")
print("RAW per-auction win counts, 10 real auctions (rows sum to 12):")
print(f"{'slot':>16} " + " ".join(f"{m:>6}" for m in ORDER) + f"  {'sum':>4}")
for sid, r in zip(rids, R):
    print(f"{sid:>16} " + " ".join(f"{v:>6}" for v in r) + f"  {r.sum():>4}")
print(f"{'COLUMN TOTALS':>16} " + " ".join(f"{v:>6}" for v in R.sum(0)) + f"  {R.sum():>4}")
print(f"{'win share %':>16} " + " ".join(f"{v*100/R.sum():>6.1f}" for v in R.sum(0)))

M, _ = matrix("folder_20")
print(f"\n30 mimic auctions: column totals = {M.sum(0).tolist()} (sum {M.sum()})")
print(f"mimic win share % = {[round(v*100/M.sum(),1) for v in M.sum(0)]}")

# Delta bootstrap for Pro under 4 different seeds -> should jiggle
print("\nPro Delta(mimic-real) bootstrap CI under different RNG seeds (proof it's resampled):")
r_pro = R[:, ORDER.index("Pro")] / 12.0
m_pro = M[:, ORDER.index("Pro")] / 12.0
for seed in [1, 42, 12345, 99999]:
    g = np.random.default_rng(seed)
    d = np.array([g.choice(m_pro, len(m_pro), True).mean() - g.choice(r_pro, len(r_pro), True).mean()
                  for _ in range(10000)])
    print(f"   seed {seed:>6}:  Delta=-2.8pp  95% CI = [{np.percentile(d,2.5)*100:+.2f}, {np.percentile(d,97.5)*100:+.2f}]")
