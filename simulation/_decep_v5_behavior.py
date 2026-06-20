"""v5 behavior-characterization detail + distributional fidelity (TVD)."""
import json
from collections import defaultdict
from pathlib import Path
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
ORDER = ["Pro", "Opus", "Llama", "GPT", "Grok"]


def logp(sid):
    return HERE / ".runtime" / "save_slots" / sid / "auction_exports" / "deception_episode" / "episode_log.json"


def load_folder(fid):
    eps = []
    for sid in byfolder.get(fid, []):
        p = logp(sid)
        if p.exists():
            d = json.load(open(p))
            if d.get("complete"):
                eps.append(d)
    return eps


# verify trust rule on one episode
d0 = load_folder("folder_65")[0]
a0 = d0["agents"][0]
print("TRUST RULE CHECK (agent0):")
print("  sum_disc_when_won:", a0["sum_discrepancy_when_won"], "win_count:", a0["win_count"])
print("  1 - sum/win_count:", [round(1 - s / max(1, a0["win_count"]), 4) for s in a0["sum_discrepancy_when_won"]])
print("  final_trust      :", a0["final_trust"])

# behavior detail
real = load_folder("folder_65")
ftrust = defaultdict(list); c_hi = defaultdict(list); c_lo = defaultdict(list)
d_hi = defaultdict(list); d_lo = defaultdict(list)
for d in real:
    idx2 = {f"agent_{i+1}": norm(d["selected_models"][i]) for i in range(len(d["selected_models"]))}
    for ag in d["agents"]:
        ftrust[norm(ag["alias"])].append(np.mean(ag["final_trust"]))
    for r in d["rounds"]:
        t = np.array(r["truth"]); hi = t >= 0.7; lo = t <= 0.3
        for aid, c in r["claims_by_agent"].items():
            al = idx2[aid]; c = np.array(c); dd = np.maximum(0, c - t)
            if hi.any(): c_hi[al].append(c[hi].mean()); d_hi[al].append(dd[hi].mean())
            if lo.any(): c_lo[al].append(c[lo].mean()); d_lo[al].append(dd[lo].mean())

print("\nBEHAVIOR DETAIL (real v5):")
print(f"{'model':<7}{'final tr':>9}{'claim|hi-t':>11}{'claim|lo-t':>11}{'disc|hi-t':>10}{'disc|lo-t':>10}")
for a in ORDER:
    print(f"{a:<7}{np.mean(ftrust[a]):>9.3f}{np.mean(c_hi[a]):>11.3f}{np.mean(c_lo[a]):>11.3f}"
          f"{np.mean(d_hi[a]):>10.3f}{np.mean(d_lo[a]):>10.3f}")

# distributional fidelity: TVD of claim histograms, real vs mimic field
mim = load_folder("folder_84")
BINS = np.linspace(0, 1, 11)


def claims_by_model(eps):
    out = defaultdict(list)
    for d in eps:
        idx2 = {f"agent_{i+1}": norm(d["selected_models"][i]) for i in range(len(d["selected_models"]))}
        for r in d["rounds"]:
            for aid, c in r["claims_by_agent"].items():
                out[idx2[aid]].extend(c)
    return out


rc = claims_by_model(real); mc = claims_by_model(mim)
print("\nDISTRIBUTIONAL FIDELITY (claim-histogram TVD, real vs mimic field):")
all_r = []; all_m = []
for a in ORDER:
    pr, _ = np.histogram(rc[a], bins=BINS, density=False); pr = pr / pr.sum()
    pm, _ = np.histogram(mc[a], bins=BINS, density=False); pm = pm / pm.sum()
    tvd = 0.5 * np.abs(pr - pm).sum()
    print(f"  {a:<7} TVD={tvd:.3f}")
    all_r.extend(rc[a]); all_m.extend(mc[a])
pr, _ = np.histogram(all_r, bins=BINS); pr = pr / pr.sum()
pm, _ = np.histogram(all_m, bins=BINS); pm = pm / pm.sum()
print(f"  pooled  TVD={0.5*np.abs(pr-pm).sum():.3f}")
