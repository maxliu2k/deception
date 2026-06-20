"""v5 deception checks:
 (A) real-LLM win shares (folder_65, n=50) + aggression profile
 (B) math-tier ladder vs mimics (folders 85-89, T1-T5, n=150) -> depth test
 (C) transfer: math-tiers vs real LLMs (folders 92-96, n=10)
 (D) mimic fidelity: real (65) vs mimic (84) per-model win shares
All win rates = mean of per-episode (rounds-won/12); CIs = t intervals.
"""
import json
from collections import defaultdict
from pathlib import Path
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
norm = lambda m: str(m).replace("Mimic-", "").replace("GPT-5.4", "GPT")
T = {9: 2.262, 49: 2.010, 149: 1.976}


def logp(sid):
    return HERE / ".runtime" / "save_slots" / sid / "auction_exports" / "deception_episode" / "episode_log.json"


def load_folder(fid):
    eps = []
    for sid in byfolder.get(fid, []):
        p = logp(sid)
        if not p.exists():
            continue
        d = json.load(open(p))
        if d.get("complete"):
            eps.append(d)
    return eps


def ci(fracs):
    a = np.array(fracs, float); n = len(a)
    if n < 2:
        return a.mean(), a.mean(), a.mean()
    se = a.std(ddof=1) / np.sqrt(n); t = T.get(n - 1, 1.98)
    return a.mean(), a.mean() - t * se, a.mean() + t * se


def model_fracs(eps):
    out = defaultdict(list)
    for d in eps:
        for ag in d["agents"]:
            out[norm(ag["alias"])].append(ag["total_reward"] / 12.0)
    return out


def focal_fracs(eps, prefix="Math-"):
    fr = []
    for d in eps:
        for ag in d["agents"]:
            if str(ag["alias"]).startswith(prefix):
                fr.append(ag["total_reward"] / 12.0)
    return fr


def aggression(eps):
    claim = defaultdict(list); disc = defaultdict(list); lie_n = defaultdict(float); lie_d = defaultdict(float)
    for d in eps:
        idx2 = {f"agent_{i+1}": norm(d["selected_models"][i]) for i in range(len(d["selected_models"]))}
        for r in d["rounds"]:
            truth = np.array(r["truth"])
            for aid, c in r["claims_by_agent"].items():
                al = idx2[aid]; c = np.array(c)
                claim[al].append(c.mean()); disc[al].append(np.maximum(0, c - truth).mean())
                lie_n[al] += np.sum(c > truth); lie_d[al] += len(c)
    return claim, disc, lie_n, lie_d


# (A) real win shares
real = load_folder("folder_65")
print(f"(A) REAL v5 win shares (n={len(real)} episodes)")
rf = model_fracs(real); cl, ds, ln, ld = aggression(real)
order = sorted(rf, key=lambda a: -np.mean(rf[a]))
print(f"{'model':<8}{'win%':>7}{'95% CI':>16}{'mean claim':>11}{'mean disc':>10}{'lie%':>7}")
for a in order:
    m, lo, hi = ci(rf[a])
    print(f"{a:<8}{m*100:>6.1f}%  [{lo*100:4.1f},{hi*100:4.1f}]{np.mean(cl[a]):>11.3f}{np.mean(ds[a]):>10.3f}{100*ln[a]/ld[a]:>6.0f}%")

# (B) math-tier vs mimics
print(f"\n(B) MATH-TIER ladder vs MIMIC field (n=150 each, fair share 20%)")
tiers = [("T1","folder_85"),("T2","folder_86"),("T3","folder_87"),("T4","folder_88"),("T5","folder_89")]
print(f"{'tier':<5}{'win%':>7}{'95% CI':>16}")
for t, fid in tiers:
    fr = focal_fracs(load_folder(fid))
    m, lo, hi = ci(fr)
    print(f"{t:<5}{m*100:>6.1f}%  [{lo*100:4.1f},{hi*100:4.1f}]   (n={len(fr)})")

# (C) transfer vs real
print(f"\n(C) TRANSFER: math-tiers vs REAL LLMs (n=10 each, fair share 20%)")
tiersR = [("T1","folder_92"),("T2","folder_96"),("T3","folder_93"),("T4","folder_94"),("T5","folder_95")]
print(f"{'tier':<5}{'win%':>7}{'95% CI':>16}")
for t, fid in tiersR:
    fr = focal_fracs(load_folder(fid))
    m, lo, hi = ci(fr)
    print(f"{t:<5}{m*100:>6.1f}%  [{lo*100:4.1f},{hi*100:4.1f}]   (n={len(fr)})")

# (D) fidelity: real vs mimic per-model win shares
print(f"\n(D) FIDELITY: real (65) vs mimic-field (84) per-model win share")
mim = load_folder("folder_84"); mf = model_fracs(mim)
print(f"{'model':<8}{'real%':>7}{'mimic%':>8}{'delta(pp)':>10}")
for a in order:
    rm = np.mean(rf[a]) * 100; mm = np.mean(mf.get(a, [0])) * 100
    print(f"{a:<8}{rm:>6.1f}%{mm:>7.1f}%{mm-rm:>+9.1f}")
print(f"   mimic episodes n={len(mim)}")
