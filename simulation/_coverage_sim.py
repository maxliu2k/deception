"""Which CI is 'more accurate' at n=5? Measure COVERAGE.
Treat each policy's 50 mimic-runs as the population (truth = its mean), draw many
n=5 samples, and check how often each method's 95% CI contains the truth.
Target = 95%. Below 95% = overconfident (too narrow); above = conservative."""
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
    return np.array(out, float)


T4 = 2.776  # t_{.975, df=4}  (n=5)
Z = 1.95996
g = np.random.default_rng(2024)


def coverage(pop, ITER=4000, B=800, n=5):
    mu = pop.mean()
    hit = {"z": 0, "t": 0, "logit-t": 0, "boot": 0}
    for _ in range(ITER):
        s = g.choice(pop, n, replace=True)
        m, sd = s.mean(), s.std(ddof=1)
        se = sd / np.sqrt(n)
        # z and t
        if m - Z * se <= mu <= m + Z * se: hit["z"] += 1
        if m - T4 * se <= mu <= m + T4 * se: hit["t"] += 1
        # logit-t (delta method, keeps interval in (0,1))
        if 0 < m < 1 and se > 0:
            lg = np.log(m / (1 - m)); se_lg = se / (m * (1 - m))
            lo = 1 / (1 + np.exp(-(lg - T4 * se_lg))); hi = 1 / (1 + np.exp(-(lg + T4 * se_lg)))
        else:
            lo = hi = m
        if lo <= mu <= hi: hit["logit-t"] += 1
        # percentile bootstrap
        bm = g.choice(s, (B, n), replace=True).mean(1)
        if np.percentile(bm, 2.5) <= mu <= np.percentile(bm, 97.5): hit["boot"] += 1
    return {k: 100 * v / ITER for k, v in hit.items()}, mu


POLS = [("Trivial","folder_21"),("Fair-Share","folder_22"),("Reactive","folder_23"),
        ("Market-Clearing","folder_24"),("RL","folder_25")]
print("Coverage of nominal 95% CIs, n=5 samples drawn from the n=50 'truth':")
print(f"{'policy':<16} {'truth%':>6} {'z':>6} {'t':>6} {'logit-t':>8} {'boot':>6}")
agg = defaultdict(list)
for pol, fid in POLS:
    pop = focal_fracs(fid)
    cov, mu = coverage(pop)
    print(f"{pol:<16} {mu*100:6.1f} {cov['z']:6.1f} {cov['t']:6.1f} {cov['logit-t']:8.1f} {cov['boot']:6.1f}")
    for k, v in cov.items(): agg[k].append(v)
print(f"{'MEAN':<16} {'':>6} " + " ".join(f"{np.mean(agg[k]):6.1f}" for k in ['z','t','logit-t','boot']))
print("\n(target = 95.0; below = overconfident/too-narrow)")
