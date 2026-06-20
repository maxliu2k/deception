"""Auction-level bootstrap CIs for the mimic-fidelity table (real vs mimic
per-model win shares). folder_19 = 10 real auctions, folder_20 = 30 mimic."""
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


def norm(m):
    return str(m).replace("Mimic-", "").replace("GPT-5.4", "GPT")


def load(sid):
    return pickle.load(open(HERE / ".runtime" / "save_slots" / sid / "runtime.pkl", "rb"))


def per_model_fracs(fid):
    """dict model -> list of per-auction win fractions (k/12)."""
    out = {m: [] for m in ORDER}
    n_auc = 0
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
        n_auc += 1
        cnt = {m: 0 for m in ORDER}
        for p in cp:
            w = seat.get(p["winner_id"], "")
            if w in cnt:
                cnt[w] += 1
        for m in ORDER:
            out[m].append(cnt[m] / 12.0)
    return out, n_auc


rng = np.random.default_rng(12345)


def boot_mean_ci(fracs, B=10000):
    a = np.array(fracs, float); n = len(a)
    means = np.array([rng.choice(a, n, replace=True).mean() for _ in range(B)])
    return a.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def naive_ci(fracs):
    a = np.array(fracs, float); N = 12 * len(a); p = a.mean()
    se = np.sqrt(p * (1 - p) / N)
    return p, p - 1.96 * se, p + 1.96 * se


def boot_delta_ci(real, mimic, B=10000):
    r = np.array(real, float); m = np.array(mimic, float)
    d = np.array([rng.choice(m, len(m), True).mean() - rng.choice(r, len(r), True).mean()
                  for _ in range(B)])
    return m.mean() - r.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


real, nr = per_model_fracs("folder_19")
mimic, nm = per_model_fracs("folder_20")
print(f"real auctions n={nr}   mimic auctions n={nm}\n")

print(f"{'Model':<7} {'Real%':>6} {'real auc-CI':>15} {'real naive-CI':>15}")
for m in ORDER:
    mean, lo, hi = boot_mean_ci(real[m]); _, nlo, nhi = naive_ci(real[m])
    print(f"{m:<7} {mean*100:6.1f} {f'[{lo*100:.1f},{hi*100:.1f}]':>15} {f'[{nlo*100:.1f},{nhi*100:.1f}]':>15}")

print()
print(f"{'Model':<7} {'Mimic%':>6} {'mimic auc-CI':>15} {'mimic naive-CI':>15}")
for m in ORDER:
    mean, lo, hi = boot_mean_ci(mimic[m]); _, nlo, nhi = naive_ci(mimic[m])
    print(f"{m:<7} {mean*100:6.1f} {f'[{lo*100:.1f},{hi*100:.1f}]':>15} {f'[{nlo*100:.1f},{nhi*100:.1f}]':>15}")

print()
print(f"{'Model':<7} {'Real%':>6} {'Mimic%':>7} {'Delta(pp)':>10} {'auction-level Delta 95% CI':>28}")
for m in ORDER:
    d, dlo, dhi = boot_delta_ci(real[m], mimic[m])
    print(f"{m:<7} {np.mean(real[m])*100:6.1f} {np.mean(mimic[m])*100:7.1f} "
          f"{d*100:+10.1f} {f'[{dlo*100:+.1f}, {dhi*100:+.1f}]':>28}")

# Cramer's V and chi-square on the pooled 2x5 contingency (paintings won)
real_tot = np.array([int(round(np.sum(real[m]) * 12)) for m in ORDER], float)
mimic_tot = np.array([int(round(np.sum(mimic[m]) * 12)) for m in ORDER], float)
obs = np.vstack([real_tot, mimic_tot])
row = obs.sum(1, keepdims=True); col = obs.sum(0, keepdims=True); N = obs.sum()
exp = row @ col / N
chi2 = np.sum((obs - exp) ** 2 / exp)
V = np.sqrt(chi2 / (N * (min(obs.shape) - 1)))
print(f"\nTotals real={real_tot.astype(int).tolist()} (sum {int(real_tot.sum())}), "
      f"mimic={mimic_tot.astype(int).tolist()} (sum {int(mimic_tot.sum())})")
print(f"chi2={chi2:.3f}  Cramer's V={V:.3f}  (N={int(N)} paintings)")
