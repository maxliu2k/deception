"""Deception v5 figures, matching the auction figure style (serif, deepened
palette, vector PDF + PNG).

  Fig 1 deception_claim_vs_truth : per-model mean claim binned by true value,
        with the honest y=x diagonal. Shows all models hug the diagonal at high
        truth and fan above it at low truth (Llama stays honest; others inflate).
  Fig 2 deception_ladder_transfer : grouped bars of the information ladder, vs
        the mimic field and vs real LLMs, with 95% CIs and the 20% fair-share line.
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path("simulation")
OUT = HERE / "exports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6})

STYLE = {  # same deepened palette as the auction figures
    "GPT":   ("#08519C", "o"),
    "Claude": ("#D26E00", "s"),
    "Gemini": ("#1A7A3D", "^"),
    "Grok":  ("#000000", "D"),
    "Llama": ("#A8327D", "v"),
}
ORDER = ["Gemini", "Claude", "Llama", "GPT", "Grok"]  # by win rate
norm = lambda m: str(m).replace("Mimic-", "").replace("GPT-5.4", "GPT").replace("Pro", "Gemini").replace("Opus", "Claude")

# ---- load real episodes (folder_65) for Fig 1 ----
cat = json.load(open(HERE / ".runtime" / "save_slots.json", encoding="utf-8"))
slots = cat.get("slots", {})
if isinstance(slots, list):
    slots = {s["slot_id"]: s for s in slots}
byf = defaultdict(list)
for sid, m in slots.items():
    byf[m.get("folder_id")].append(sid)


def lp(sid):
    return HERE / ".runtime" / "save_slots" / sid / "auction_exports" / "deception_episode" / "episode_log.json"


pairs = defaultdict(lambda: ([], []))  # model -> (truths, claims)
for sid in byf.get("folder_65", []):
    p = lp(sid)
    if not p.exists():
        continue
    d = json.load(open(p))
    if not d.get("complete"):
        continue
    idx2 = {f"agent_{i+1}": norm(d["selected_models"][i]) for i in range(len(d["selected_models"]))}
    for r in d["rounds"]:
        t = r["truth"]
        for aid, c in r["claims_by_agent"].items():
            al = idx2[aid]
            pairs[al][0].extend(t)
            pairs[al][1].extend(c)

# ===== Figure 1: claim vs truth =====
BINS = np.linspace(0, 1, 11)
ctr = 0.5 * (BINS[:-1] + BINS[1:])
fig, ax = plt.subplots(figsize=(3.4, 2.5))
ax.plot([0, 1], [0, 1], ls="--", lw=0.9, color="0.55", zorder=1, label="honest ($c=t$)")
for m in ORDER:
    t = np.array(pairs[m][0]); c = np.array(pairs[m][1])
    means = [c[(t >= BINS[i]) & (t < BINS[i + 1])].mean() if np.any((t >= BINS[i]) & (t < BINS[i + 1])) else np.nan
             for i in range(len(BINS) - 1)]
    col, mk = STYLE[m]
    ax.plot(ctr, means, color=col, marker=mk, ms=3, lw=1.2, label=m, zorder=3)
ax.set_xlabel("True attribute value $t_a$")
ax.set_ylabel("Mean claim $c_a$")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.grid(color="0.92", lw=0.5)
ax.legend(loc="lower right", fontsize=6.5, frameon=False, handletextpad=0.4, labelspacing=0.25)
fig.tight_layout(pad=0.4)
fig.savefig(OUT / "deception_claim_vs_truth.pdf")
fig.savefig(OUT / "deception_claim_vs_truth.png", dpi=300)
plt.close(fig)
print("wrote deception_claim_vs_truth.(pdf|png)")

# ===== Figure 2: ladder + transfer =====
tiers = ["Trivial-Max", "Truth-Anch.", "Self-Aware", "Pack-Aware", "RL"]
mim = np.array([12.8, 19.0, 22.9, 24.4, 35.2])
mim_lo = np.array([11.6, 17.0, 20.6, 21.9, 33.2]); mim_hi = np.array([14.0, 21.1, 25.3, 26.9, 37.2])
real = np.array([13.6, 18.3, 25.8, 23.3, 29.2])
real_lo = np.array([9.3, 11.0, 17.2, 14.1, 21.6]); real_hi = np.array([17.9, 25.7, 34.5, 32.6, 36.7])
x = np.arange(len(tiers)); w = 0.38
fig, ax = plt.subplots(figsize=(3.5, 2.6))
ax.bar(x - w / 2, mim, w, yerr=[mim - mim_lo, mim_hi - mim], capsize=2,
       color="#3B6EA5", label="vs. mimic field", error_kw=dict(lw=0.7))
ax.bar(x + w / 2, real, w, yerr=[real - real_lo, real_hi - real], capsize=2,
       color="#B23A48", label="vs. real LLMs", error_kw=dict(lw=0.7))
ax.axhline(20, ls=":", lw=0.8, color="0.4", zorder=0)
ax.text(0.015, 0.53, "fair share", transform=ax.transAxes, fontsize=6.5, color="0.4", ha="left")
ax.set_ylabel("Win rate (%)")
ax.set_xticks(x); ax.set_xticklabels(tiers, rotation=22, ha="right", fontsize=7)
ax.set_ylim(0, 40)
ax.grid(axis="y", color="0.92", lw=0.5)
ax.legend(loc="upper left", fontsize=7, frameon=False)
fig.tight_layout(pad=0.4)
fig.savefig(OUT / "deception_ladder_transfer.pdf")
fig.savefig(OUT / "deception_ladder_transfer.png", dpi=300)
plt.close(fig)
print("wrote deception_ladder_transfer.(pdf|png)")
