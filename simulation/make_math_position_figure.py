"""Small-multiples version of 'auctions won by painting position' per math
policy. Each policy gets its own panel (bold), with the other four drawn
faintly for context -> no line tangle, none de-emphasized.

Win fractions are recomputed from folders 21-25 (50 auctions each); they
average to the per-policy totals in the baseline table.
"""
import sys, json, pickle
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path("simulation")
OUT = HERE / "exports" / "figures"
cat = json.load(open(HERE / ".runtime" / "save_slots.json", encoding="utf-8"))
folders = {f["folder_id"]: f.get("name", "") for f in cat.get("folders", [])}
slots = cat.get("slots", {})
if isinstance(slots, list):
    slots = {s["slot_id"]: s for s in slots}
byfolder = defaultdict(list)
for sid, m in slots.items():
    byfolder[m.get("folder_id")].append(sid)
name2fid = {v: k for k, v in folders.items()}


def load(sid):
    return pickle.load(open(HERE / ".runtime" / "save_slots" / sid / "runtime.pkl", "rb"))


WANT = [("Math v7 T1", "Trivial", "#D55E00"),
        ("Math v7 T2", "Fair-Share", "#E69F00"),
        ("Math v7 T3", "Reactive", "#009E73"),
        ("Math v7 T4", "Market-Clearing", "#0072B2"),
        ("Math v7 T5", "RL", "#000000")]

curves = {}
for fname, tier, c in WANT:
    fid = name2fid.get(fname)
    win = np.zeros(12); tot = np.zeros(12)
    for sid in byfolder.get(fid, []):
        try:
            obj = load(sid)
        except Exception:
            continue
        der = obj["last_result"]["derived"]; reset = obj["last_reset"]
        cp = der.get("completed_paintings", [])
        if len(cp) != 12:
            continue
        seat = {f"bidder_{i+1}": reset["selected_models"][i] for i in range(len(reset["selected_models"]))}
        for idx, p in enumerate(cp):
            tot[idx] += 1
            if str(seat.get(p["winner_id"], "")).startswith("Math-"):
                win[idx] += 1
    curves[tier] = (100 * win / np.maximum(tot, 1), c)

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6})
tiers = [t for _, t, _ in WANT]
# full-width thin banner: five panels in a row
fig, axes = plt.subplots(1, 5, sharey=True, figsize=(7.16, 1.45))
x = np.arange(1, 13)
for ax, focal in zip(axes, tiers):
    for t in tiers:                       # faint backdrop of all others
        if t == focal:
            continue
        ax.plot(x, curves[t][0], color="0.72", lw=0.8, alpha=0.6, zorder=1)
    y, c = curves[focal]                  # bold focal
    ax.plot(x, y, color=c, lw=1.5, marker="o", ms=2.2, zorder=3)
    ax.axhline(20, ls="--", lw=0.6, color="0.55", zorder=0)
    ax.set_title(focal, fontsize=7.5, color=c)
    ax.set_xticks([1, 6, 12])
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 50, 100])
    ax.grid(color="0.94", lw=0.5)
axes[0].set_ylabel("Auctions won (%)")
fig.supxlabel("Painting #", fontsize=8, y=0.04)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "math_wins_by_position.pdf")
fig.savefig(OUT / "math_wins_by_position.png", dpi=300)
print("wrote math_wins_by_position.(pdf|png)")
