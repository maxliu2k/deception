"""Auction bid figure: side-by-side scatter of winning bid per painting,
points colored+marker-coded by winning bidder, for real vs mimic auctions.
Density is shown by overplotting (transparent markers -> darker = denser),
with a dashed per-painting mean line. Shared y-axis for comparability.

Outputs auction_bids.pdf (paper) and .png (preview).
"""
import csv
import collections
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DATASETS = HERE / "datasets"
OUT = HERE / "exports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6})

ORDER = ["Llama", "Pro", "Opus", "GPT", "Grok"]
STYLE = {
    "GPT":   ("#0072B2", "o"),
    "Opus":  ("#E69F00", "s"),
    "Pro":   ("#009E73", "^"),
    "Grok":  ("#000000", "D"),
    "Llama": ("#CC79A7", "v"),
}


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            m = r["winner_model"].replace("GPT-5.4", "GPT")
            rows.append((int(r["painting_index"]), m, float(r["winning_bid"])))
    return rows


def draw(ax, rows, ylim, alpha):
    rng = np.random.default_rng(0)
    for m in ORDER:
        xs = [p + rng.uniform(-0.20, 0.20) for p, mm, _ in rows if mm == m]
        ys = [b for _, mm, b in rows if mm == m]
        c, mk = STYLE[m]
        # sparse panel -> opaque; dense panel -> transparent so overplotting reads as density
        ax.scatter(xs, ys, s=11, color=c, marker=mk, linewidths=0,
                   label=m, alpha=alpha, zorder=3)
    by = collections.defaultdict(list)
    for p, _, b in rows:
        by[p].append(b)
    px = sorted(by)
    ax.plot(px, [np.mean(by[p]) for p in px], color="0.25", lw=1.1,
            ls="--", zorder=4, label="mean")
    ax.set_xlabel("Painting #")
    ax.set_xticks(range(1, 13))
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color="0.92", lw=0.5)


real = load(DATASETS / "recent_real_auction_bids_long.csv")
mimic = load(DATASETS / "mimic_auction_bids_long.csv")
ymax = max(b for _, _, b in real + mimic)
ylim = (0, int(np.ceil(ymax / 1000.0)) * 1000 + 300)

# single-column: stack the two panels vertically (real over mimic)
fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(3.4, 3.7))
draw(axes[0], real, ylim, alpha=1.0)
draw(axes[1], mimic, ylim, alpha=0.4)
for ax in axes:
    ax.set_ylabel("Winning bid")
axes[0].set_xlabel("")
for ax, lab in zip(axes, ["(a) Real LLMs", "(b) Mimics"]):
    ax.annotate(lab, xy=(0.03, 0.88), xycoords="axes fraction",
                fontsize=8, fontweight="bold")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=3, fontsize=6.5, frameon=False,
           loc="upper center", bbox_to_anchor=(0.5, 1.04),
           handletextpad=0.2, columnspacing=1.0)
fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "auction_bids.pdf", bbox_inches="tight")
fig.savefig(OUT / "auction_bids.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote auction_bids.(pdf|png)")
