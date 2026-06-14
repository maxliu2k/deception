"""Deception-mimic figures: the evidence that each mimic reproduces its LLM.

Produces three paper-ready figures from the existing dataset + trained mimics +
fit report (no new games are run):

  1. deception_claim_dist.{pdf,png}
        Real-LLM vs mimic claim distributions, one panel per LLM (pooled over
        attributes). Visual proof the calibrated-noise mimic matches the real
        spread rather than collapsing to the conditional mean.

  2. deception_info_gradient.{pdf,png}
        Mean total-variation distance (real vs mimic) against the number of
        known attributes k, one line per LLM, with the acceptance threshold.
        Shows where fidelity holds (high info) and degrades (low info).

  3. deception_lie_fingerprint.{pdf,png}
        Per-LLM lie propensity on visible attributes, real vs mimic, as paired
        bars. Shows the clone reproduces each model's tendency to deceive.

Usage:
    python -m simulation.make_deception_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation.validate_deception_mimics import load_dataset, sample_mimic_claims, _row_level
from simulation.mimic_agent import _load_deception_mimic

HERE = Path(__file__).parent
DATASETS = HERE / "datasets"
OUT = HERE / "exports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6})

DATA = DATASETS / "deception_dataset_v3.jsonl"
META = DATASETS / "deception_dataset_v3_meta.json"
FIT = DATASETS / "deception_fit_report_v1.json"

# Display order + colors (mirrors make_auction_figures.py; dataset uses GPT-5.4).
ORDER = ["GPT-5.4", "Opus", "Pro", "Grok", "Llama"]
COLOR = {
    "GPT-5.4": "#08519C",  # deep blue
    "Opus":    "#D26E00",  # deep orange
    "Pro":     "#1A7A3D",  # deep green
    "Grok":    "#000000",  # black
    "Llama":   "#A8327D",  # deep magenta
}
MARK = {"GPT-5.4": "o", "Opus": "s", "Pro": "^", "Grok": "D", "Llama": "v"}


def _save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / (stem + '.pdf')}  (+ .png)")


def _load_rows_by_alias():
    rows = load_dataset(DATA)
    meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}
    vocab = meta.get("bidder_vocab") or {}
    idx_to_alias = {v: k for k, v in vocab.items()}
    by_alias: dict[str, list[dict]] = {}
    for r in rows:
        alias = idx_to_alias.get(int(r["bidder_index"]), f"bidder_{r['bidder_index']}")
        by_alias.setdefault(alias, []).append(r)
    return by_alias


# ── Figure 1: real vs mimic claim distributions ──────────────────────────────

def fig_claim_distributions(by_alias: dict[str, list[dict]]) -> None:
    aliases = [a for a in ORDER if a in by_alias]
    fig, axes = plt.subplots(1, len(aliases), figsize=(2.05 * len(aliases), 2.1), sharey=True)
    if len(aliases) == 1:
        axes = [axes]
    bins = np.linspace(0.0, 1.0, 21)
    for ax, alias in zip(axes, aliases):
        rows = by_alias[alias]
        real = np.array([r["y_claim"] for r in rows], dtype=np.float32).ravel()
        mimic = sample_mimic_claims(alias, rows, n_samples_per_row=4)
        c = COLOR.get(alias, "0.2")
        ax.hist(real, bins=bins, density=True, histtype="stepfilled",
                color=c, alpha=0.30, label="real LLM")
        if mimic is not None:
            ax.hist(mimic.ravel(), bins=bins, density=True, histtype="step",
                    color=c, lw=1.4, linestyle="--", label="mimic")
        ax.set_title(alias.replace("GPT-5.4", "GPT"), fontsize=9)
        ax.set_xlabel("claim value")
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("density")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.10), fontsize=8)
    fig.suptitle("Claim distributions: real LLM vs. calibrated mimic", y=1.18, fontsize=9)
    _save(fig, "deception_claim_dist")


# ── Figure 2: information gradient (TVD vs known-attr count) ──────────────────

def fig_info_gradient() -> None:
    if not FIT.exists():
        print("  skip info-gradient: fit report not found")
        return
    report = json.loads(FIT.read_text(encoding="utf-8"))
    thr = float(report.get("max_tvd", 0.15))
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    for entry in report.get("per_llm", []):
        if entry.get("skipped"):
            continue
        alias = entry["alias"]
        pl = sorted(entry.get("per_level", []), key=lambda d: d["known"])
        ks = [p["known"] for p in pl]
        tvds = [p["mean_tvd"] for p in pl]
        ax.plot(ks, tvds, marker=MARK.get(alias, "o"), color=COLOR.get(alias, "0.2"),
                lw=1.2, ms=4, label=alias.replace("GPT-5.4", "GPT"))
    ax.axhline(thr, color="0.4", ls=":", lw=1.0)
    ax.text(1.05, thr + 0.012, f"accept ≤ {thr:g}", fontsize=7, color="0.35")
    ax.set_xlabel("known attributes  k  (information level)")
    ax.set_ylabel("mean TVD  (real vs mimic)")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(bottom=0)
    ax.set_title("Fidelity by information level\n(lower = closer match)", fontsize=9)
    ax.legend(frameon=False, fontsize=7, ncol=1, loc="upper right")
    _save(fig, "deception_info_gradient")


# ── Figure 3: lie-rate fingerprint (real vs mimic) ───────────────────────────

def _lie_fingerprint_stats(alias: str, rows: list[dict], *, n_boot: int = 2000,
                           seed: int = 0) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Real and mimic lie rate (%) on visible attrs, each with a 95% bootstrap CI.

    The bootstrap resamples whole EPISODES with replacement — episodes are the
    independent unit, since the 12 rounds within one are correlated. Resampling
    rows would understate uncertainty. Returns ((real_pt, real_lo, real_hi),
    (mimic_pt, mimic_lo, mimic_hi))."""
    yl = np.array([r["y_lied"] for r in rows], dtype=np.float32)
    vis = np.array([r.get("visible_mask", [1, 1, 1, 1, 1]) for r in rows], dtype=np.float32)
    X = np.array([r["x"] for r in rows], dtype=np.float32)
    eps = np.array([int(r.get("episode_index", 0)) for r in rows])

    model = _load_deception_mimic(alias)
    if model is not None:
        with torch.no_grad():
            lie_logits, _ = model(torch.from_numpy(X))
            p_lie = torch.sigmoid(lie_logits).cpu().numpy()
    else:
        p_lie = np.zeros_like(yl)

    uniq = np.unique(eps)
    idx_by_ep = [np.where(eps == u)[0] for u in uniq]
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))

    def stats(values5: np.ndarray) -> tuple[float, float, float]:
        # Per-episode numerator/denominator, then bootstrap by summing sampled
        # episodes (a pooled rate, not a mean-of-episode-rates).
        ep_num = np.array([float((values5[ix] * vis[ix]).sum()) for ix in idx_by_ep])
        ep_den = np.array([float(vis[ix].sum()) for ix in idx_by_ep])
        tot_den = ep_den.sum()
        point = 100.0 * ep_num.sum() / tot_den if tot_den > 0 else float("nan")
        num = ep_num[boot_idx].sum(axis=1)
        den = ep_den[boot_idx].sum(axis=1)
        rates = 100.0 * np.where(den > 0, num / den, np.nan)
        lo, hi = np.nanpercentile(rates, [2.5, 97.5])
        return point, float(lo), float(hi)

    return stats(yl), stats(p_lie)


def fig_lie_fingerprint(by_alias: dict[str, list[dict]]) -> None:
    aliases = [a for a in ORDER if a in by_alias]
    stats = {a: _lie_fingerprint_stats(a, by_alias[a]) for a in aliases}
    real = np.array([stats[a][0][0] for a in aliases])
    real_lo = np.array([stats[a][0][1] for a in aliases])
    real_hi = np.array([stats[a][0][2] for a in aliases])
    mimic = np.array([stats[a][1][0] for a in aliases])
    mimic_lo = np.array([stats[a][1][1] for a in aliases])
    mimic_hi = np.array([stats[a][1][2] for a in aliases])
    err_real = np.vstack([real - real_lo, real_hi - real])
    err_mimic = np.vstack([mimic - mimic_lo, mimic_hi - mimic])

    x = np.arange(len(aliases))
    w = 0.38
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.bar(x - w / 2, real, w, color=[COLOR.get(a, "0.2") for a in aliases],
           alpha=0.85, label="real LLM",
           yerr=err_real, capsize=2.5, error_kw={"lw": 0.8, "ecolor": "0.25"})
    ax.bar(x + w / 2, mimic, w, color=[COLOR.get(a, "0.2") for a in aliases],
           alpha=0.85, hatch="////", edgecolor="white", linewidth=0.0, label="mimic",
           yerr=err_mimic, capsize=2.5, error_kw={"lw": 0.8, "ecolor": "0.25"})
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("GPT-5.4", "GPT") for a in aliases])
    ax.set_ylabel("lie rate on visible attrs (%)")
    ax.set_title("Deception fingerprint: real LLM vs. mimic\n(95% bootstrap CI over episodes)", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    _save(fig, "deception_lie_fingerprint")


def main() -> int:
    print("Building deception figures ->", OUT)
    by_alias = _load_rows_by_alias()
    fig_claim_distributions(by_alias)
    fig_info_gradient()
    fig_lie_fingerprint(by_alias)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
