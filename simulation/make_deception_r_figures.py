"""Figures for the Option-R deception results (deception_results_R.tex).

Produces three PDFs/PNGs in simulation/exports/figures/:
  1. deception_r_ladder      - strategy reward vs the mimic field (the ladder)
  2. deception_r_transfer    - mimic-field ladder vs real-LLM transfer (it doesn't transfer)
  3. deception_r_escalation  - over-claim vs incoming trust, per LLM (no trust reasoning)

Reads deception_vsmimic_report_R.json + scans save_slots for the transfer and
LLM-vs-LLM episodes. Run:  python -m simulation.make_deception_r_figures
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
SLOTS = HERE / ".runtime" / "save_slots"
VSMIMIC = HERE / "datasets" / "deception_vsmimic_report_R.json"
FIGDIR = HERE / "exports" / "figures"
LLMS = {"GPT-5.4", "Opus", "Pro", "Grok", "Llama"}
FAIR = 0.20

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.6})


def _logs():
    for f in glob.glob(str(SLOTS / "*" / "auction_exports" / "deception_episode" / "episode_log.json")):
        try:
            yield f, json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue


# ── Figure 1: ladder vs mimic field ──────────────────────────────────────────
def fig_ladder():
    d = json.loads(VSMIMIC.read_text(encoding="utf-8"))["per_focal"]
    order = ["Math-T2", "Math-T1", "Mimic-Grok", "Mimic-GPT-5.4", "Mimic-Pro",
             "Mimic-Llama", "Mimic-Opus", "Math-T4", "Math-T3"]
    lab = {"Math-T1": "T1 honest", "Math-T2": "T2 naive", "Math-T3": "T3 conditional",
           "Math-T4": "T4 backloaded", "Mimic-GPT-5.4": "GPT", "Mimic-Opus": "Opus",
           "Mimic-Pro": "Pro", "Mimic-Grok": "Grok", "Mimic-Llama": "Llama"}
    order = [k for k in order if k in d]
    vals = [d[k]["reward_per_round"] for k in order]
    pts = [v[0] for v in vals]
    err = [[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]]
    colors = ["#B03050" if k.startswith("Math") else "#2A6F97" for k in order]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    y = np.arange(len(order))
    ax.barh(y, pts, xerr=err, color=colors, alpha=0.9, capsize=2, error_kw={"lw": 0.7, "ecolor": "0.4"})
    ax.axvline(FAIR, color="0.4", ls=":", lw=1.0)
    ax.text(FAIR + 0.005, len(order) - 0.4, "fair share", fontsize=7.5, color="0.35")
    ax.set_yticks(y); ax.set_yticklabels([lab[k] for k in order], fontsize=8)
    ax.set_xlabel("reward per round (95% CI)")
    ax.set_title("Strategy vs. the mimic field: calibrated deception (T3/T4) wins;\n"
                 "every LLM lands between honest and naive", fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#B03050", label="math tier"), Patch(color="#2A6F97", label="LLM")],
              frameon=False, fontsize=7.5, loc="lower right")
    _save(fig, "deception_r_ladder")


# ── Figure 2: transfer ────────────────────────────────────────────────────────
def fig_transfer():
    vm = json.loads(VSMIMIC.read_text(encoding="utf-8"))["per_focal"]
    tiers = ["Math-T1", "Math-T2", "Math-T3", "Math-T4"]
    lab = {"Math-T1": "T1 honest", "Math-T2": "T2 naive", "Math-T3": "T3 cond.", "Math-T4": "T4 backl."}
    # real-LLM reward: scan transfer episodes (one Math-T* focal + the 4 real LLMs)
    real = defaultdict(list)
    field = {"GPT-5.4", "Opus", "Pro", "Grok"}
    for _, d in _logs():
        if not d.get("complete"):
            continue
        ag = d.get("agents", [])
        foc = [a for a in ag if str(a.get("alias", "")).startswith("Math-T")]
        if len(foc) != 1 or ({a["alias"] for a in ag} - {foc[0]["alias"]}) != field:
            continue
        nr = int(d.get("num_rounds", 12)) or 12
        real[foc[0]["alias"]].append(float(foc[0].get("total_reward", 0.0)) / nr)
    mimic_r = [vm[t]["reward_per_round"][0] for t in tiers]
    real_r = [float(np.mean(real[t])) if real.get(t) else np.nan for t in tiers]
    x = np.arange(len(tiers)); wbar = 0.38
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.bar(x - wbar / 2, mimic_r, wbar, label="vs. mimic field", color="#2A6F97", alpha=0.9)
    ax.bar(x + wbar / 2, real_r, wbar, label="vs. real LLMs", color="#B03050", alpha=0.9)
    ax.axhline(FAIR, color="0.4", ls=":", lw=1.0)
    ax.text(len(tiers) - 0.5, FAIR + 0.008, "fair share", fontsize=7.5, color="0.35", ha="right")
    ax.set_xticks(x); ax.set_xticklabels([lab[t] for t in tiers], fontsize=8)
    ax.set_ylabel("reward per round")
    ax.set_title("The mimic-field ladder does not transfer:\n"
                 "honest collapses to 0, the T3/T4 advantage flattens", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)
    _save(fig, "deception_r_transfer")


# ── Figure 3: escalation (over-claim vs incoming trust) ───────────────────────
def fig_escalation():
    cand = []
    for f, d in _logs():
        if not d.get("complete") or {a["alias"] for a in d.get("agents", [])} != LLMS:
            continue
        cand.append((os.path.getmtime(f), d))
    cand.sort(reverse=True)
    cand = [d for _, d in cand[:50]]                      # newest 50 = the v2 run
    bins = defaultdict(lambda: {"hi": [], "mid": [], "lo": []})
    for d in cand:
        id2al = {a["agent_id"]: a["alias"] for a in d["agents"]}
        for r in d.get("rounds", []):
            tb = r.get("trust_before", {}); disc = r.get("discrepancy_by_agent", {})
            for aid, al in id2al.items():
                t, o = tb.get(aid), disc.get(aid)
                if t is None or o is None:
                    continue
                b = "hi" if t >= 0.6 else ("mid" if t >= 0.3 else "lo")
                bins[al][b].append(float(o))
    models = ["Llama", "GPT-5.4", "Grok", "Opus", "Pro"]
    keys = ["hi", "mid", "lo"]; klab = ["trust≥0.6", "0.3-0.6", "<0.3"]
    cols = ["#9ECAE1", "#4292C6", "#08519C"]
    x = np.arange(len(models)); wbar = 0.26
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for j, k in enumerate(keys):
        vals = [np.mean(bins[m][k]) if bins[m][k] else 0.0 for m in models]
        ax.bar(x + (j - 1) * wbar, vals, wbar, label=klab[j], color=cols[j])
    ax.set_xticks(x); ax.set_xticklabels(["Llama", "GPT", "Grok", "Opus", "Pro"], fontsize=8)
    ax.set_ylabel("mean over-claim  $d=\\max(0, w\\cdot(c-t))$")
    ax.set_title("LLMs over-claim MORE as their trust craters\n"
                 "(opposite of recovery; futile at trust≈0)", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, title="incoming trust", title_fontsize=7.5)
    _save(fig, "deception_r_escalation")


def _save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIGDIR / (name + '.pdf')}  (+ .png)")


if __name__ == "__main__":
    fig_ladder()
    fig_transfer()
    fig_escalation()
    print("done")
