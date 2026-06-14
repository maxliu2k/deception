"""Strategy performance against a realistic opponent field (parallels the auction
study's "baselines vs. the mimic field", Fig. 2).

Each strategy under test plays as the focal agent (seat 0) against a FIXED field
of four behavioral mimics (validated LLM surrogates), over the same matched truth
seeds / information levels as the real episodes. Because every strategy faces an
IDENTICAL field, outcomes — including reward — are directly comparable across
strategies (unlike the homogeneous-field benchmark, where compositions differ).

Focal strategies: the nine math models (ladder T1-T4 + five archetypes) and the
five LLMs (represented by their mimics). Metrics: catch rate, win rate, and
reward per round, each with 95% bootstrap CIs over episodes.

Requires the server running (default http://localhost:8010).

Usage:
    python -m simulation.benchmark_vs_mimics [--limit N] [--parallel 8] [--replot]
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from simulation.run_deception_competition import run_one, find_or_create_folder
from simulation.validate_mimic_outcomes import _episode_log_path, _read_log, gather_real, MANIFEST

HERE = Path(__file__).parent
OUT_REPORT = HERE / "datasets" / "deception_vsmimic_report_v1.json"
FIG_DIR = HERE / "exports" / "figures"

# Fixed realistic opponent field (4 mimics; the focal is the 5th seat).
MIMIC_FIELD = ["Mimic-Grok", "Mimic-Opus", "Mimic-GPT-5.4", "Mimic-Pro"]

LADDER = ["Math-T1", "Math-T2", "Math-T3", "Math-T4"]
ARCHETYPES = ["Math-Evade", "Math-Risk", "Math-Target", "Math-Optimist", "Math-Reactive"]
MATH_MODELS = LADDER + ARCHETYPES
TIER_LABEL = {"Math-T1": "T1\nhonest", "Math-T2": "T2\ninflate",
              "Math-T3": "T3\n1-attr", "Math-T4": "T4\nNash",
              "Math-Evade": "Evade", "Math-Risk": "Risk", "Math-Target": "Target",
              "Math-Optimist": "Optimist", "Math-Reactive": "Reactive"}
LLM_ORDER = ["GPT-5.4", "Opus", "Pro", "Grok", "Llama"]
LLM_COLOR = {"GPT-5.4": "#08519C", "Opus": "#D26E00", "Pro": "#1A7A3D",
             "Grok": "#000000", "Llama": "#A8327D"}


def _boot_ratio_ci(nums, dens, *, n_boot=2000, seed=0):
    """Point estimate sum(num)/sum(den) and 95% bootstrap CI over records."""
    nums = np.asarray(nums, float); dens = np.asarray(dens, float)
    if dens.sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    point = float(nums.sum() / dens.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(nums), size=(n_boot, len(nums)))
    bn = nums[idx].sum(axis=1); bd = dens[idx].sum(axis=1)
    rates = np.where(bd > 0, bn / bd, np.nan)
    lo, hi = np.nanpercentile(rates, [2.5, 97.5])
    return point, float(lo), float(hi)


def aggregate(records: list[dict]) -> dict:
    if not records:
        return {}
    lie = np.array([r["lie"] for r in records], float)
    vis = np.array([r["vis"] for r in records], float)
    caught = np.array([r["caught"] for r in records], float)
    wins = np.array([r["wins"] for r in records], float)
    ar = np.array([r["agent_rounds"] for r in records], float)
    reward = np.array([r["reward"] for r in records], float)
    return {
        "n_records": len(records),
        "lie_rate": [round(100 * x, 3) for x in _boot_ratio_ci(lie, vis, seed=1)],
        "catch_rate": [round(x, 4) for x in _boot_ratio_ci(caught, ar, seed=2)],
        "win_rate": [round(x, 4) for x in _boot_ratio_ci(wins, ar, seed=3)],
        "reward_per_round": [round(x, 4) for x in _boot_ratio_ci(reward, ar, seed=4)],
    }


def _focal_record(log: dict, focal_index: int = 0) -> dict | None:
    """Behaviour record for the focal agent (seat 0) only — identified by seat,
    not alias, so a mimic focal that also appears in the field is not double
    counted."""
    agents = log.get("agents", [])
    focal = min(agents, key=lambda a: int(a.get("agent_index", 99)), default=None) if agents else None
    if focal is None:
        return None
    aid = focal["agent_id"]
    n = int(log.get("num_rounds", 12))
    rec = {"lie": 0, "vis": 0, "caught": int(focal.get("caught_count", 0)),
           "wins": int(focal.get("win_count", 0)),
           "reward": float(focal.get("total_reward", 0.0)), "agent_rounds": n}
    for r in log.get("rounds", []):
        truth = r["truth"]
        c = (r.get("claims_by_agent", {}) or {}).get(aid)
        if c is None:
            continue
        visible = (r.get("visible_attrs_by_agent", {}) or {}).get(aid) or list(range(len(truth)))
        for a_ in visible:
            rec["vis"] += 1
            if round(float(c[a_]), 2) > round(float(truth[a_]), 2):
                rec["lie"] += 1
    return rec


def run_focal_vs_field(base: str, focal: str, configs: list, *, parallel: int,
                       timeout_s: float, poll_s: float) -> list[dict]:
    folder_id = find_or_create_folder(base, "VsMimic Benchmark")
    loadout = [focal] + MIMIC_FIELD
    lock = threading.Lock()
    done = {"n": 0}
    recs: list[dict] = []

    def _task(item):
        i, (key, cfg) = item
        truth_seed, known = key
        mode = "deception_competition" if known >= 5 else "deception_competition_partial_info"
        res = run_one(
            base, folder_id=folder_id, seed=truth_seed, truth_seed=truth_seed,
            slot_name=f"{focal} vsfield ts{truth_seed} k{known}", loadout=loadout, mode=mode,
            known_attrs=known, threshold=cfg["threshold"], penalty=cfg["penalty"],
            num_rounds=12, preferences=cfg["preferences"] or None,
            poll_interval_s=poll_s, timeout_s=timeout_s,
        )
        out = None
        if res.get("ok") and res.get("slot_id"):
            log = _read_log(_episode_log_path(res["slot_id"]))
            if log and log.get("complete"):
                out = _focal_record(log)
        with lock:
            done["n"] += 1
            print(f"    [{focal}] {done['n']:3d}/{len(configs)} "
                  + ("ok" if out else f"FAIL ({res.get('error')})"))
        return out

    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futs = [ex.submit(_task, it) for it in enumerate(configs)]
        for f in as_completed(futs):
            r = f.result()
            if r:
                recs.append(r)
    return recs


def make_figure(summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6})
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ps = summary["per_strategy"]
    order = [s for s in (MATH_MODELS + LLM_ORDER) if ps.get(s)]  # only strategies with data

    def grp_color(s):
        if s in LADDER:
            return "#B03050" if s == "Math-T4" else "#9E9E9E"
        if s in ARCHETYPES:
            return "#2A9D8F"
        return LLM_COLOR.get(s, "0.3")

    labels = [TIER_LABEL.get(s, s.replace("GPT-5.4", "GPT")) for s in order]
    colors = [grp_color(s) for s in order]
    x = np.arange(len(order))
    panels = [("reward_per_round", "reward per round"),
              ("win_rate", "win rate (per round)"),
              ("catch_rate", "catch rate (per round)")]
    fig, axes = plt.subplots(1, 3, figsize=(max(8.5, 1.6 * len(order) + 3.0), 3.4))
    for ax, (key, ylabel) in zip(axes, panels):
        pts, los, his = [], [], []
        for s in order:
            v = ps.get(s, {}).get(key, [np.nan, np.nan, np.nan])
            pts.append(v[0]); los.append(v[0] - v[1]); his.append(v[2] - v[0])
        ax.bar(x, pts, color=colors, alpha=0.9,
               yerr=np.vstack([los, his]), capsize=2, error_kw={"lw": 0.7, "ecolor": "0.3"})
        if ps.get("Math-T4"):  # Nash reference only if it was run
            ax.axhline(ps["Math-T4"][key][0], color="#B03050", ls=":", lw=0.9, zorder=0)
        ax.axhline(0, color="0.6", lw=0.6, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.0, rotation=20, ha="right")
        ax.set_ylabel(ylabel, fontsize=7.8)
        ax.tick_params(labelsize=6.8)
    fig.suptitle("Performance against a fixed 4-mimic field   "
                 "(teal = new strategy archetypes; all face the same opponents)", y=1.04, fontsize=9.5)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"deception_vs_mimic_field.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG_DIR / 'deception_vs_mimic_field.pdf'}  (+ .png)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="http://localhost:8010")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--poll", type=float, default=0.5)
    p.add_argument("--replot", action="store_true")
    p.add_argument("--models", default="", help="Comma list of focal strategies to (re)run; "
                   "default = all 9 math models + 5 LLM mimics. Others reused from saved report.")
    args = p.parse_args()

    if args.replot:
        make_figure(json.loads(OUT_REPORT.read_text(encoding="utf-8")))
        print(f"Replotted from {OUT_REPORT}")
        return 0

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    real = gather_real(manifest)
    configs = list(real.items())
    if args.limit and args.limit > 0:
        configs = configs[: args.limit]
    print(f"Configs: {len(configs)}  |  field: {MIMIC_FIELD}")

    per_strategy: dict[str, dict] = {}
    if OUT_REPORT.exists():
        per_strategy = (json.loads(OUT_REPORT.read_text(encoding="utf-8")) or {}).get("per_strategy", {})

    # Focal aliases to run; LLM focals run as their mimic but are STORED under the
    # bare LLM name so the figure keys line up with LLM_ORDER.
    default_focals = list(MATH_MODELS) + [f"Mimic-{a}" for a in LLM_ORDER]
    focals = [m.strip() for m in args.models.split(",") if m.strip()] or default_focals

    for focal in focals:
        store_key = focal[len("Mimic-"):] if focal.startswith("Mimic-") else focal
        print(f"Running focal {focal} vs field ...")
        recs = run_focal_vs_field(args.base, focal, configs, parallel=args.parallel,
                                  timeout_s=args.timeout, poll_s=args.poll)
        per_strategy[store_key] = aggregate(recs)

    summary = {"n_configs": len(configs), "field": MIMIC_FIELD, "per_strategy": per_strategy}
    OUT_REPORT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'strategy':12s} {'lie%':>7s} {'catch':>7s} {'win':>7s} {'rwd/rnd':>8s}  (n)")
    for s in MATH_MODELS + LLM_ORDER:
        r = per_strategy.get(s, {})
        if r:
            print(f"{s:12s} {r['lie_rate'][0]:7.2f} {r['catch_rate'][0]:7.3f} "
                  f"{r['win_rate'][0]:7.3f} {r['reward_per_round'][0]:8.3f}  ({r['n_records']})")
    make_figure(summary)
    print(f"Report -> {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
