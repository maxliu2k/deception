"""Sweep DECEPTION_MIMIC_TEMPERATURE and report fidelity at each value.

For each T in the sweep, run 50 all-mimic episodes and compute outcome-level
fidelity (per-LLM win share delta + Cramer's V) against the real 50-episode
dataset. Picks the best T by lowest Cramer's V.

Each fidelity pass takes ~30s, so a 10-value sweep is ~5 minutes.
"""
from __future__ import annotations
import asyncio
import glob
import json
import math
import os
import re
import statistics
from collections import defaultdict

import numpy as np

from simulation.env import TravelGameEnv
from simulation.server import _build_actions_live_deception_competition

LLM_LIST = ["GPT-5.4", "Grok", "Opus", "Pro", "Llama"]
MIMICS = [f"Mimic-{a}" for a in LLM_LIST]
SWEEP = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]


def load_real_wins() -> dict[str, int]:
    wins = defaultdict(int)
    for p in glob.glob("simulation/.runtime/save_slots/save_slot_*/auction_exports/deception_episode/episode_log.json"):
        m = re.search(r"save_slot_(\d+)", p)
        if not m or int(m.group(1)) < 1203:
            continue
        try:
            ep = json.load(open(p))
        except Exception:
            continue
        if not ep.get("complete"):
            continue
        for a in ep["agents"]:
            if a["alias"] in LLM_LIST:
                wins[a["alias"]] += a["win_count"]
    return wins


async def collect_mimic_wins(*, num_truth_seeds: int = 50, num_replicates: int = 3, num_rounds: int = 12) -> dict[str, int]:
    """Run num_replicates x num_truth_seeds = total mimic episodes (default 150)."""
    wins = defaultdict(int)
    for rep_i in range(num_replicates):
        for truth_i in range(num_truth_seeds):
            truth_seed = truth_i + 1
            env_seed = truth_seed * 1000 + rep_i + 1
            env = TravelGameEnv({
                "mode": "deception_competition",
                "selected_models": MIMICS,
                "num_rounds": num_rounds,
                "truth_seed": truth_seed,
            })
            env.reset(seed=env_seed)
            while not env.done:
                actions = await _build_actions_live_deception_competition(env, {"use_models": False})
                env.step(actions)
            for a in env.world["deception_episode"].agent_states:
                alias = a.alias.replace("Mimic-", "")
                wins[alias] += a.win_count
    return wins


def chi_sq_v(real: dict[str, int], mimic: dict[str, int]) -> tuple[float, float, float]:
    obs = np.array([[real[a] for a in LLM_LIST], [mimic[a] for a in LLM_LIST]], dtype=float)
    row_sums = obs.sum(axis=1, keepdims=True)
    col_sums = obs.sum(axis=0, keepdims=True)
    grand = obs.sum()
    expected = (row_sums @ col_sums) / grand
    chi2 = float(np.nansum((obs - expected) ** 2 / expected))
    df = len(LLM_LIST) - 1
    z = ((chi2 / df) ** (1/3) - (1 - 2/(9*df))) / math.sqrt(2/(9*df))
    p = 0.5 * math.erfc(z / math.sqrt(2))
    v = math.sqrt(chi2 / grand)
    return chi2, p, v


async def main():
    real_wins = load_real_wins()
    real_total = sum(real_wins.values())
    real_pct = {a: real_wins[a] / real_total * 100 for a in LLM_LIST}

    print(f"Real LLM win %: " + ", ".join(f"{a}={real_pct[a]:.1f}%" for a in LLM_LIST))
    print()
    print(f"{'T':>5} {'chi2':>8} {'p':>8} {'V':>8}  " + "  ".join(f"{a:>9}" for a in LLM_LIST))
    print("-" * 90)

    results = []
    for T in SWEEP:
        # Set env var and clear cached mimics so each agent re-reads T per call
        # (actually only the dispatch reads T, not the loader — so cache reset
        # is unnecessary; we just need the env var set before mimic calls fire).
        os.environ["DECEPTION_MIMIC_TEMPERATURE"] = str(T)
        mimic_wins = await collect_mimic_wins(num_truth_seeds=50, num_replicates=3)
        mimic_total = sum(mimic_wins.values())
        chi2, p, v = chi_sq_v(real_wins, mimic_wins)
        deltas = {a: mimic_wins[a] / mimic_total * 100 - real_pct[a] for a in LLM_LIST}
        results.append({"T": T, "chi2": chi2, "p": p, "v": v, "deltas": deltas})
        delta_str = "  ".join(f"{deltas[a]:>+8.1f}pp" for a in LLM_LIST)
        print(f"{T:>5.2f} {chi2:>8.2f} {p:>8.4f} {v:>8.4f}  {delta_str}")

    print()
    best = min(results, key=lambda r: r["v"])
    print(f"BEST: T = {best['T']:.2f}  ->  Cramer's V = {best['v']:.4f},  chi2 = {best['chi2']:.2f},  p = {best['p']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
