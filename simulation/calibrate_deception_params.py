"""Find (threshold τ, penalty) cells where the deception math-tier ladder is monotone.

Per D7: the default params (τ=0.5, penalty=1.0) make honest play strictly
dominant — T1 wins, ladder collapses. This script runs the math tiers against
themselves across a (τ, penalty) grid and reports the cell with the cleanest
monotone separation T1 < T2 < T3 < T4, where T_n is `Math-T_n × 5` and the
score is mean per-agent total reward across N episodes.

Acceptance: a cell satisfies the ladder if T1 < T2 < T3 < T4 with each
adjacent gap > min_gap (default 0.05). Among satisfying cells, pick the one
with the largest minimum adjacent-gap (most robust ladder).

Escape hatch: if no cell satisfies the criteria, exit with status 2 and the
report records all cells for inspection — this signals "the mechanic itself
needs revisiting" rather than just "try a different cell."

Output: JSON report at simulation/datasets/deception_calibration_report_v1.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from simulation.env import TravelGameEnv
from simulation.server import _build_actions_live_deception_competition


async def run_loadout(loadout: list[str], *, seed: int, threshold: float, penalty: float, num_rounds: int = 12) -> list[float]:
    env = TravelGameEnv({
        "mode": "deception_competition",
        "selected_models": loadout,
        "threshold": threshold,
        "penalty": penalty,
        "num_rounds": num_rounds,
        "truth_seed": seed,
    })
    env.reset(seed=seed)
    while not env.done:
        actions = await _build_actions_live_deception_competition(env, {"use_models": False})
        env.step(actions)
    return [a.total_reward for a in env.world["deception_episode"].agent_states]


async def evaluate_cell(*, threshold: float, penalty: float, num_episodes: int, num_rounds: int) -> dict:
    tier_rewards: dict[str, float] = {}
    for tier in ["Math-T1", "Math-T2", "Math-T3", "Math-T4"]:
        loadout = [tier] * 5
        per_agent_sum = 0.0
        n_agents = 0
        for ep in range(num_episodes):
            rewards = await run_loadout(loadout, seed=10_000 + ep, threshold=threshold, penalty=penalty, num_rounds=num_rounds)
            per_agent_sum += sum(rewards)
            n_agents += len(rewards)
        tier_rewards[tier] = per_agent_sum / n_agents if n_agents else 0.0
    return tier_rewards


def check_ladder(tier_rewards: dict[str, float], *, min_gap: float = 0.05) -> tuple[bool, float, dict]:
    """Check the achievable ladder ordering T2 < T1 < T3 < T4.

    T2 (uniform inflate) is the 'broken control' — always caught, anchors the
    bottom. T1 (honest) is the safe baseline. T3 (single-attr lie) and T4
    (symmetric-Nash optimal subset) sit above T1 when lying is +EV under the
    chosen (τ, penalty) calibration.
    """
    t1, t2, t3, t4 = (tier_rewards["Math-T1"], tier_rewards["Math-T2"],
                      tier_rewards["Math-T3"], tier_rewards["Math-T4"])
    gaps = [t1 - t2, t3 - t1, t4 - t3]
    monotone = all(g > 0 for g in gaps)
    min_gap_observed = min(gaps) if gaps else 0.0
    ok = monotone and min_gap_observed > min_gap
    return ok, min_gap_observed, {"gaps_T1-T2_T3-T1_T4-T3": [round(g, 4) for g in gaps]}


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="simulation/datasets/deception_calibration_report_v1.json")
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--num-rounds", type=int, default=12)
    p.add_argument("--min-gap", type=float, default=0.05)
    p.add_argument("--taus", default="0.2,0.3,0.4,0.5")
    p.add_argument("--penalties", default="0.2,0.4,0.7,1.0")
    args = p.parse_args()

    taus = [float(x) for x in args.taus.split(",")]
    penalties = [float(x) for x in args.penalties.split(",")]
    out_path = Path(args.out)
    print(f"Sweeping τ × penalty = {len(taus)} × {len(penalties)} = {len(taus) * len(penalties)} cells, "
          f"{args.num_episodes} episodes per cell.")

    report = {
        "version": "v1",
        "num_episodes_per_cell": args.num_episodes,
        "num_rounds": args.num_rounds,
        "min_adjacent_gap": args.min_gap,
        "cells": [],
        "best_cell": None,
    }
    valid_cells = []
    for tau in taus:
        for penalty in penalties:
            tier_rewards = await evaluate_cell(threshold=tau, penalty=penalty,
                                               num_episodes=args.num_episodes, num_rounds=args.num_rounds)
            ok, min_gap, gap_info = check_ladder(tier_rewards, min_gap=args.min_gap)
            cell = {
                "threshold": tau,
                "penalty": penalty,
                "tier_mean_reward": {k: round(v, 4) for k, v in tier_rewards.items()},
                **gap_info,
                "min_adjacent_gap": round(min_gap, 4),
                "monotone_and_separated": bool(ok),
            }
            print(f"  τ={tau:.2f}  pen={penalty:.2f}  T1={tier_rewards['Math-T1']:+.4f}  "
                  f"T2={tier_rewards['Math-T2']:+.4f}  T3={tier_rewards['Math-T3']:+.4f}  "
                  f"T4={tier_rewards['Math-T4']:+.4f}  min_gap={min_gap:+.4f}  "
                  f"{'OK' if ok else '..'}")
            report["cells"].append(cell)
            if ok:
                valid_cells.append(cell)

    if valid_cells:
        valid_cells.sort(key=lambda c: c["min_adjacent_gap"], reverse=True)
        best = valid_cells[0]
        report["best_cell"] = best
        print(f"\nBEST CELL: τ={best['threshold']:.2f}  penalty={best['penalty']:.2f}  "
              f"min_gap={best['min_adjacent_gap']:.4f}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report -> {out_path}")
        return 0
    else:
        print(f"\nNO CELL produces a monotone T1<T2<T3<T4 ladder with min_gap > {args.min_gap}.")
        print("Per D7 escape hatch: revisit the trust scaling parameters before LLM collection.")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report -> {out_path}")
        return 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
