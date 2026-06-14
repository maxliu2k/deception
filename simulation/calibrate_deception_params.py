"""Sweep over threshold τ and report the deception math-tier ladder for each value.

Under Option R the only game knob is τ (no penalty, no tolerance). This script
runs each math tier (Math-T1..T4) against itself×5 across a τ grid and reports
mean per-agent reward + stable trust per tier.

Acceptance: the ladder is "well-formed" at τ if T2 < T1 < T3 < T4 with each
adjacent gap > min_gap (default 0.05). Among satisfying τ values, the script
picks the one with the largest minimum adjacent-gap.

Escape hatch: if no τ satisfies the criteria, exit with status 2 — signals
"the mechanic itself needs revisiting."

Output: JSON report at simulation/datasets/deception_calibration_report_v2.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from simulation.env import TravelGameEnv
from simulation.server import _build_actions_live_deception_competition


async def run_loadout(loadout: list[str], *, seed: int, threshold: float, num_rounds: int = 12) -> tuple[list[float], list[float]]:
    """Return (per-agent total rewards, per-agent final trust)."""
    env = TravelGameEnv({
        "mode": "deception_competition",
        "selected_models": loadout,
        "threshold": threshold,
        "num_rounds": num_rounds,
        "truth_seed": seed,
    })
    env.reset(seed=seed)
    while not env.done:
        actions = await _build_actions_live_deception_competition(env, {"use_models": False})
        env.step(actions)
    agents = env.world["deception_episode"].agent_states
    return [a.total_reward for a in agents], [a.trust for a in agents]


async def evaluate_tau(*, threshold: float, num_episodes: int, num_rounds: int) -> dict:
    tier_rewards: dict[str, float] = {}
    tier_trusts: dict[str, float] = {}
    for tier in ["Math-T1", "Math-T2", "Math-T3", "Math-T4"]:
        loadout = [tier] * 5
        per_agent_sum = 0.0
        per_agent_trust = 0.0
        n_agents = 0
        for ep in range(num_episodes):
            rewards, trusts = await run_loadout(loadout, seed=10_000 + ep, threshold=threshold, num_rounds=num_rounds)
            per_agent_sum += sum(rewards)
            per_agent_trust += sum(trusts)
            n_agents += len(rewards)
        tier_rewards[tier] = per_agent_sum / n_agents if n_agents else 0.0
        tier_trusts[tier] = per_agent_trust / n_agents if n_agents else 0.0
    return {"reward": tier_rewards, "trust": tier_trusts}


def check_ladder(tier_rewards: dict[str, float], *, min_gap: float = 0.05) -> tuple[bool, float, dict]:
    """Check the ladder T2 < T1 < T3 < T4.

    T2 (uniform inflate) is the broken control (steady ~0.8 trust). T1 (honest) is the
    safe baseline at trust 1.0. T3 (single-attr lie) and T4 (backloaded) sit above T1
    when the trust-degraded score boost outweighs honest play.
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
    p.add_argument("--out", default="simulation/datasets/deception_calibration_report_v2.json")
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--num-rounds", type=int, default=12)
    p.add_argument("--min-gap", type=float, default=0.05)
    p.add_argument("--taus", default="0.2,0.3,0.4,0.5,0.6")
    args = p.parse_args()

    taus = [float(x) for x in args.taus.split(",")]
    out_path = Path(args.out)
    print(f"Sweeping τ across {len(taus)} values, {args.num_episodes} episodes each.")

    report = {
        "version": "v2_optionR",
        "num_episodes_per_tau": args.num_episodes,
        "num_rounds": args.num_rounds,
        "min_adjacent_gap": args.min_gap,
        "taus": [],
        "best_tau": None,
    }
    valid_taus = []
    for tau in taus:
        result = await evaluate_tau(threshold=tau, num_episodes=args.num_episodes, num_rounds=args.num_rounds)
        tier_rewards = result["reward"]
        tier_trusts = result["trust"]
        ok, min_gap, gap_info = check_ladder(tier_rewards, min_gap=args.min_gap)
        entry = {
            "threshold": tau,
            "tier_mean_reward": {k: round(v, 4) for k, v in tier_rewards.items()},
            "tier_mean_final_trust": {k: round(v, 4) for k, v in tier_trusts.items()},
            **gap_info,
            "min_adjacent_gap": round(min_gap, 4),
            "monotone_and_separated": bool(ok),
        }
        print(f"  τ={tau:.2f}  T1={tier_rewards['Math-T1']:+.4f}  T2={tier_rewards['Math-T2']:+.4f}  "
              f"T3={tier_rewards['Math-T3']:+.4f}  T4={tier_rewards['Math-T4']:+.4f}  "
              f"min_gap={min_gap:+.4f}  trust(T2/T3/T4)={tier_trusts['Math-T2']:.2f}/{tier_trusts['Math-T3']:.2f}/{tier_trusts['Math-T4']:.2f}  "
              f"{'OK' if ok else '..'}")
        report["taus"].append(entry)
        if ok:
            valid_taus.append(entry)

    if valid_taus:
        valid_taus.sort(key=lambda c: c["min_adjacent_gap"], reverse=True)
        best = valid_taus[0]
        report["best_tau"] = best
        print(f"\nBEST τ: {best['threshold']:.2f}  min_gap={best['min_adjacent_gap']:.4f}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report -> {out_path}")
        return 0
    else:
        print(f"\nNO τ produces a monotone T2<T1<T3<T4 ladder with min_gap > {args.min_gap}.")
        print("Escape hatch fired: trust dynamics or T4 heuristic need revisiting.")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report -> {out_path}")
        return 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
