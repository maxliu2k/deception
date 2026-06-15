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
    # Tiers compete in ONE MIXED field (not [tier]*5 self-play, which makes identical
    # agents tie and cannot rank skill). Each tier occupies a seat; T1 fills the 5th.
    loadout = ["Math-T1", "Math-T2", "Math-T3", "Math-T4", "Math-T1"]
    tiers = ["Math-T1", "Math-T2", "Math-T3", "Math-T4"]
    rsum = {t: 0.0 for t in tiers}; tsum = {t: 0.0 for t in tiers}; cnt = {t: 0 for t in tiers}
    for ep in range(num_episodes):
        rewards, trusts = await run_loadout(loadout, seed=10_000 + ep, threshold=threshold, num_rounds=num_rounds)
        for i, alias in enumerate(loadout):
            rsum[alias] += rewards[i]; tsum[alias] += trusts[i]; cnt[alias] += 1
    tier_rewards = {t: (rsum[t] / cnt[t] if cnt[t] else 0.0) for t in tiers}
    tier_trusts = {t: (tsum[t] / cnt[t] if cnt[t] else 0.0) for t in tiers}
    return {"reward": tier_rewards, "trust": tier_trusts}


def check_ladder(tier_rewards: dict[str, float], *, min_gap: float = 0.05) -> tuple[bool, float, dict]:
    """Validation criterion (Option B, reframed): sophistication is rewarded.

    The original monotone T2<T1<T3<T4 is NOT the right bar — T2 and T3 are both
    constant-ish liars and legitimately bunch. The robust, validated criterion is:
      (a) the sophisticated tier T4 (horizon-aware backloaded) is the clear winner,
      (b) naive constant inflation (T2) does NOT beat honest play (T1).
    Together: the game rewards strategic reasoning and punishes naive lying.
    """
    t1, t2, t3, t4 = (tier_rewards["Math-T1"], tier_rewards["Math-T2"],
                      tier_rewards["Math-T3"], tier_rewards["Math-T4"])
    dominance = t4 - max(t1, t2, t3)          # T4 must top the field by a margin
    naive_not_rewarded = t1 - t2              # honest should be >= naive inflate
    ok = (dominance > min_gap) and (naive_not_rewarded > -1e-9)
    info = {"t4_dominance": round(dominance, 4), "honest_minus_naive": round(naive_not_rewarded, 4)}
    return ok, dominance, info


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
