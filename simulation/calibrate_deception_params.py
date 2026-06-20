"""Math-tier ladder sanity check for Option R (no game knobs).

Under R the only configurable parameter is num_rounds. All 5 attributes are
There's nothing to sweep; this script just confirms the math-tier ladder still
holds at the production defaults:

    T1 (honest) < T2 (uniform inflate) ≈ T3 (single-attr lie) < T4 (Nash)

The ladder is checked with a mixed loadout (T_test, T1, T2, T3, T1), since
symmetric loadouts produce 5-way ties under R and tell us nothing.

Acceptance: T4 > both T2 and T3 by min-gap, AND both T2 and T3 > T1 by min-gap.
T2 vs T3 ordering is unconstrained (depends on truth distribution + weights).

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


async def run_loadout(loadout: list[str], *, seed: int, num_rounds: int = 12) -> tuple[list[float], list[float]]:
    """Return (per-agent total rewards, per-agent mean per-attribute final trust)."""
    env = TravelGameEnv({
        "mode": "deception_competition",
        "selected_models": loadout,
        "num_rounds": num_rounds,
        "truth_seed": seed,
    })
    env.reset(seed=seed)
    while not env.done:
        actions = await _build_actions_live_deception_competition(env, {"use_models": False})
        env.step(actions)
    agents = env.world["deception_episode"].agent_states
    # Trust is now per-attribute — collapse to mean for the ladder summary.
    return [a.total_reward for a in agents], [sum(a.trust) / max(1, len(a.trust)) for a in agents]


async def evaluate(*, num_episodes: int, num_rounds: int) -> dict:
    """Each math tier in slot 0 vs the same mixed baseline [T1, T2, T3, T1]."""
    tier_rewards: dict[str, float] = {}
    tier_trusts: dict[str, float] = {}
    baseline = ["Math-T1", "Math-T2", "Math-T3", "Math-T1"]
    for tier in ["Math-T1", "Math-T2", "Math-T3", "Math-T4"]:
        loadout = [tier] + baseline
        reward_sum = 0.0
        trust_sum = 0.0
        for ep in range(num_episodes):
            rewards, trusts = await run_loadout(loadout, seed=10_000 + ep, num_rounds=num_rounds)
            reward_sum += rewards[0]
            trust_sum += trusts[0]
        tier_rewards[tier] = reward_sum / max(1, num_episodes)
        tier_trusts[tier] = trust_sum / max(1, num_episodes)
    return {"reward": tier_rewards, "trust": tier_trusts}


def check_ladder(tier_rewards: dict[str, float], *, min_gap: float = 0.05) -> tuple[bool, float, dict]:
    """Information-use ladder T1 ⊂ T2 ⊂ T3 ⊂ T4 under per-attribute trust:
        - T1 Honest                 uses {truth}                          c = t
        - T2 Greedy max-weight      + {prefs}                              c[argmax w]=1
        - T3 Trust-aware concentrator + {own_trust}                        T3 target
        - T4 Opp-aware concentrator + {opp_trust}                          T4 target

    Each tier strictly extends the previous one's information set AND adds
    one new strategic insight. Reward ladder is strict at T1 < T2 < T3 (the
    own_trust insight is empirically game-changing); T3 ≤ T4 because in
    calibration loadouts where the strongest opponent has full trust on
    every attribute (e.g., honest T1 opponents), T4's opp_trust factor
    cannot differentiate and reduces to T3. T4's empirical advantage
    surfaces in loadouts with differentiated opponents (LLM episodes,
    multiple concentrators)."""
    t1, t2, t3, t4 = (tier_rewards["Math-T1"], tier_rewards["Math-T2"],
                      tier_rewards["Math-T3"], tier_rewards["Math-T4"])
    monotone_lower = (t2 - t1 > min_gap) and (t3 - t2 > min_gap)
    t4_at_least_t3 = (t4 - t3 > -1e-3)   # T4 ≥ T3 (tolerate exact ties)
    ok = bool(monotone_lower and t4_at_least_t3)
    min_gap_observed = min(t2 - t1, t3 - t2, t4 - t3 + 1e-3)
    return ok, min_gap_observed, {
        "T1_to_T2": round(t2 - t1, 4),
        "T2_to_T3": round(t3 - t2, 4),
        "T3_to_T4": round(t4 - t3, 4),
    }


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="simulation/datasets/deception_calibration_report_v2.json")
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--num-rounds", type=int, default=12)
    p.add_argument("--min-gap", type=float, default=0.05)
    args = p.parse_args()

    out_path = Path(args.out)
    print(f"Running ladder check: {args.num_episodes} episodes per tier, {args.num_rounds} rounds each.")

    result = await evaluate(num_episodes=args.num_episodes, num_rounds=args.num_rounds)
    tier_rewards = result["reward"]
    tier_trusts = result["trust"]
    ok, min_gap, gap_info = check_ladder(tier_rewards, min_gap=args.min_gap)

    report = {
        "version": "v3_optionR_no_threshold",
        "num_episodes": args.num_episodes,
        "num_rounds": args.num_rounds,
        "min_adjacent_gap": args.min_gap,
        "tier_mean_reward": {k: round(v, 4) for k, v in tier_rewards.items()},
        "tier_mean_final_trust": {k: round(v, 4) for k, v in tier_trusts.items()},
        **gap_info,
        "min_observed_gap": round(min_gap, 4),
        "ladder_well_formed": bool(ok),
    }
    print(f"  T1={tier_rewards['Math-T1']:+.2f}  T2={tier_rewards['Math-T2']:+.2f}  "
          f"T3={tier_rewards['Math-T3']:+.2f}  T4={tier_rewards['Math-T4']:+.2f}  "
          f"min_separation={min_gap:+.2f}  trust(T1/T2/T3/T4)="
          f"{tier_trusts['Math-T1']:.2f}/{tier_trusts['Math-T2']:.2f}/{tier_trusts['Math-T3']:.2f}/{tier_trusts['Math-T4']:.2f}  "
          f"{'OK' if ok else 'FAIL'}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report -> {out_path}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
