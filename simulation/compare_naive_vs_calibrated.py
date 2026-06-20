"""Matched-seed head-to-head: Naive (claim [1]*5 always) vs Calibrated (T3).

Both face the same fixed field of math-tier opponents, with truth schedules
matched seed-by-seed so per-seed differences come from the test policy only.
Reports each policy's mean per-episode win share with 95% t-CIs over the
episode as the independent replicate (matches the auction paper convention).

Verdict logic:
  - Calibrated CI strictly above Naive CI -> trust mechanic has teeth,
    over-aggression is genuinely punished, depth is real.
  - CIs overlap or Naive >= Calibrated -> trust is decoration; the game
    reduces to "claim the highest number" and the per-attr trust machinery
    isn't doing real work.
"""
from __future__ import annotations

import asyncio
import math
import statistics

from simulation.env import TravelGameEnv
from simulation.server import _build_actions_live_deception_competition


# Same opposition for both policies, ensuring the only thing that varies is
# the test seat (slot 0). T3 appears in the field; the Calibrated trial puts
# a second T3 in slot 0 — wins of slot 0 are still attributed to the test,
# but ties with slot 3 will split (so this is a conservative estimate for
# Calibrated, if anything biased against the calibrated-wins conclusion).
FIXED_FIELD = ["Math-T1", "Math-T2", "Math-T3", "Math-T4"]


async def run_one(test_alias: str, *, seed: int, num_rounds: int = 12) -> int:
    """Run one episode with `test_alias` in slot 0 vs the fixed field.

    Returns the test seat's win_count for the episode.
    """
    loadout = [test_alias] + FIXED_FIELD
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
    return int(agents[0].win_count)


def t_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float, float, float]:
    """Return (mean_pct, ci_lo_pct, ci_hi_pct, sd_pct) over a list of [0,1] shares."""
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    # t_{49, 0.975} ≈ 2.01; close enough for any n >= 30.
    t_crit = 2.01
    half = t_crit * sd / math.sqrt(n)
    return mean * 100, max(0, (mean - half)) * 100, min(100, (mean + half)) * 100, sd * 100


async def main() -> None:
    num_episodes = 50
    num_rounds = 12
    print(f"Running {num_episodes} matched-seed episodes per policy, {num_rounds} rounds each.")
    print(f"Field: {FIXED_FIELD}")
    print()

    naive_wins: list[int] = []
    calibrated_wins: list[int] = []
    for seed in range(1, num_episodes + 1):
        n_wins = await run_one("Math-Naive", seed=seed, num_rounds=num_rounds)
        c_wins = await run_one("Math-Calibrated", seed=seed, num_rounds=num_rounds)
        naive_wins.append(n_wins)
        calibrated_wins.append(c_wins)

    naive_shares = [w / num_rounds for w in naive_wins]
    calibrated_shares = [w / num_rounds for w in calibrated_wins]

    print(f"{'Policy':<22}{'mean':>8}{'95% t-CI':>20}{'sd_per_ep':>12}   (n={num_episodes} episodes)")
    for name, shares in [("Naive (claim=1)", naive_shares),
                          ("Calibrated (proper)", calibrated_shares)]:
        mean, lo, hi, sd = t_ci(shares)
        print(f"{name:<22}{mean:>7.1f}%  [{lo:>5.1f}, {hi:>5.1f}]   {sd:>9.1f}pp")

    # Pairwise (matched-seed): the per-seed difference
    diffs = [c - n for n, c in zip(naive_shares, calibrated_shares)]
    mean_diff = statistics.mean(diffs) * 100
    sd_diff = statistics.stdev(diffs) * 100 if len(diffs) > 1 else 0.0
    t_crit = 2.01
    half = t_crit * sd_diff / math.sqrt(len(diffs))
    print()
    print(f"Per-seed paired difference (Calibrated - Naive):")
    print(f"  mean = {mean_diff:+.1f}pp  95% t-CI = [{mean_diff-half:+.1f}, {mean_diff+half:+.1f}]")
    if mean_diff - half > 0:
        verdict = "Calibrated significantly beats Naive: TRUST MECHANIC HAS TEETH"
    elif mean_diff + half < 0:
        verdict = "Naive significantly beats Calibrated: TRUST IS DECORATION (failure mode)"
    else:
        verdict = "CIs overlap zero: comparable; trust may be marginal"
    print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    asyncio.run(main())
