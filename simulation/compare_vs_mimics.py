"""Test math policies against the rotating mimic ensemble.

Mirrors the auction paper's methodology: each baseline plays 50 episodes
where it occupies one seat and 4 of the 5 mimics fill the rest, with the
displaced mimic rotating so each is replaced an equal number of times.

Reports per-policy mean win share with 95% t-CIs at the episode-level
(the episode is the independent replicate, matching the auction paper).

Verdict on whether the trust mechanic is load-bearing:
  - Calibrated (T3) CI strictly above Naive CI -> trust has teeth.
  - CIs overlap or Naive >= Calibrated -> trust is decoration.
"""
from __future__ import annotations

import asyncio
import math
import statistics
import sys

from simulation.env import TravelGameEnv
from simulation.server import _build_actions_live_deception_competition


MIMICS = ["Mimic-GPT-5.4", "Mimic-Grok", "Mimic-Opus", "Mimic-Pro", "Mimic-Llama"]


async def run_one(test_alias: str, *, displaced_mimic_idx: int, seed: int, num_rounds: int = 12) -> int:
    """Test policy in slot 0; the 4 non-displaced mimics fill slots 1-4."""
    opps = [m for i, m in enumerate(MIMICS) if i != displaced_mimic_idx]
    loadout = [test_alias] + opps
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
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    t_crit = 2.01  # t_{49, 0.975}
    half = t_crit * sd / math.sqrt(n) if n > 1 else 0.0
    return mean * 100, max(0, (mean - half)) * 100, min(100, (mean + half)) * 100, sd * 100


async def evaluate(test_alias: str, *, num_episodes: int = 50, num_rounds: int = 12) -> list[int]:
    """Returns per-episode win counts (length num_episodes)."""
    wins = []
    for ep_i in range(num_episodes):
        displaced = ep_i % len(MIMICS)  # rotate which mimic is displaced
        seed = ep_i + 1
        w = await run_one(test_alias, displaced_mimic_idx=displaced, seed=seed, num_rounds=num_rounds)
        wins.append(w)
    return wins


async def main() -> None:
    num_episodes = 50
    num_rounds = 12
    policies = [
        "Math-Trivial-Max", "Math-Truth-Anchored", "Math-Self-Aware",
        "Math-Pack-Aware", "Math-RL",
        "Math-Naive", "Math-Smart", "Math-Calibrated",
    ]

    print(f"Each policy plays {num_episodes} episodes vs the rotating 4-mimic field "
          f"(each LLM displaced exactly {num_episodes // len(MIMICS)} times). "
          f"Per-episode win share = wins / {num_rounds}; episode is the independent replicate.")
    print(f"Mimic pool: {[m.replace('Mimic-', '') for m in MIMICS]}")
    print()
    print(f"  {'Policy':<14}{'mean_share':>12}{'95% t-CI':>18}{'sd_per_ep':>13}   (n=50 episodes)")

    results: dict[str, list[int]] = {}
    for pol in policies:
        wins = await evaluate(pol, num_episodes=num_episodes, num_rounds=num_rounds)
        results[pol] = wins
        shares = [w / num_rounds for w in wins]
        mean, lo, hi, sd = t_ci(shares)
        print(f"  {pol:<14}{mean:>11.1f}%  [{lo:>5.1f}, {hi:>5.1f}]   {sd:>10.1f}pp")

    # Paired comparison: Calibrated (proper) vs Naive over matched seeds
    print()
    naive_shares = [w / num_rounds for w in results["Math-Naive"]]
    calibrated_shares = [w / num_rounds for w in results["Math-Calibrated"]]
    diffs = [c - n for n, c in zip(naive_shares, calibrated_shares)]
    mean_diff = statistics.mean(diffs) * 100
    sd_diff = statistics.stdev(diffs) * 100 if len(diffs) > 1 else 0.0
    half = 2.01 * sd_diff / math.sqrt(len(diffs))
    print(f"Paired (Calibrated - Naive) per matched seed:")
    print(f"  mean diff = {mean_diff:+.1f}pp  95% t-CI = [{mean_diff - half:+.1f}, {mean_diff + half:+.1f}]")
    if mean_diff - half > 0:
        verdict = "Calibrated SIGNIFICANTLY beats Naive: TRUST MECHANIC HAS TEETH"
    elif mean_diff + half < 0:
        verdict = "Naive SIGNIFICANTLY beats Calibrated: TRUST IS DECORATION (failure mode)"
    else:
        verdict = "CIs overlap zero: COMPARABLE; trust likely marginal"
    print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    asyncio.run(main())
