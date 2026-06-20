"""Grid-sweep the trust-sharpness exponent k for the deception game's score formula.

score_a = c_a * trust_a^k

For each candidate k, runs every policy in {T1, T2, T3, T4, Naive, Calibrated}
against the rotating 4-mimic field for 30 episodes and reports the mean win
share. The goal: pick the smallest k that produces a clear ordering
Calibrated > Naive (i.e., strategic backoff strictly outperforms max
aggression), so the game has teeth without becoming a Calibrated landslide.
"""
from __future__ import annotations

import asyncio
import math
import os
import statistics
import sys

# Force fresh reward.py evaluation each k by re-importing — but env var is
# read inside the function each call, so module reload isn't strictly needed.
os.environ.setdefault("DECEPTION_TRUST_EXPONENT", "1.0")

from simulation.env import TravelGameEnv
from simulation.server import _build_actions_live_deception_competition


MIMICS = ["Mimic-GPT-5.4", "Mimic-Grok", "Mimic-Opus", "Mimic-Pro", "Mimic-Llama"]
NUM_EPS = 50
NUM_ROUNDS = 12
POLICIES = ["Math-T1", "Math-T2", "Math-T3", "Math-T4", "Math-Naive", "Math-Smart", "Math-Calibrated"]
KS = [3.0, 3.5, 4.0, 4.5, 5.0]


async def run_one(test_alias, *, displaced_mimic_idx, seed):
    opps = [m for i, m in enumerate(MIMICS) if i != displaced_mimic_idx]
    loadout = [test_alias] + opps
    env = TravelGameEnv({
        "mode": "deception_competition",
        "selected_models": loadout,
        "num_rounds": NUM_ROUNDS,
        "truth_seed": seed,
    })
    env.reset(seed=seed)
    while not env.done:
        actions = await _build_actions_live_deception_competition(env, {"use_models": False})
        env.step(actions)
    return int(env.world["deception_episode"].agent_states[0].win_count)


async def eval_policy(alias):
    wins = []
    for ep in range(NUM_EPS):
        wins.append(await run_one(alias, displaced_mimic_idx=ep % len(MIMICS), seed=ep + 1))
    return [w / NUM_ROUNDS for w in wins]


async def main():
    print(f"Sweeping trust exponent k over {KS}", flush=True)
    print(f"Each policy: {NUM_EPS} episodes vs rotating mimic field ({NUM_ROUNDS} rounds each).", flush=True)
    print(flush=True)
    header = "  k    " + "".join(f"{p.replace('Math-',''):>14}" for p in POLICIES) + "   Naive-Smart"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for k in KS:
        os.environ["DECEPTION_TRUST_EXPONENT"] = str(k)
        means = {}
        for alias in POLICIES:
            shares = await eval_policy(alias)
            means[alias] = statistics.mean(shares) * 100
        diff = means["Math-Naive"] - means["Math-Smart"]
        row = f"  {k:<5.2f}" + "".join(f"{means[p]:>13.1f}%" for p in POLICIES) + f"   {diff:+.1f}pp"
        print(row, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
