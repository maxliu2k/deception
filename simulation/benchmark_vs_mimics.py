"""Phase 4 — strategy performance against a fixed mimic field (R design).

Each focal strategy plays seat 0 against a FIXED field of 4 behavioral mimics
(validated LLM surrogates). Because every focal faces the identical field, the
outcomes — including reward — are directly comparable. Runs IN-PROCESS via the
env (no server, no API spend); mimics are the trained 14-dim nets.

Focals: the math-tier ladder (T1 honest -> T4 backloaded) plus one LLM mimic, so
you can see where an LLM sits on the skill ladder.

Usage:
    python -m simulation.benchmark_vs_mimics --count 200
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np

from simulation.env import TravelGameEnv
from simulation.server import _build_actions_live_deception_competition

ALL_MIMICS = ["Mimic-Grok", "Mimic-Opus", "Mimic-GPT-5.4", "Mimic-Pro", "Mimic-Llama"]
FOCALS = ["Math-T1", "Math-T2", "Math-T3", "Math-T4",
          "Mimic-GPT-5.4", "Mimic-Opus", "Mimic-Pro", "Mimic-Grok", "Mimic-Llama"]
LABEL = {"Math-T1": "T1 honest", "Math-T2": "T2 naive-inflate",
         "Math-T3": "T3 conditional", "Math-T4": "T4 backloaded",
         "Mimic-GPT-5.4": "GPT (LLM)", "Mimic-Opus": "Opus (LLM)", "Mimic-Pro": "Pro (LLM)",
         "Mimic-Grok": "Grok (LLM)", "Mimic-Llama": "Llama (LLM)"}
OUT = Path(__file__).parent / "datasets" / "deception_vsmimic_report_R.json"


def field_for(focal: str) -> list[str]:
    """4 near-honest LLM mimics. For an LLM focal, leave it out (no self-duplication);
    math focals get the first four. All focals thus face '4 LLM mimics'."""
    return [m for m in ALL_MIMICS if m != focal][:4]


async def run_one(focal: str, seed: int, num_rounds: int = 12) -> dict:
    loadout = [focal] + field_for(focal)
    env = TravelGameEnv({
        "mode": "deception_competition", "selected_models": loadout,
        "threshold": 0.0, "num_rounds": num_rounds, "truth_seed": seed,
    })
    env.reset(seed=seed)
    while not env.done:
        actions = await _build_actions_live_deception_competition(env, {"use_models": False})
        env.step(actions)
    ep = env.world["deception_episode"]
    focal_agent = ep.agent_states[0]          # seat 0 is the focal
    aid = focal_agent.agent_id
    lie = vis = 0
    for rec in ep.rounds:
        c = (rec.claims_by_agent or {}).get(aid)
        if not c:
            continue
        for a in range(len(rec.truth)):
            vis += 1
            if round(float(c[a]), 2) > round(float(rec.truth[a]), 2):
                lie += 1
    return {
        "reward_per_round": focal_agent.total_reward / num_rounds,
        "win_rate": focal_agent.win_count / num_rounds,
        "final_trust": float(focal_agent.trust),
        "lie_rate": (lie / vis) if vis else 0.0,
    }


def boot_ci(vals: list[float], *, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    bs = a[idx].mean(axis=1)
    return float(a.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--count", type=int, default=200, help="episodes per focal (matched seeds across focals)")
    p.add_argument("--num-rounds", type=int, default=12)
    args = p.parse_args()

    print(f"Field: 4 LLM mimics (leave-one-out for LLM focals)  |  {args.count} episodes/focal, T={args.num_rounds}\n")
    per_focal: dict[str, dict] = {}
    for focal in FOCALS:
        recs = []
        for s in range(args.count):
            recs.append(await run_one(focal, seed=s, num_rounds=args.num_rounds))
        agg = {}
        for key in ("reward_per_round", "win_rate", "final_trust", "lie_rate"):
            agg[key] = [round(x, 4) for x in boot_ci([r[key] for r in recs], seed=hash(key) % 10_000)]
        per_focal[focal] = {"n": len(recs), **agg}
        print(f"  done {focal}")

    summary = {"field": "4 LLM mimics (leave-one-out for LLM focals)", "n_episodes": args.count, "per_focal": per_focal}
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'focal':18s} {'lie%':>6s} {'win':>16s} {'reward/rnd':>20s} {'trust':>7s}")
    for f in FOCALS:
        r = per_focal[f]
        lie = r["lie_rate"][0] * 100
        win = r["win_rate"]; rwd = r["reward_per_round"]
        print(f"{LABEL.get(f, f):18s} {lie:6.1f} "
              f"{win[0]:6.3f}[{win[1]:.3f},{win[2]:.3f}] "
              f"{rwd[0]:8.3f}[{rwd[1]:.3f},{rwd[2]:.3f}] {r['final_trust'][0]:7.3f}")
    print(f"\nReport -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
