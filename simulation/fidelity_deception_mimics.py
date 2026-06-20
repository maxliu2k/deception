"""Compute mimic fidelity metrics analogous to the auction's Table 1+2.

Single-head architecture: there's no lie classifier, so we report regressor
MAE on the raw claim values plus the per-LLM lie-frequency match (how often
does the mimic emit non-honest claims, compared to the LLM's actual rate).

Two-level evaluation:

  ACTION LEVEL (decision-by-decision):
    - claim MAE: mean absolute error of the predicted claim vs the LLM's
      claim, averaged over all (row, attr) pairs.
    - lie-rate match: |observed_lie_rate - mimic_lie_rate|, where "lie" is
      defined as claim != truth (after 2dp rounding) -- a magnitude proxy
      rather than a binary signal.

  OUTCOME LEVEL (per-LLM win distribution):
    - run 50 all-mimic episodes; compare per-LLM win share to real-LLM share
    - report Cramer's V effect size + chi-squared test of independence.
"""
from __future__ import annotations

import asyncio
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from simulation.env import TravelGameEnv
from simulation.mimic_agent import _load_deception_mimic, build_deception_mimic_input
from simulation.server import _build_actions_live_deception_competition


DATASET = "simulation/datasets/deception_dataset_v3.jsonl"
LLM_LIST = ["GPT-5.4", "Grok", "Opus", "Pro", "Llama"]
MIMIC_LIST = [f"Mimic-{a}" for a in LLM_LIST]


def load_rows() -> list[dict]:
    rows = []
    with open(DATASET, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def action_level_fidelity(rows: list[dict]) -> dict[str, dict]:
    """For each LLM, compute per-decision metrics on held-out validation rows.

    We use the same 10% per-alias slice the trainer used as the val split.
    The mimic is called in deterministic argmax mode (lie iff p_lie > 0.5)
    so the accuracy isn't blurred by the sampling temperature.
    """
    # Group rows by alias index (matches bidder_vocab from extraction).
    bidder_vocab = json.load(open("simulation/datasets/deception_dataset_v3_meta.json"))["bidder_vocab"]
    inv_vocab = {v: k for k, v in bidder_vocab.items()}
    by_alias: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_alias[inv_vocab[r["bidder_index"]]].append(r)

    out: dict[str, dict] = {}
    for alias in LLM_LIST:
        bidder_rows = by_alias[alias]
        # Match trainer's split deterministically: seed 0, shuffle, take first 10%.
        rng = random.Random(0)
        rng.shuffle(bidder_rows)
        n_val = max(1, len(bidder_rows) // 10)
        val = bidder_rows[:n_val]

        model = _load_deception_mimic(f"Mimic-{alias}")
        if model is None:
            out[alias] = {"error": "mimic not loaded"}
            continue

        # Predict on the val set in deterministic mode.
        X = np.array([row["x"] for row in val], dtype=np.float32)
        truth = X[:, :5]   # first 5 features of the input vector are the truth
        y_lied = np.array([row["y_lied"] for row in val], dtype=np.int32)
        y_claim = np.array([row["y_claim"] for row in val], dtype=np.float32)
        with torch.no_grad():
            claim_pred = model(torch.tensor(X)).cpu().numpy()
        # Quantize to 2dp to match the dispatch path's rounding.
        claim_pred_q = np.round(claim_pred, 2)
        # "Lie" definition matches the extractor: claim != truth after 2dp.
        mimic_lied = (claim_pred_q != np.round(truth, 2)).astype(np.int32)
        mae_all = float(np.abs(claim_pred - y_claim).mean())

        out[alias] = {
            "n_val_rows": len(val),
            "n_decisions": int(y_lied.size),
            "claim_mae": mae_all,
            "lie_rate_observed": float(y_lied.mean()),
            "lie_rate_predicted": float(mimic_lied.mean()),
        }
    return out


async def collect_mimic_win_shares(*, num_episodes: int = 50, num_rounds: int = 12) -> dict[str, int]:
    """Run num_episodes all-mimic episodes; return total wins per LLM alias."""
    wins: dict[str, int] = {a: 0 for a in LLM_LIST}
    for ep_i in range(num_episodes):
        env = TravelGameEnv({
            "mode": "deception_competition",
            "selected_models": MIMIC_LIST,
            "num_rounds": num_rounds,
            "truth_seed": ep_i + 1,
        })
        env.reset(seed=ep_i + 1)
        while not env.done:
            actions = await _build_actions_live_deception_competition(env, {"use_models": False})
            env.step(actions)
        for a in env.world["deception_episode"].agent_states:
            alias = a.alias.replace("Mimic-", "")
            wins[alias] += a.win_count
    return wins


def load_real_llm_wins() -> dict[str, int]:
    """Sum up per-LLM wins across the 50 v3 episodes we already have on disk."""
    import glob, re
    wins: dict[str, int] = {a: 0 for a in LLM_LIST}
    paths = glob.glob("simulation/.runtime/save_slots/save_slot_*/auction_exports/deception_episode/episode_log.json")
    n_eps = 0
    for p in paths:
        m = re.search(r"save_slot_(\d+)", p)
        if not m or int(m.group(1)) < 1203:
            continue
        try:
            ep = json.load(open(p))
        except Exception:
            continue
        if not ep.get("complete"):
            continue
        n_eps += 1
        for a in ep["agents"]:
            if a["alias"] in wins:
                wins[a["alias"]] += a["win_count"]
    print(f"  (loaded {n_eps} real-LLM episodes from disk)")
    return wins


def chi_sq_and_cramers_v(real: dict[str, int], mimic: dict[str, int]) -> tuple[float, float, float]:
    """Return (chi^2, p_value_approx, cramers_v) over the 2 x len(LLM_LIST) table."""
    # Two rows: real, mimic. Columns: LLMs.
    obs = np.array([[real[a] for a in LLM_LIST], [mimic[a] for a in LLM_LIST]], dtype=float)
    row_sums = obs.sum(axis=1, keepdims=True)
    col_sums = obs.sum(axis=0, keepdims=True)
    grand = obs.sum()
    if grand == 0:
        return 0.0, 1.0, 0.0
    expected = (row_sums @ col_sums) / grand
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.nansum((obs - expected) ** 2 / expected))
    # df = (rows - 1) * (cols - 1) = 1 * (k - 1) = k - 1
    df = len(LLM_LIST) - 1
    # Approx p-value via survival function of chi-squared. Use a small ad-hoc
    # CDF approximation to avoid scipy dependency.
    # Wilson-Hilferty: ((chi2/df)^(1/3) - (1 - 2/(9*df))) / sqrt(2/(9*df)) ~ N(0,1)
    if df > 0:
        z = ((chi2 / df) ** (1.0 / 3.0) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
        # Survival function of N(0,1) approx via erfc
        p = 0.5 * math.erfc(z / math.sqrt(2))
    else:
        p = 1.0
    # Cramer's V = sqrt(chi2 / (N * min(rows-1, cols-1)))
    min_dim = min(obs.shape[0] - 1, obs.shape[1] - 1)
    cramers_v = math.sqrt(chi2 / (grand * min_dim)) if grand > 0 and min_dim > 0 else 0.0
    return chi2, p, cramers_v


async def main() -> None:
    print("=" * 70)
    print("ACTION-LEVEL FIDELITY (held-out 10% of training set, argmax decisions)")
    print("=" * 70)
    rows = load_rows()
    action_metrics = action_level_fidelity(rows)
    print(f"\n{'LLM':<10}{'n_val':>7}{'decisions':>12}{'claim_MAE':>12}"
          f"{'lie_rate':>11}{'pred_rate':>11}{'lie_gap':>10}")
    for alias in LLM_LIST:
        m = action_metrics[alias]
        if "error" in m:
            print(f"{alias:<10} ERROR: {m['error']}")
            continue
        gap = abs(m["lie_rate_observed"] - m["lie_rate_predicted"]) * 100
        print(f"{alias:<10}{m['n_val_rows']:>7}{m['n_decisions']:>12}"
              f"{m['claim_mae']:>12.4f}"
              f"{m['lie_rate_observed']:>10.3f}{m['lie_rate_predicted']:>10.3f}"
              f"{gap:>9.1f}pp")

    print()
    print("=" * 70)
    print("OUTCOME-LEVEL FIDELITY (per-LLM win distribution, 50 episodes)")
    print("=" * 70)
    print("\nReal LLM wins (from 50 stored episodes):")
    real_wins = load_real_llm_wins()
    real_total = sum(real_wins.values())
    for a in LLM_LIST:
        print(f"  {a:<10}{real_wins[a]:>4} wins  ({real_wins[a]/real_total*100:5.1f}%)")

    print("\nMimic wins (50 fresh all-mimic episodes, seeds 1-50):")
    mimic_wins = await collect_mimic_win_shares(num_episodes=50)
    mimic_total = sum(mimic_wins.values())
    for a in LLM_LIST:
        print(f"  {a:<10}{mimic_wins[a]:>4} wins  ({mimic_wins[a]/mimic_total*100:5.1f}%)")

    chi2, p_approx, v = chi_sq_and_cramers_v(real_wins, mimic_wins)
    print()
    print(f"chi-squared = {chi2:.2f}  (df={len(LLM_LIST)-1})  p~{p_approx:.3f}")
    print(f"Cramer's V  = {v:.4f}")
    if v < 0.1:
        print("  -> negligible effect size (mimics reproduce LLM win distribution)")
    elif v < 0.3:
        print("  -> small-to-moderate effect (mimics diverge somewhat)")
    else:
        print("  -> meaningful effect (mimics differ substantially from LLMs)")

    # Per-LLM win share deltas (analog of auction Table 1).
    print()
    print(f"{'LLM':<10}{'real':>8}{'mimic':>8}{'delta_pp':>10}")
    for a in LLM_LIST:
        rs = real_wins[a] / real_total * 100
        ms = mimic_wins[a] / mimic_total * 100
        print(f"{a:<10}{rs:>7.1f}%{ms:>7.1f}%{ms - rs:>+9.1f}")


if __name__ == "__main__":
    asyncio.run(main())
