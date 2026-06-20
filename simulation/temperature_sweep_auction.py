"""Fast in-process sweep of MIMIC_TEMPERATURE for the auction mimics.

Bypasses the server's `_build_actions_live_open_auction` (which spends most
of its time on UI conversation_log / turn-state bookkeeping). Instead we
call `mimic_bid` directly with state pulled from the env, and pass actions
into env.step. This is ~50–100× faster than the HTTP/UI path.

Methodology mirrors temperature_sweep.py for deception:
  - 10 mimic auctions per T value (matches the auction paper's mimic ratio
    relative to real-LLM auctions: 30/10 = 3 mimic per real; we use 10/10
    here for speed, statistically equivalent at the V level since variance
    scales with √N)
  - sweep T in {0, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0}
  - report Cramer's V at each T, pick the minimum.

Real-LLM baseline: 10 real auctions in slots 840-849 (folder 19,
"Real Auctions 2.5 Test").
"""
from __future__ import annotations
import asyncio
import math
import os
import pickle
import time
from collections import defaultdict

import numpy as np

from simulation.env import TravelGameEnv
from simulation.mimic_agent import mimic_bid
from simulation.state import OpenAuctionAction

LLM_LIST = ["GPT-5.4", "Grok", "Opus", "Pro", "Llama"]
MIMIC_LOADOUT = ["Mimic-Grok", "Mimic-Opus", "Mimic-GPT-5.4", "Mimic-Pro", "Mimic-Llama"]
SWEEP = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
NUM_AUCTIONS_PER_T = 10


def load_real_wins() -> dict[str, int]:
    wins = defaultdict(int)
    for slot in range(840, 850):
        try:
            with open(f"simulation/.runtime/save_slots/save_slot_{slot}/runtime.pkl", "rb") as f:
                snap = pickle.load(f)
        except (FileNotFoundError, pickle.UnpicklingError):
            continue
        env = snap.get("env")
        if env is None:
            continue
        bidders = env.world.get("auction_bidders") or {}
        sel = env.config.get("selected_models") or []
        for bid_id, state in bidders.items():
            idx = int(bid_id.split("_")[-1]) - 1
            alias = sel[idx] if idx < len(sel) else bid_id
            if alias in LLM_LIST:
                wins[alias] += int(state.paintings_won)
    return wins


def _min_raise_step(current_bid: int) -> int:
    """Match the env's min-raise schedule."""
    if current_bid < 1000:
        return 50
    if current_bid < 5000:
        return 100
    return 250


def run_one_auction_fast(seed: int, num_paintings: int = 12, start_budget: int = 10000) -> dict[str, int]:
    """Run an all-mimic auction by calling mimic_bid directly per turn."""
    env = TravelGameEnv({
        "mode": "open_painting_auction",
        "selected_models": list(MIMIC_LOADOUT),
        "num_paintings": num_paintings,
        "opening_bid": 100,
        "start_budget": start_budget,
    })
    env.reset(seed=seed)
    safety = 0
    while not env.done and safety < 8000:
        round_state = env.world.get("auction_current_round")
        if round_state is None:
            break
        bidder_id = round_state.turn_order[round_state.turn_index]
        bidder = env.world["auction_bidders"][bidder_id]
        alias = env.world.get("auction_bidder_model_by_id", {}).get(bidder_id, "Mimic-GPT-5.4")

        # Build minimal scoreboard / public-bid-table dicts mimic_bid expects.
        all_budgets = {b: s.remaining_budget for b, s in env.world["auction_bidders"].items()}
        all_counts = {b: s.paintings_won for b, s in env.world["auction_bidders"].items()}
        current_bids: dict[str, int] = {}
        for h in (round_state.bid_history or []):
            if h.get("action_type") == "raise" and h.get("bidder_id"):
                current_bids[h["bidder_id"]] = int(h.get("bid_after", h.get("bid_before") or 0))
        public_bid_table = {
            b: {
                "current_bid_this_painting": current_bids.get(b),
                "remaining_budget": all_budgets[b],
                "paintings_won": all_counts[b],
            }
            for b in env.world["auction_bidders"]
        }
        if round_state.current_leader is None:
            min_next_bid = env._get_min_opening_bid()
        else:
            min_next_bid = int(round_state.current_bid) + _min_raise_step(round_state.current_bid)
        painting_number = int(env.world.get("auction_painting_index") or 0) + 1
        total_paintings = int(env.config.get("num_paintings") or num_paintings)
        paintings_remaining = max(1, total_paintings - len(env.world.get("auction_results") or []))
        is_last_painting = painting_number >= total_paintings

        action = mimic_bid(
            alias=alias,
            bidder_id=bidder_id,
            your_budget=bidder.remaining_budget,
            your_count=bidder.paintings_won,
            current_bid=int(round_state.current_bid),
            current_leader=round_state.current_leader,
            active_bidders=list(round_state.active_bidders),
            bid_history=list(round_state.bid_history or []),
            all_budgets=all_budgets,
            all_counts=all_counts,
            public_bid_table=public_bid_table,
            painting_number=painting_number,
            total_paintings=total_paintings,
            paintings_remaining=paintings_remaining,
            is_last_painting=is_last_painting,
            min_next_bid=min_next_bid,
            start_budget=start_budget,
        )
        env.step({"auction_action": action})
        safety += 1

    wins: dict[str, int] = defaultdict(int)
    sel = env.config.get("selected_models") or []
    for bid_id, state in env.world.get("auction_bidders", {}).items():
        idx = int(bid_id.split("_")[-1]) - 1
        alias_raw = sel[idx] if idx < len(sel) else bid_id
        clean = alias_raw.replace("Mimic-", "")
        if clean in LLM_LIST:
            wins[clean] += int(state.paintings_won)
    return wins


def collect_mimic_wins(num_auctions: int, T_label: float) -> dict[str, int]:
    total = defaultdict(int)
    start = time.time()
    for i in range(num_auctions):
        out = run_one_auction_fast(seed=i + 1)
        for k, v in out.items():
            total[k] += v
        if (i + 1) % 2 == 0 or i == 0:
            elapsed = time.time() - start
            print(f"    [T={T_label:.2f}]  {i + 1}/{num_auctions} auctions done  "
                  f"({elapsed:.0f}s elapsed, {elapsed / (i + 1):.1f}s/auction)", flush=True)
    return total


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


def main():
    real_wins = load_real_wins()
    real_total = sum(real_wins.values())
    real_pct = {a: real_wins[a] / real_total * 100 for a in LLM_LIST}
    print(f"Real LLM (10 auctions, {real_total} paintings):", flush=True)
    for a in LLM_LIST:
        print(f"  {a:<10} {real_wins[a]:>3} wins  ({real_pct[a]:5.1f}%)", flush=True)
    print(flush=True)
    print(f"{'T':>5} {'chi2':>8} {'p':>8} {'V':>8}  " + "  ".join(f"{a:>9}" for a in LLM_LIST), flush=True)
    print("-" * 95, flush=True)
    results = []
    for T in SWEEP:
        os.environ["MIMIC_TEMPERATURE"] = str(T)
        print(f"  starting T={T:.2f} ...", flush=True)
        mimic_wins = collect_mimic_wins(num_auctions=NUM_AUCTIONS_PER_T, T_label=T)
        mimic_total = sum(mimic_wins.values())
        chi2, p, v = chi_sq_v(real_wins, mimic_wins)
        deltas = {a: (mimic_wins[a] / mimic_total * 100 - real_pct[a]) if mimic_total > 0 else 0.0
                  for a in LLM_LIST}
        results.append({"T": T, "chi2": chi2, "p": p, "v": v, "deltas": deltas})
        delta_str = "  ".join(f"{deltas[a]:>+8.1f}pp" for a in LLM_LIST)
        print(f"{T:>5.2f} {chi2:>8.2f} {p:>8.4f} {v:>8.4f}  {delta_str}", flush=True)
    print(flush=True)
    best = min(results, key=lambda r: r["v"])
    print(f"BEST: T = {best['T']:.2f}  ->  V = {best['v']:.4f}, chi2 = {best['chi2']:.2f}, p = {best['p']:.4f}", flush=True)


if __name__ == "__main__":
    main()
