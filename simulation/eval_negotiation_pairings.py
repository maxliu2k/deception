"""In-process negotiation pairings eval (no LLM cost).

Runs the full back-and-forth between two agents (mimic-vs-mimic, math-vs-mimic,
math-vs-math) using the env directly (no HTTP server, no LLM). Reports per-pair
agreement rate, mean agreed price, mean buyer surplus, mean seller surplus.

Usage:
    python -m simulation.eval_negotiation_pairings --episodes 50 \\
        --pairings mimic_vs_mimic math_vs_mimic
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from itertools import product

from simulation.env import TravelGameEnv
from simulation.negotiation_mimic import negotiation_mimic_action
from simulation.negotiation_policies import NEGOTIATION_TIER_POLICIES
from simulation.state import NegotiationTurnAction


MIMICS = ["Mimic-GPT-5.4", "Mimic-Grok", "Mimic-Opus", "Mimic-Pro", "Mimic-Llama"]
MATH_TIERS = list(NEGOTIATION_TIER_POLICIES.keys())
MATH_RL_ALIAS = "Math-RL"

# Lazy-loaded RL policy (auto-detects v1 asymmetric vs v2-sym symmetric)
_RL_POLICY = None
_RL_SYMMETRIC = False

def _rl_policy():
    global _RL_POLICY, _RL_SYMMETRIC
    if _RL_POLICY is None:
        import torch
        from pathlib import Path
        # Prefer the newer symmetric checkpoint
        sym_path = Path(__file__).parent / "models" / "rl_negotiation" / "negotiation_rl_v2-sym.pt"
        old_path = Path(__file__).parent / "models" / "rl_negotiation" / "negotiation_rl_v1.pt"
        if sym_path.exists():
            from simulation.train_rl_negotiation_symmetric import SymmetricPolicy
            payload = torch.load(sym_path, map_location="cpu", weights_only=False)
            p = SymmetricPolicy()
            p.load_state_dict(payload["policy"])
            p.eval()
            _RL_POLICY = p; _RL_SYMMETRIC = True
        elif old_path.exists():
            from simulation.train_rl_negotiation import NegotiationPolicy
            payload = torch.load(old_path, map_location="cpu", weights_only=False)
            p = NegotiationPolicy()
            p.load_state_dict(payload["policy"])
            p.eval()
            _RL_POLICY = p; _RL_SYMMETRIC = False
        else:
            return None
    return _RL_POLICY


def _rl_action(role: str, own_value: float, own_target: float | None,
               turn_history: list[dict], standing_price: float | None,
               turn_index: int, message_limit: int) -> dict:
    import torch
    from simulation.negotiation_mimic import build_feature_vector, extraction_to_price
    pol = _rl_policy()
    if pol is None:
        return {"action": "continue", "proposed_price": int(own_value), "message_text": ""}
    x = build_feature_vector(role=role, own_private_value=own_value, own_target_price=own_target,
                             turn_history=turn_history, standing_price=standing_price,
                             turn_index=turn_index, message_limit=message_limit)
    xt = torch.tensor([x], dtype=torch.float32)
    with torch.no_grad():
        accept_logit, mean, _ = pol(xt)
        accept_prob = torch.sigmoid(accept_logit).item()
        out_val = float(mean.item())
    legal = standing_price is not None and (
        (role == "buyer" and standing_price <= own_value) or
        (role == "seller" and standing_price >= own_value))
    if accept_prob > 0.5 and legal:
        return {"action": "accept", "proposed_price": int(standing_price), "message_text": ""}
    price = extraction_to_price(out_val, own_value, role)
    return {"action": "continue", "proposed_price": price, "message_text": ""}


def _dispatch(alias: str, *, role: str, buyer, seller, turns, standing_price,
              turn_index: int, message_limit: int) -> dict:
    own_value = float(seller.baseline_value) if role == "seller" else float(buyer.budget)
    own_target = (float(getattr(seller, "target_price", 0) or seller.asking_price)
                  if role == "seller" else float(buyer.target_price))
    history = [{"speaker": t.speaker, "price": int(t.proposed_price)} for t in turns]
    if alias.startswith("Mimic-"):
        return negotiation_mimic_action(
            alias, role=role,
            own_private_value=own_value, own_target_price=own_target,
            turn_history=history, standing_price=standing_price,
            turn_index=turn_index, message_limit=message_limit,
        )
    if alias in NEGOTIATION_TIER_POLICIES:
        return NEGOTIATION_TIER_POLICIES[alias](
            role=role,
            own_private_value=own_value, own_target_price=own_target,
            turn_history=history, standing_price=standing_price,
            turn_index=turn_index, message_limit=message_limit,
        )
    if alias == MATH_RL_ALIAS:
        return _rl_action(role, own_value, own_target, history,
                          standing_price, turn_index, message_limit)
    raise ValueError(f"Unknown alias {alias!r}")


def _ensure_legal(role: str, price: int, buyer, seller, standing_price: float | None) -> int:
    """Clip a proposed price to the role's legal range."""
    if role == "buyer":
        return max(1, min(int(buyer.budget), int(price)))
    # seller
    return max(int(seller.baseline_value), int(price))


def run_one(buyer_alias: str, seller_alias: str, seed: int,
            message_limit: int = 10) -> dict:
    """Run one full negotiation between two agents and return outcome."""
    env = TravelGameEnv({
        "mode": "buyer_seller_negotiation",
        "selected_models": [buyer_alias, seller_alias, "GPT-5.4"],
        "seed": seed,
        "negotiation_message_limit": message_limit,
    })
    env.reset(seed=seed)
    buyer = env.world["buyer_true"]
    seller = env.world["seller_true"]

    turns: list[NegotiationTurnAction] = []
    standing_price = None
    agreed_price = None

    # Opener is determined by seed parity (matches server.py _negotiation_opener_for_seed).
    opener_role = "buyer" if (int(seed) % 2 == 1) else "seller"
    other_role = "seller" if opener_role == "buyer" else "buyer"

    def alias_for(role: str) -> str:
        return buyer_alias if role == "buyer" else seller_alias

    def legal_accept(role: str, p: float) -> bool:
        return (role == "buyer" and p <= buyer.budget) or (role == "seller" and p >= seller.baseline_value)

    # Opener acts (no standing price)
    op_res = _dispatch(alias_for(opener_role), role=opener_role, buyer=buyer, seller=seller,
                       turns=turns, standing_price=None, turn_index=0, message_limit=message_limit)
    opening_price = _ensure_legal(opener_role, op_res["proposed_price"], buyer, seller, None)
    turns.append(NegotiationTurnAction(speaker=opener_role, proposed_price=int(opening_price), message_text=""))
    standing_price = float(opening_price)

    # Alternating loop
    for turn_idx in range(1, message_limit):
        role = opener_role if (turn_idx % 2 == 0) else other_role
        res = _dispatch(alias_for(role), role=role, buyer=buyer, seller=seller,
                        turns=turns, standing_price=standing_price,
                        turn_index=turn_idx, message_limit=message_limit)
        if res["action"] == "accept" and legal_accept(role, standing_price):
            agreed_price = int(standing_price)
            turns.append(NegotiationTurnAction(speaker=role, proposed_price=int(standing_price), message_text=""))
            break
        new_price = _ensure_legal(role, res["proposed_price"], buyer, seller, standing_price)
        if role == "buyer":
            new_price = min(new_price, int(buyer.budget))
        else:
            new_price = max(new_price, int(seller.baseline_value))
        turns.append(NegotiationTurnAction(speaker=role, proposed_price=int(new_price), message_text=""))
        standing_price = float(new_price)

    out = {
        "buyer": buyer_alias, "seller": seller_alias, "seed": seed,
        "agreed": agreed_price is not None,
        "agreed_price": agreed_price,
        "n_turns": len(turns),
        "buyer_budget": int(buyer.budget),
        "seller_baseline": int(seller.baseline_value),
        "buyer_surplus": (int(buyer.budget) - agreed_price) if agreed_price is not None else 0,
        "seller_surplus": (agreed_price - int(seller.baseline_value)) if agreed_price is not None else 0,
    }
    return out


def summarize(results: list[dict]) -> dict:
    agreed = [r for r in results if r["agreed"]]
    summary = {
        "n_episodes": len(results),
        "n_agreed": len(agreed),
        "agreement_rate": len(agreed) / max(1, len(results)),
        "mean_turns": statistics.mean(r["n_turns"] for r in results),
        "mean_agreed_price": statistics.mean(r["agreed_price"] for r in agreed) if agreed else None,
        "mean_buyer_surplus": statistics.mean(r["buyer_surplus"] for r in agreed) if agreed else 0,
        "mean_seller_surplus": statistics.mean(r["seller_surplus"] for r in agreed) if agreed else 0,
        "mean_total_welfare": statistics.mean(r["buyer_surplus"] + r["seller_surplus"] for r in agreed) if agreed else 0,
    }
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=50,
                   help="Episodes per pairing.")
    p.add_argument("--message-limit", type=int, default=10)
    p.add_argument("--mode", choices=["mimic_vs_mimic","math_vs_mimic","math_vs_math","tier_summary"],
                   default="tier_summary")
    p.add_argument("--start-seed", type=int, default=1000)
    args = p.parse_args()

    if args.mode in ("mimic_vs_mimic", "tier_summary"):
        print("\n=== Mimic-vs-Mimic ===")
        all_results = []
        for b, s in product(MIMICS, MIMICS):
            results = [run_one(b, s, seed=args.start_seed + i, message_limit=args.message_limit)
                       for i in range(args.episodes // len(MIMICS) // len(MIMICS) or 1)]
            all_results.extend(results)
        summ = summarize(all_results)
        print(f"  n={summ['n_episodes']}, agree_rate={summ['agreement_rate']:.2%}, "
              f"mean_welfare={summ['mean_total_welfare']:.1f}, mean_turns={summ['mean_turns']:.1f}")

    if args.mode in ("math_vs_mimic", "tier_summary"):
        print("\n=== Math tier vs Mimic (each tier as both buyer and seller) ===")
        print(f"  {'tier':<22}{'role':<8}{'agree%':>9}{'mean_welfare':>14}{'tier_surplus':>14}{'turns':>8}")
        print("  " + "-" * 75)
        for tier in MATH_TIERS + [MATH_RL_ALIAS]:
            for tier_role in ("buyer", "seller"):
                results = []
                for mimic in MIMICS:
                    for i in range(args.episodes // len(MIMICS) or 1):
                        b = tier if tier_role == "buyer" else mimic
                        s = tier if tier_role == "seller" else mimic
                        results.append(run_one(b, s, seed=args.start_seed + len(results) * 7, message_limit=args.message_limit))
                summ = summarize(results)
                # Tier-side surplus
                tier_surplus = summ["mean_buyer_surplus"] if tier_role == "buyer" else summ["mean_seller_surplus"]
                print(f"  {tier:<22}{tier_role:<8}{summ['agreement_rate']*100:>8.1f}%{summ['mean_total_welfare']:>14.1f}{tier_surplus:>14.1f}{summ['mean_turns']:>8.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
