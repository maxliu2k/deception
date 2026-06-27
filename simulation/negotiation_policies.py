"""Math tier policies for the buyer/seller negotiation game.

Information ladder (each tier uses strictly more state than the prior):

Tier         | Info used                              | Strategy core
-------------|----------------------------------------|--------------------------------
Trivial-Open | (none)                                 | always counter at extreme; never accept
Truth-Anchor | own_private_value                      | concede to a fixed fraction of own value
Reactive     | + standing offer                       | bisect between standing and own value
Deadline     | + turn_index, opp offer history        | concede curve toward opp; accept at deadline
RL           | + everything via learned PPO           | learned

All formulas use LOG-SYMMETRIC extraction:
    Buyer:  price = own_value · exp(-e)
    Seller: price = own_value · exp(+e)
With e ∈ [0, ∞). e=0 means I'm at my reservation; larger e means more extraction.
This is truly multiplicative-symmetric across roles: swapping (role, v) flips
the sign in log-price space and leaves all per-tier behavior identical.

All policies return {"action": str, "proposed_price": int, "message_text": str}.
"""
from __future__ import annotations
import math


def _apply_ext(role: str, v: float, e: float) -> float:
    """Apply log-symmetric extraction to produce a target price."""
    if role == "buyer":
        return v * math.exp(-e)
    return v * math.exp(e)


def _ext_of(role: str, price: float, v: float) -> float:
    """Inverse: extraction implied by an (own_value, price) pair."""
    if v <= 0 or price <= 0:
        return 0.0
    if role == "buyer":
        return math.log(v / price)
    return math.log(price / v)


def _legal(role: str, price: float, v: float) -> bool:
    return (role == "buyer" and price <= v) or (role == "seller" and price >= v)


def trivial_open(
    *, role: str, own_private_value: float, own_target_price: float | None,
    turn_history: list[dict], standing_price: float | None,
    turn_index: int, message_limit: int,
) -> dict:
    """T1: extreme counter, never accept.

    e = 4.0  ⇒  Buyer offers v·exp(-4) ≈ 0.018·v; Seller asks v·exp(4) ≈ 54.6·v
    Both demand virtually all the surplus. Deal essentially impossible."""
    v = float(own_private_value)
    e = 4.0
    price = max(1, int(round(_apply_ext(role, v, e))))
    return {"action": "continue", "proposed_price": price, "message_text": ""}


def truth_anchored(
    *, role: str, own_private_value: float, own_target_price: float | None,
    turn_history: list[dict], standing_price: float | None,
    turn_index: int, message_limit: int,
) -> dict:
    """T2: state-free fixed log-extraction. Symmetric across buyer/seller.

        target_extraction = 0.30   (log-units)
        Buyer target  = v · exp(-0.30) ≈ 0.741·v
        Seller target = v · exp(+0.30) ≈ 1.350·v
    Accept when the opponent has met or beaten this extraction."""
    v = float(own_private_value)
    target_e = 0.30
    target_price = _apply_ext(role, v, target_e)
    if standing_price is not None:
        if (role == "buyer" and standing_price <= target_price) or \
           (role == "seller" and standing_price >= target_price):
            return {"action": "accept", "proposed_price": int(standing_price), "message_text": ""}
    return {"action": "continue", "proposed_price": int(round(target_price)), "message_text": ""}


def reactive(
    *, role: str, own_private_value: float, own_target_price: float | None,
    turn_history: list[dict], standing_price: float | None,
    turn_index: int, message_limit: int,
) -> dict:
    """T3: bisect between standing and target in log-price space.

        opening_e = 0.70   (start aggressive; buyer offers v·exp(-0.7)≈0.50v)
        accept_e  = 0.15   (accept if opp has given me e ≥ 0.15)
        target_e  = 0.50   (target ext 0.50 in counters)

        Buyer:  opening  = v · exp(-0.70) ≈ 0.497·v
                accept_T = v · exp(-0.15) ≈ 0.861·v
                target   = v · exp(-0.50) ≈ 0.607·v
        Seller: opening  = v · exp(+0.70) ≈ 2.014·v
                accept_T = v · exp(+0.15) ≈ 1.162·v
                target   = v · exp(+0.50) ≈ 1.649·v

    Counter is the geometric midpoint of standing and my target — log-space
    bisection rather than arithmetic, preserving multiplicative symmetry.
    """
    v = float(own_private_value)
    opening_e, accept_e, target_e = 0.70, 0.15, 0.50

    if standing_price is None:
        opening = int(round(_apply_ext(role, v, opening_e)))
        if role == "buyer":
            opening = max(1, opening)
        return {"action": "continue", "proposed_price": opening, "message_text": ""}

    accept_threshold = _apply_ext(role, v, accept_e)
    if (role == "buyer" and standing_price <= accept_threshold) or \
       (role == "seller" and standing_price >= accept_threshold):
        return {"action": "accept", "proposed_price": int(standing_price), "message_text": ""}

    # Geometric (log-space) bisection between standing and target.
    target_price = _apply_ext(role, v, target_e)
    new_price = math.sqrt(max(1.0, standing_price) * max(1.0, target_price))
    if role == "buyer":
        new_price = max(1, min(int(v), int(round(new_price))))
    else:
        new_price = max(int(v), int(round(new_price)))
    return {"action": "continue", "proposed_price": new_price, "message_text": ""}


def deadline_aware(
    *, role: str, own_private_value: float, own_target_price: float | None,
    turn_history: list[dict], standing_price: float | None,
    turn_index: int, message_limit: int,
) -> dict:
    """T4: T3 + concession curve toward deadline + opp-history awareness.

    Extraction parameters decrease linearly with `progress = turn_index/limit`:
        accept_e(p) = 0.30 − 0.25·p   (early: hold out for big surplus; late: take anything)
        target_e(p) = 0.70 − 0.40·p   (early: ask big; late: relent)
    Plus weighted-midpoint counter (weight on opp grows from 0.5 → 0.8 in log space).
    At the deadline turn, accept any legal offer.
    """
    v = float(own_private_value)
    progress = float(turn_index) / max(1, int(message_limit))
    turns_remaining = max(0, message_limit - len(turn_history))

    if standing_price is None:
        opening_e = 0.70  # same as T3 opener — aggressive anchor
        opening = int(round(_apply_ext(role, v, opening_e)))
        if role == "buyer":
            opening = max(1, opening)
        return {"action": "continue", "proposed_price": opening, "message_text": ""}

    # Deadline override: accept any legal offer
    if turns_remaining <= 1 and _legal(role, standing_price, v):
        return {"action": "accept", "proposed_price": int(standing_price), "message_text": ""}

    accept_e = max(0.0, 0.30 - 0.25 * progress)
    target_e = max(0.0, 0.70 - 0.40 * progress)
    accept_threshold = _apply_ext(role, v, accept_e)
    if (role == "buyer" and standing_price <= accept_threshold) or \
       (role == "seller" and standing_price >= accept_threshold):
        return {"action": "accept", "proposed_price": int(standing_price), "message_text": ""}

    # Geometric weighted midpoint: weight on opp's standing grows from 0.5 → 0.8.
    weight_opp = 0.5 + 0.3 * progress
    target_price = _apply_ext(role, v, target_e)
    # log-space weighted average: exp(w·log(standing) + (1-w)·log(target))
    log_combined = (weight_opp * math.log(max(1.0, standing_price))
                    + (1.0 - weight_opp) * math.log(max(1.0, target_price)))
    new_price = math.exp(log_combined)
    if role == "buyer":
        new_price = max(1, min(int(v), int(round(new_price))))
    else:
        new_price = max(int(v), int(round(new_price)))
    return {"action": "continue", "proposed_price": new_price, "message_text": ""}


NEGOTIATION_TIER_POLICIES = {
    "Math-Trivial-Open":  trivial_open,
    "Math-Truth-Anchored": truth_anchored,
    "Math-Reactive":      reactive,
    "Math-Deadline-Aware": deadline_aware,
    # Math-RL handled separately via PPO checkpoint
}
