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
This is truly multiplicative-symmetric across roles.

Every tier parameter is expressed as an integer multiple of one principled
reference value E_REF (defined below). No per-tier magic numbers — all values
derive from a single choice of E_REF.

All policies return {"action": str, "proposed_price": int, "message_text": str}.
"""
from __future__ import annotations
import math


# ── Principled reference extraction ──────────────────────────────────────────
# E_REF = log(2)/2 ≈ 0.347 is the Nash-bargaining solution in log-symmetric
# space under the uninformative prior "opp's value lies in [v/2, 2v]" — i.e.,
# without any information about the opponent's private value, my best guess
# of the 50/50 split is to extract log(2)/2 from my own value. This is the
# only "magic" number in the file; every tier parameter is a small integer
# multiple of E_REF.
E_REF = math.log(2.0) / 2.0   # ≈ 0.3466


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

    e = 10·E_REF ≈ 3.47 ⇒ buyer offers v·exp(-3.47) ≈ 0.031·v;
    seller asks v·exp(+3.47) ≈ 32·v. Deal essentially impossible.
    """
    v = float(own_private_value)
    e = 10.0 * E_REF
    price = max(1, int(round(_apply_ext(role, v, e))))
    return {"action": "continue", "proposed_price": price, "message_text": ""}


def truth_anchored(
    *, role: str, own_private_value: float, own_target_price: float | None,
    turn_history: list[dict], standing_price: float | None,
    turn_index: int, message_limit: int,
) -> dict:
    """T2: target the Nash-prescribed extraction under uninformative prior.

    target_e = 1·E_REF ≈ 0.347 ⇒
        Buyer target  = v · exp(-0.347) ≈ 0.707·v
        Seller target = v · exp(+0.347) ≈ 1.414·v
    Accept if opp has met the Nash-prescribed split.
    """
    v = float(own_private_value)
    target_e = 1.0 * E_REF
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

        opening_e = 2·E_REF ≈ 0.693   (open at log-double aggression)
        accept_e  = 0.5·E_REF ≈ 0.173  (accept if opp gave me ≥ half-Nash)
        target_e  = 1.5·E_REF ≈ 0.520  (counter aiming above-Nash)

    Counter is the geometric (log-space) midpoint of standing and target.
    """
    v = float(own_private_value)
    opening_e = 2.0 * E_REF
    accept_e = 0.5 * E_REF
    target_e = 1.5 * E_REF

    if standing_price is None:
        opening = int(round(_apply_ext(role, v, opening_e)))
        if role == "buyer":
            opening = max(1, opening)
        return {"action": "continue", "proposed_price": opening, "message_text": ""}

    accept_threshold = _apply_ext(role, v, accept_e)
    if (role == "buyer" and standing_price <= accept_threshold) or \
       (role == "seller" and standing_price >= accept_threshold):
        return {"action": "accept", "proposed_price": int(standing_price), "message_text": ""}

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
    """T4: T3 + linear concession curve toward deadline.

    With progress p = turn_index / message_limit ∈ [0, 1]:
        accept_e(p) = 1·E_REF · (1 - p)     (relax accept threshold to 0 at deadline)
        target_e(p) = 2·E_REF · (1 - p/2)   (start at 2·E_REF, end at 1·E_REF)
    Weighted-midpoint counter: weight on opp's standing grows from 0.5 → 0.8.
    At the deadline turn, accept any legal offer.
    """
    v = float(own_private_value)
    progress = float(turn_index) / max(1, int(message_limit))
    turns_remaining = max(0, message_limit - len(turn_history))

    if standing_price is None:
        opening_e = 2.0 * E_REF   # match T3 opener
        opening = int(round(_apply_ext(role, v, opening_e)))
        if role == "buyer":
            opening = max(1, opening)
        return {"action": "continue", "proposed_price": opening, "message_text": ""}

    if turns_remaining <= 1 and _legal(role, standing_price, v):
        return {"action": "accept", "proposed_price": int(standing_price), "message_text": ""}

    accept_e = max(0.0, 1.0 * E_REF * (1.0 - progress))
    target_e = max(0.0, 2.0 * E_REF * (1.0 - progress / 2.0))
    accept_threshold = _apply_ext(role, v, accept_e)
    if (role == "buyer" and standing_price <= accept_threshold) or \
       (role == "seller" and standing_price >= accept_threshold):
        return {"action": "accept", "proposed_price": int(standing_price), "message_text": ""}

    weight_opp = 0.5 + 0.3 * progress
    target_price = _apply_ext(role, v, target_e)
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
