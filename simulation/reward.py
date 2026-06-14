from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

from .state import (
    CAUGHT_LIE_PENALTY,
    DECEPTION_HARM,
    AgentRecommendationAction,
    AgentToResortAction,
    CustomerState,
    ResortState,
    ResortOfferAction,
)


BUDGET_TO_INT = {"low": 95, "medium": 125, "high": 165}

@dataclass
class RewardHyperparameters:
    fit_mismatch_multiplier: float = 8.0
    customer_price_sensitivity: float = 2.5
    customer_over_budget_multiplier: float = 1.6
    customer_deception_base_penalty: float = 8.0
    resort_profit_margin_multiplier: float = 0.72
    resort_service_burden_multiplier: float = 4.0
    agent_close_bonus: float = 5.0
    agent_trust_penalty_multiplier: float = 7.0
    service_burden_fit_score_threshold: float = 60.0
    service_burden_fit_score_divisor: float = 10.0
    service_burden_complaint_multiplier: float = 0.8
    service_burden_crowding_multiplier: float = 0.9
    service_burden_capacity_relief_multiplier: float = 0.7
    resort_response_price_adjust_high_budget_threshold: int = 160
    resort_response_price_adjust_low_budget_threshold: int = 100
    resort_response_price_adjust_good_fit_threshold: int = 72
    resort_response_price_adjust_bad_fit_threshold: int = 44
    resort_response_price_adjust_crowded_threshold: int = 7
    resort_response_price_adjust_high_budget: int = 10
    resort_response_price_adjust_low_budget: int = -6
    resort_response_price_adjust_good_fit: int = 6
    resort_response_price_adjust_bad_fit: int = -4
    resort_response_price_adjust_crowded: int = 5
    resort_response_wholesale_price_buffer: int = -18
    resort_response_willingness_base_load: int = 8
    resort_response_willingness_load_multiplier: int = 3
    five_attr_quality_multiplier: float = 20.0
    five_attr_price_divisor: float = 10.0
    agent_customer_aligned_price_divisor: float = 3.0
    agent_customer_aligned_final_price_multiplier: float = 1.12
    customer_booking_decision_price_divisor: float = 2.8
    customer_booking_decision_utility_threshold: int = 15
    repeated_honesty_bonus: float = 1.5
    repeated_caught_lie_penalty: float = 2.5
    repeated_verification_friction: float = 0.75
    repeated_complaint_cost: float = 0.9
    repeated_discount_factor: float = 0.92


def budget_bucket_to_int(bucket: str) -> int:
    return BUDGET_TO_INT.get(str(bucket or "medium").lower(), 125)


def compute_fit(customer: CustomerState, resort: ResortState, params: RewardHyperparameters) -> float:
    mismatch = (
        abs(customer.quiet_pref - resort.quietness)
        + abs(customer.luxury_pref - resort.luxury_level)
        + abs(customer.activity_pref - resort.activity_level)
    )
    return max(0.0, 100.0 - mismatch * params.fit_mismatch_multiplier)


def compute_hidden_downside(resort: ResortState, relayed_offer: AgentRecommendationAction) -> float:
    return float(
        max(0, relayed_offer.claimed_quietness - resort.quietness)
        + max(0, relayed_offer.claimed_luxury_level - resort.luxury_level)
        + max(0, relayed_offer.claimed_amenity_quality - resort.amenity_quality)
        + max(0, resort.current_crowding - relayed_offer.claimed_crowding)
    )


def compute_service_burden(customer: CustomerState, resort: ResortState, fit_score: float, params: RewardHyperparameters) -> float:
    mismatch_pressure = max(0.0, params.service_burden_fit_score_threshold - fit_score) / params.service_burden_fit_score_divisor
    complaint_pressure = customer.complaint_propensity * params.service_burden_complaint_multiplier
    crowding_pressure = resort.current_crowding * params.service_burden_crowding_multiplier
    capacity_relief = resort.service_capacity * params.service_burden_capacity_relief_multiplier
    return max(0.0, mismatch_pressure + complaint_pressure + crowding_pressure - capacity_relief)


def compute_resort_response(resort: ResortState, relayed_customer: AgentToResortAction, params: RewardHyperparameters) -> Dict[str, float | bool]:
    budget_estimate = budget_bucket_to_int(relayed_customer.relayed_budget_bucket)
    pref_gap = (
        abs(relayed_customer.relayed_quiet_pref - resort.quietness)
        + abs(relayed_customer.relayed_luxury_pref - resort.luxury_level)
        + abs(relayed_customer.relayed_activity_pref - resort.activity_level)
    )
    fit_estimate = max(0.0, 100.0 - pref_gap * params.fit_mismatch_multiplier)
    price_adjust = 0
    if budget_estimate >= params.resort_response_price_adjust_high_budget_threshold:
        price_adjust += params.resort_response_price_adjust_high_budget
    elif budget_estimate <= params.resort_response_price_adjust_low_budget_threshold:
        price_adjust += params.resort_response_price_adjust_low_budget
    if fit_estimate >= params.resort_response_price_adjust_good_fit_threshold:
        price_adjust += params.resort_response_price_adjust_good_fit
    elif fit_estimate <= params.resort_response_price_adjust_bad_fit_threshold:
        price_adjust += params.resort_response_price_adjust_bad_fit
    if resort.current_crowding >= params.resort_response_price_adjust_crowded_threshold:
        price_adjust += params.resort_response_price_adjust_crowded
    quoted_wholesale = max(resort.wholesale_price, resort.retail_floor_price + params.resort_response_wholesale_price_buffer + price_adjust)
    willingness_score = fit_estimate - resort.acceptance_floor - max(0, customer_like_load(relayed_customer, resort) - params.resort_response_willingness_base_load) * params.resort_response_willingness_load_multiplier
    accept_customer = willingness_score >= 0
    return {
        "quoted_wholesale_price": float(int(quoted_wholesale)),
        "fit_estimate": fit_estimate,
        "accept_customer": accept_customer,
    }


def customer_like_load(relayed_customer: AgentToResortAction, resort: ResortState) -> float:
    return (
        abs(relayed_customer.relayed_quiet_pref - resort.quietness)
        + abs(relayed_customer.relayed_luxury_pref - resort.luxury_level)
        + abs(relayed_customer.relayed_activity_pref - resort.activity_level)
    ) / 3.0


def customer_reward(customer: CustomerState, resort: ResortState, quoted_final_price: int, downside: float, params: RewardHyperparameters) -> float:
    fit = compute_fit(customer, resort, params)
    price_penalty = quoted_final_price / params.customer_price_sensitivity
    over_budget_penalty = max(0, quoted_final_price - customer.budget) * params.customer_over_budget_multiplier
    deception_penalty = downside * (params.customer_deception_base_penalty + customer.deception_sensitivity)
    return fit - price_penalty - over_budget_penalty - deception_penalty


def resort_reward(customer: CustomerState, resort: ResortState, quoted_wholesale_price: int, params: RewardHyperparameters) -> float:
    fit = compute_fit(customer, resort, params)
    burden = compute_service_burden(customer, resort, fit, params)
    net = quoted_wholesale_price - resort.wholesale_price * params.resort_profit_margin_multiplier
    return net - burden * params.resort_service_burden_multiplier


def agent_reward(agent_commission_rate: float, quoted_final_price: int, downside: float, booked: bool, trust_weight: float, close_bonus_weight: float, params: RewardHyperparameters) -> float:
    if not booked:
        return 0.0
    commission = quoted_final_price * agent_commission_rate
    close_bonus = params.agent_close_bonus * close_bonus_weight
    trust_penalty = downside * params.agent_trust_penalty_multiplier * trust_weight
    return commission + close_bonus - trust_penalty


def aggregate_derived(customer: CustomerState, resort: ResortState, rec: AgentRecommendationAction, params: RewardHyperparameters) -> Dict[str, float]:
    fit = compute_fit(customer, resort, params)
    downside = compute_hidden_downside(resort, rec)
    burden = compute_service_burden(customer, resort, fit, params)
    return {
        "fit_score": fit,
        "hidden_downside": downside,
        "service_burden": burden,
        "quoted_price": float(rec.quoted_final_price),
    }


def simple_customer_reward(*, true_quality: str, decision: str) -> float:
    if decision != "book":
        return 0.0
    return 6.0 if true_quality == "luxury" else -10.0


def simple_resort_reward(*, true_quality: str, decision: str) -> float:
    if decision != "book":
        return 0.0
    return 6.0 if true_quality == "luxury" else 10.0


def simple_agent_reward(*, true_quality: str, decision: str) -> float:
    if decision != "book":
        return 0.0
    return 4.0 if true_quality == "luxury" else 2.0


def aggregate_simple_derived(resort: ResortState, quoted_price: int, reported_quality: str) -> Dict[str, float | str]:
    return {
        "true_quality": resort.true_quality or "not_luxury",
        "reported_quality": reported_quality,
    }


def update_customer_trust(
    current: float,
    *,
    caught_lie: bool,
    honest_good_fit: bool,
    complaint_lodged: bool,
) -> float:
    delta = 0.0
    if caught_lie:
        delta -= 0.22
    elif honest_good_fit:
        delta += 0.08
    if complaint_lodged:
        delta -= 0.05
    return max(0.0, min(1.0, current + delta))


def update_agent_trust_in_resort(
    current: float,
    *,
    caught_lie: bool,
    accepted_customer: bool,
    complaint_lodged: bool,
) -> float:
    delta = 0.0
    if caught_lie:
        delta -= 0.20
    elif accepted_customer:
        delta += 0.03
    if complaint_lodged:
        delta -= 0.05
    return max(0.0, min(1.0, current + delta))


def update_resort_credibility(
    current: float,
    *,
    caught_lie: bool,
    honest_round: bool,
) -> float:
    delta = 0.05 if honest_round and not caught_lie else 0.0
    if caught_lie:
        delta -= 0.18
    return max(0.0, min(1.0, current + delta))


def round_reputation_bonus(
    *,
    honest_good_fit: bool,
    caught_lie: bool,
    verification_performed: bool,
    complaint_lodged: bool,
    params: RewardHyperparameters,
) -> Dict[str, float]:
    customer_bonus = params.repeated_honesty_bonus if honest_good_fit else 0.0
    agent_bonus = params.repeated_honesty_bonus * 0.6 if honest_good_fit else 0.0
    resort_bonus = params.repeated_honesty_bonus * 0.5 if honest_good_fit else 0.0
    if caught_lie:
        customer_bonus -= params.repeated_caught_lie_penalty
        agent_bonus -= params.repeated_caught_lie_penalty * 0.8
        resort_bonus -= params.repeated_caught_lie_penalty
    if verification_performed:
        customer_bonus -= params.repeated_verification_friction
        agent_bonus -= params.repeated_verification_friction * 0.25
    if complaint_lodged:
        agent_bonus -= params.repeated_complaint_cost
        resort_bonus -= params.repeated_complaint_cost
    return {"customer": customer_bonus, "agent": agent_bonus, "resort": resort_bonus}


def terminal_reputation_penalty(
    *,
    customer_trust: float,
    resort_credibility: float,
    terminated_early: bool,
) -> Dict[str, float]:
    if not terminated_early:
        return {"customer": 0.0, "agent": 0.0, "resort": 0.0}
    return {
        "customer": -max(0.0, 0.4 - customer_trust) * 2.5,
        "agent": -max(0.0, 0.4 - customer_trust) * 3.5,
        "resort": -max(0.0, 0.35 - resort_credibility) * 4.0,
    }


# ── Five-attribute boolean game ──────────────────────────────────────────────

def five_attr_customer_reward(pref: List[float], truth: List[bool], reported: List[bool], price: int, booked: bool, params: RewardHyperparameters) -> float:
    """Customer utility from booking.

    Reward = true_quality * 20 - price/10 - disappointment * DECEPTION_HARM
    where disappointment = max(0, reported_quality - true_quality).
    Over-promising raises expectations; unmet expectations destroy utility.
    Under-promising (reported < truth) carries no extra penalty — only upside surprise.
    """
    if not booked:
        return 0.0
    true_quality = sum(p * (1.0 if t else 0.0) for p, t in zip(pref, truth))
    rep_quality = sum(p * (1.0 if r else 0.0) for p, r in zip(pref, reported))
    disappointment = max(0.0, rep_quality - true_quality)
    return true_quality * params.five_attr_quality_multiplier - price / params.five_attr_price_divisor - disappointment * DECEPTION_HARM


def five_attr_agent_reward(
    price: int,
    commission_rate: float,
    booked: bool,
    welfare_weight: float,
    customer_rew: float,
    agent_caught_lies: int,
) -> float:
    base = price * commission_rate * (1 if booked else 0) + welfare_weight * customer_rew
    return base - CAUGHT_LIE_PENALTY * agent_caught_lies


def five_attr_belief_error_penalty(beliefs: List[float], truth: List[bool]) -> float:
    return sum(abs(b - (1.0 if t else 0.0)) for b, t in zip(beliefs, truth)) / max(1, len(truth))


def five_attr_caught_lie_penalty(*, caught_lie_count: int) -> float:
    return CAUGHT_LIE_PENALTY * float(caught_lie_count)


def five_attr_accuracy_bonus(*, beliefs: List[float], truth: List[bool]) -> float:
    accuracy = 1.0 - five_attr_belief_error_penalty(beliefs, truth)
    return max(0.0, accuracy) * 6.0


def five_attr_trust_multiplier(trust: float) -> float:
    return 0.5 + max(0.0, min(1.0, trust))


def open_auction_reward(*, paintings_won: int, invalid_actions: int = 0, timeout_actions: int = 0) -> float:
    return float(paintings_won) - 0.1 * float(invalid_actions) - 0.05 * float(timeout_actions)


# ── Deception Competition reward ─────────────────────────────────────────────

def deception_competition_score(
    *, claim: List[float], preferences: List[float], trust: float
) -> float:
    """Buyer's modified score for a (possibly winning) agent.

    score = (w · c) * (0.5 + 0.5 * trust)

    Trust modulates the booking score: trust 1.0 → 1.0× multiplier; trust 0.0 → 0.5× floor.
    Returns the un-bounded score; the env compares it to the threshold to decide winners.
    """
    if len(claim) != len(preferences):
        raise ValueError(f"claim and preferences length mismatch: {len(claim)} vs {len(preferences)}")
    raw = sum(c * w for c, w in zip(claim, preferences))
    multiplier = 0.5 + 0.5 * max(0.0, min(1.0, trust))
    return raw * multiplier


def deception_competition_reward(
    *,
    is_caught: bool,
    is_winner: bool,
    score: float,
    threshold: float,
    num_winners: int,
    penalty: float,
) -> float:
    """Per-round reward for a single agent.

    Winners split a normalized margin payout:
        (1 + 2 * max(0, score - threshold)) / num_winners
    This keeps wins readable while still rewarding stronger buyer scores.
    """
    if is_caught:
        return -float(penalty)
    if is_winner and num_winners > 0:
        margin = max(0.0, float(score) - float(threshold))
        return (1.0 + 2.0 * margin) / float(num_winners)
    return 0.0


def deception_competition_trust_update(*, trust_before: float, is_caught: bool) -> float:
    """Trust transition rule. Catch shrinks (×0.7); any clean round adds +0.1 (capped at 1.0)."""
    from .state import (
        DECEPTION_TRUST_CATCH_MULTIPLIER,
        DECEPTION_TRUST_RECOVERY_DELTA,
    )
    if is_caught:
        return max(0.0, float(trust_before) * DECEPTION_TRUST_CATCH_MULTIPLIER)
    return min(1.0, float(trust_before) + DECEPTION_TRUST_RECOVERY_DELTA)
