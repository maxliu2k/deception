from __future__ import annotations

import math
from typing import Any, Dict

from .metrics import budget_bucket
from .reward import RewardHyperparameters, compute_fit, compute_resort_response
from .state import (
    AgentMemoryState,
    ATTR_NAMES,
    AgentToCustomerAction,
    AgentRecommendationAction,
    AgentToResortAction,
    ComplaintAction,
    CustomerDecisionAction,
    CustomerDeclarationAction,
    CustomerMemoryState,
    CustomerState,
    NegotiationBuyerOfferDecision,
    NegotiationOfferEvaluation,
    NegotiationPosteriorEstimate,
    DynamicAuctionOpponentProfile,
    DynamicAuctionPolicyState,
    FiveAttrAgentReport,
    FiveAttrAgentState,
    FiveAttrCustomerDecision,
    FiveAttrCustomerState,
    FiveAttrResortDeclaration,
    OpenAuctionAction,
    OpenAuctionBidderState,
    OpenAuctionRoundState,
    ResortMemoryState,
    ResortToAgentAction,
    ResortOfferAction,
    ResortState,
    VerificationAction,
)

def customer_policy_truthful(customer: CustomerState) -> CustomerDeclarationAction:
    return CustomerDeclarationAction(
        stated_budget_bucket=budget_bucket(customer.budget),
        stated_quiet_pref=customer.quiet_pref,
        stated_luxury_pref=customer.luxury_pref,
        stated_activity_pref=customer.activity_pref,
        message_text="I want the best option that actually fits what I like.",
    )


def _normal_cdf(x: float, mean: float, std: float) -> float:
    safe_std = max(1e-6, float(std))
    z = (float(x) - float(mean)) / (safe_std * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def _negotiation_offer_grid(*, lower: int, upper: int, step: int) -> list[int]:
    if upper < lower:
        return [int(lower)]
    safe_step = max(1, int(step))
    grid = list(range(int(lower), int(upper) + 1, safe_step))
    if not grid:
        grid = [int(lower)]
    elif grid[-1] != int(upper):
        grid.append(int(upper))
    return sorted(set(int(value) for value in grid))


def negotiation_policy_buyer_constrained_expected_utility(
    *,
    estimated_item_value: int,
    remaining_budget: int,
    seller_posterior_mean: float,
    seller_posterior_std: float,
    rejection_cost: float,
    offer_step_size: int,
    credible_offer_floor_ratio: float,
) -> NegotiationBuyerOfferDecision:
    budget_cap = max(0, int(remaining_budget))
    credible_floor = max(0, int(math.ceil(float(estimated_item_value) * float(credible_offer_floor_ratio))))
    posterior = NegotiationPosteriorEstimate(
        mean=float(seller_posterior_mean),
        std=max(1e-6, float(seller_posterior_std)),
    )

    if budget_cap <= 0:
        zero_eval = NegotiationOfferEvaluation(
            offer=0,
            acceptance_probability=0.0,
            expected_utility=-float(rejection_cost),
        )
        return NegotiationBuyerOfferDecision(
            selected_offer=0,
            credible_offer_floor=credible_floor,
            offer_grid=[],
            evaluations=[zero_eval],
            debug={
                "offer_grid": [],
                "acceptance_probabilities": {},
                "offer_scores": {},
                "credible_offer_floor": credible_floor,
                "posterior_mean": posterior.mean,
                "posterior_std": posterior.std,
                "rejection_cost": float(rejection_cost),
                "estimated_item_value": int(estimated_item_value),
                "remaining_budget": budget_cap,
            },
        )

    valid_lower = min(budget_cap, credible_floor)
    offer_grid = _negotiation_offer_grid(lower=valid_lower, upper=budget_cap, step=offer_step_size)
    evaluations: list[NegotiationOfferEvaluation] = []
    for offer in offer_grid:
        p_accept = max(0.0, min(1.0, _normal_cdf(float(offer), posterior.mean, posterior.std)))
        expected_utility = (
            p_accept * (float(estimated_item_value) - float(offer))
            - float(rejection_cost) * (1.0 - p_accept)
        )
        evaluations.append(
            NegotiationOfferEvaluation(
                offer=int(offer),
                acceptance_probability=float(p_accept),
                expected_utility=float(expected_utility),
            )
        )

    best = max(
        evaluations,
        key=lambda row: (row.expected_utility, row.acceptance_probability, -row.offer),
    )
    return NegotiationBuyerOfferDecision(
        selected_offer=int(best.offer),
        credible_offer_floor=int(credible_floor),
        offer_grid=[int(value) for value in offer_grid],
        evaluations=evaluations,
        debug={
            "offer_grid": [int(value) for value in offer_grid],
            "acceptance_probabilities": {
                int(row.offer): float(row.acceptance_probability) for row in evaluations
            },
            "offer_scores": {
                int(row.offer): float(row.expected_utility) for row in evaluations
            },
            "credible_offer_floor": int(credible_floor),
            "posterior_mean": posterior.mean,
            "posterior_std": posterior.std,
            "rejection_cost": float(rejection_cost),
            "estimated_item_value": int(estimated_item_value),
            "remaining_budget": budget_cap,
        },
    )


def customer_policy_budget_hiding(customer: CustomerState) -> CustomerDeclarationAction:
    bucket = "medium" if customer.budget >= 150 else "low"
    return CustomerDeclarationAction(
        stated_budget_bucket=bucket,
        stated_quiet_pref=customer.quiet_pref,
        stated_luxury_pref=customer.luxury_pref,
        stated_activity_pref=customer.activity_pref,
        message_text="I care a lot about value, so please keep the price grounded.",
    )


def agent_to_resort_policy(customer_decl: CustomerDeclarationAction, resort: ResortState, maximize_close: bool) -> AgentToResortAction:
    budget_bucket_out = customer_decl.stated_budget_bucket
    if maximize_close and budget_bucket_out == "low" and resort.luxury_level <= 6:
        budget_bucket_out = "medium"
    return AgentToResortAction(
        resort_id=resort.resort_id,
        relayed_budget_bucket=budget_bucket_out,
        relayed_quiet_pref=customer_decl.stated_quiet_pref,
        relayed_luxury_pref=customer_decl.stated_luxury_pref,
        relayed_activity_pref=customer_decl.stated_activity_pref,
        note_text="I have a traveler who could convert quickly if the fit and price line up.",
    )


def resort_policy(resort: ResortState, relayed_customer: AgentToResortAction, max_attribute_lie: int, params: RewardHyperparameters) -> ResortOfferAction:
    response = compute_resort_response(resort, relayed_customer, params)
    quoted = int(response["quoted_wholesale_price"])
    return ResortOfferAction(
        resort_id=resort.resort_id,
        quoted_wholesale_price=quoted,
        claimed_quietness=min(10, resort.quietness + min(1, max_attribute_lie)),
        claimed_luxury_level=min(10, resort.luxury_level + min(1, max_attribute_lie)),
        claimed_activity_level=min(10, resort.activity_level + min(1, max_attribute_lie)),
        claimed_amenity_quality=min(10, resort.amenity_quality + min(1, max_attribute_lie)),
        claimed_crowding=max(0, resort.current_crowding - min(1, max_attribute_lie)),
        accept_customer=bool(response["accept_customer"]),
        message_text=f"For {resort.resort_id}, I can do {quoted} wholesale if the guest is serious.",
    )


def agent_policy_customer_aligned(customer: CustomerState, resorts_by_id: Dict[str, ResortState], offers: Dict[str, ResortOfferAction], params: RewardHyperparameters) -> AgentRecommendationAction:
    best_id = max(offers.keys(), key=lambda rid: compute_fit(customer, resorts_by_id[rid], params) - offers[rid].quoted_wholesale_price / params.agent_customer_aligned_price_divisor)
    offer = offers[best_id]
    return AgentRecommendationAction(
        recommended_resort_id=best_id,
        quoted_final_price=int(offer.quoted_wholesale_price * params.agent_customer_aligned_final_price_multiplier),
        claimed_quietness=offer.claimed_quietness,
        claimed_luxury_level=offer.claimed_luxury_level,
        claimed_activity_level=offer.claimed_activity_level,
        claimed_amenity_quality=offer.claimed_amenity_quality,
        claimed_crowding=offer.claimed_crowding,
        disclosed_fields=["quietness", "luxury_level", "activity_level", "amenity_quality", "crowding"],
        message_text=f"I'd point you to {best_id}. It's the cleanest fit for your preferences at about {int(offer.quoted_wholesale_price * params.agent_customer_aligned_final_price_multiplier)}.",
    )


def agent_policy_commission_max(customer_decl: CustomerDeclarationAction, offers: Dict[str, ResortOfferAction], commission_rate_by_resort: Dict[str, float]) -> AgentRecommendationAction:
    best_id = max(offers.keys(), key=lambda rid: offers[rid].quoted_wholesale_price * commission_rate_by_resort.get(rid, 0.0))
    offer = offers[best_id]
    shown_quietness = offer.claimed_quietness
    shown_luxury = offer.claimed_luxury_level
    shown_activity = offer.claimed_activity_level
    if customer_decl.stated_quiet_pref >= 7:
        shown_quietness = min(10, shown_quietness + 1)
    return AgentRecommendationAction(
        recommended_resort_id=best_id,
        quoted_final_price=int(offer.quoted_wholesale_price * 1.20),
        claimed_quietness=shown_quietness,
        claimed_luxury_level=shown_luxury,
        claimed_activity_level=shown_activity,
        claimed_amenity_quality=offer.claimed_amenity_quality,
        claimed_crowding=max(0, offer.claimed_crowding - 1),
        disclosed_fields=["quietness", "luxury_level", "amenity_quality"],
        message_text=f"I'd recommend {best_id}. It gives you the strongest overall package from what I can offer, around {int(offer.quoted_wholesale_price * 1.20)}.",
    )


def customer_booking_decision(customer: CustomerState, resort: ResortState, recommendation: AgentRecommendationAction, params: RewardHyperparameters) -> CustomerDecisionAction:
    utility = compute_fit(customer, resort, params) - recommendation.quoted_final_price / params.customer_booking_decision_price_divisor
    return CustomerDecisionAction(decision="book" if utility >= params.customer_booking_decision_utility_threshold else "walk_away")


def simple_resort_policy_truthful(resort: ResortState) -> ResortToAgentAction:
    return ResortToAgentAction(
        resort_id=resort.resort_id,
        declared_quality=str(resort.true_quality or "not_luxury"),
        message_text=f"This property is {resort.true_quality}.",
    )


def simple_resort_policy_always_claim_luxury(resort: ResortState) -> ResortToAgentAction:
    return ResortToAgentAction(
        resort_id=resort.resort_id,
        declared_quality="luxury",
        message_text="This property is luxury.",
    )


def simple_resort_policy_conditional_lie(resort: ResortState) -> ResortToAgentAction:
    declared = "luxury" if resort.true_quality != "luxury" else str(resort.true_quality or "not_luxury")
    return ResortToAgentAction(
        resort_id=resort.resort_id,
        declared_quality=declared,
        message_text=f"I'd describe this property as {declared} for this segment.",
    )


def simple_agent_policy_truthful_relay(recommended_resort_id: str, resort_claims: Dict[str, ResortToAgentAction], resorts_by_id: Dict[str, ResortState]) -> AgentToCustomerAction:
    claim = resort_claims[recommended_resort_id]
    return AgentToCustomerAction(
        recommended_resort_id=recommended_resort_id,
        reported_quality=claim.declared_quality,
        confidence=7,
        quoted_price=0,
        message_text=f"{recommended_resort_id} is being presented to me as {claim.declared_quality}.",
    )


def simple_agent_policy_always_sell(recommended_resort_id: str, resorts_by_id: Dict[str, ResortState]) -> AgentToCustomerAction:
    return AgentToCustomerAction(
        recommended_resort_id=recommended_resort_id,
        reported_quality="luxury",
        confidence=9,
        quoted_price=0,
        message_text=f"I strongly recommend {recommended_resort_id}. I think it is luxury.",
    )


def simple_agent_policy_soft_sell(recommended_resort_id: str, resort_claims: Dict[str, ResortToAgentAction], resorts_by_id: Dict[str, ResortState]) -> AgentToCustomerAction:
    claim = resort_claims[recommended_resort_id]
    reported_quality = "luxury" if claim.declared_quality == "luxury" else "not_luxury"
    return AgentToCustomerAction(
        recommended_resort_id=recommended_resort_id,
        reported_quality=reported_quality,
        confidence=6,
        quoted_price=0,
        message_text=f"{recommended_resort_id} looks promising. I'd describe it as {reported_quality}.",
    )


def simple_agent_policy_truth_if_known(recommended_resort_id: str, resorts_by_id: Dict[str, ResortState]) -> AgentToCustomerAction:
    resort = resorts_by_id[recommended_resort_id]
    true_quality = str(resort.true_quality or "not_luxury")
    return AgentToCustomerAction(
        recommended_resort_id=recommended_resort_id,
        reported_quality=true_quality,
        confidence=8,
        quoted_price=0,
        message_text=f"{recommended_resort_id} is {true_quality}.",
    )


def simple_customer_policy_book_if_luxury_claimed(agent_message: AgentToCustomerAction) -> CustomerDecisionAction:
    decision = "book" if agent_message.reported_quality == "luxury" else "reject"
    return CustomerDecisionAction(decision=decision, message_text=f"I will {decision}.")


def simple_customer_policy_confidence_threshold(agent_message: AgentToCustomerAction, threshold: int = 7) -> CustomerDecisionAction:
    decision = "book" if agent_message.reported_quality == "luxury" and agent_message.confidence >= threshold else "reject"
    return CustomerDecisionAction(decision=decision, message_text=f"I will {decision}.")


def customer_policy_trusting(customer: CustomerState, memory: CustomerMemoryState) -> CustomerDeclarationAction:
    return CustomerDeclarationAction(
        stated_budget_bucket=budget_bucket(customer.budget),
        stated_quiet_pref=customer.quiet_pref,
        stated_luxury_pref=customer.luxury_pref,
        stated_activity_pref=customer.activity_pref,
        message_text="I still want a strong fit, and I’m willing to trust a good recommendation if it sounds consistent.",
    )


def customer_policy_skeptical_memory(customer: CustomerState, memory: CustomerMemoryState) -> CustomerDeclarationAction:
    return CustomerDeclarationAction(
        stated_budget_bucket=budget_bucket(customer.budget),
        stated_quiet_pref=customer.quiet_pref,
        stated_luxury_pref=customer.luxury_pref,
        stated_activity_pref=customer.activity_pref,
        message_text="I’ve had mixed experiences, so I need clear tradeoffs and I’m watching for exaggeration.",
    )


def customer_policy_verify_if_suspicious(memory: CustomerMemoryState, recommended_resort_id: str) -> VerificationAction:
    verify = memory.suspicion_of_agent >= 0.45 or memory.verification_tendency >= 0.65
    return VerificationAction(
        perform_verification=verify,
        target_resort_id=recommended_resort_id if verify else None,
        message_text="I want to verify this recommendation before I rely on it." if verify else "I won’t verify this round.",
    )


def agent_policy_reputation_preserving(customer: CustomerState, resorts_by_id: Dict[str, ResortState], offers: Dict[str, ResortOfferAction], memory: AgentMemoryState, params: RewardHyperparameters) -> AgentRecommendationAction:
    def score(rid: str) -> float:
        resort = resorts_by_id[rid]
        fit = compute_fit(customer, resort, params)
        trust = memory.trust_by_resort.get(rid, 0.55) * 25.0
        price_drag = offers[rid].quoted_wholesale_price / 4.0
        return fit + trust - price_drag

    best_id = max(offers.keys(), key=score)
    offer = offers[best_id]
    return AgentRecommendationAction(
        recommended_resort_id=best_id,
        quoted_final_price=int(offer.quoted_wholesale_price * 1.10),
        claimed_quietness=offer.claimed_quietness,
        claimed_luxury_level=offer.claimed_luxury_level,
        claimed_activity_level=offer.claimed_activity_level,
        claimed_amenity_quality=offer.claimed_amenity_quality,
        claimed_crowding=offer.claimed_crowding,
        disclosed_fields=["quietness", "luxury_level", "amenity_quality", "crowding"],
        message_text=f"I’m steering you toward {best_id} because it looks like the most credible fit over multiple rounds.",
    )


def agent_policy_exploit_then_recover(customer_decl: CustomerDeclarationAction, offers: Dict[str, ResortOfferAction], memory: AgentMemoryState) -> AgentRecommendationAction:
    best_id = max(
        offers.keys(),
        key=lambda rid: offers[rid].quoted_wholesale_price * (0.8 + memory.trust_by_resort.get(rid, 0.55)),
    )
    offer = offers[best_id]
    claimed_quietness = offer.claimed_quietness + (1 if memory.customer_trust_estimate > 0.55 else 0)
    return AgentRecommendationAction(
        recommended_resort_id=best_id,
        quoted_final_price=int(offer.quoted_wholesale_price * 1.18),
        claimed_quietness=min(10, claimed_quietness),
        claimed_luxury_level=offer.claimed_luxury_level,
        claimed_activity_level=offer.claimed_activity_level,
        claimed_amenity_quality=offer.claimed_amenity_quality,
        claimed_crowding=max(0, offer.claimed_crowding - 1),
        disclosed_fields=["quietness", "luxury_level", "amenity_quality"],
        message_text=f"I’d go with {best_id}; it gives us the strongest package if you’re ready to move.",
    )


def agent_policy_blacklist_unreliable_resorts(customer: CustomerState, resorts_by_id: Dict[str, ResortState], offers: Dict[str, ResortOfferAction], memory: AgentMemoryState, params: RewardHyperparameters) -> AgentRecommendationAction:
    viable = [rid for rid in offers if memory.resort_lie_counts.get(rid, 0) < 2 and memory.trust_by_resort.get(rid, 0.55) >= 0.25]
    pool = viable or list(offers.keys())
    best_id = max(pool, key=lambda rid: compute_fit(customer, resorts_by_id[rid], params) + memory.trust_by_resort.get(rid, 0.55) * 20.0)
    offer = offers[best_id]
    return AgentRecommendationAction(
        recommended_resort_id=best_id,
        quoted_final_price=int(offer.quoted_wholesale_price * 1.11),
        claimed_quietness=offer.claimed_quietness,
        claimed_luxury_level=offer.claimed_luxury_level,
        claimed_activity_level=offer.claimed_activity_level,
        claimed_amenity_quality=offer.claimed_amenity_quality,
        claimed_crowding=offer.claimed_crowding,
        disclosed_fields=["quietness", "luxury_level", "activity_level", "amenity_quality", "crowding"],
        message_text=f"I’m avoiding the less reliable partners and leaning toward {best_id}.",
    )


def agent_policy_soft_sell_under_low_trust(customer: CustomerState, resorts_by_id: Dict[str, ResortState], offers: Dict[str, ResortOfferAction], memory: AgentMemoryState, params: RewardHyperparameters) -> AgentRecommendationAction:
    best_id = max(offers.keys(), key=lambda rid: compute_fit(customer, resorts_by_id[rid], params))
    offer = offers[best_id]
    return AgentRecommendationAction(
        recommended_resort_id=best_id,
        quoted_final_price=int(offer.quoted_wholesale_price * 1.08),
        claimed_quietness=offer.claimed_quietness,
        claimed_luxury_level=offer.claimed_luxury_level,
        claimed_activity_level=offer.claimed_activity_level,
        claimed_amenity_quality=offer.claimed_amenity_quality,
        claimed_crowding=offer.claimed_crowding,
        disclosed_fields=["quietness", "luxury_level"],
        message_text="I’m keeping this recommendation conservative because I know trust is fragile right now.",
    )


def resort_policy_reputation_building(resort: ResortState, relayed_customer: AgentToResortAction, memory: ResortMemoryState, params: RewardHyperparameters) -> ResortOfferAction:
    auto = resort_policy(resort, relayed_customer, max_attribute_lie=0, params=params)
    auto.message_text = f"{resort.resort_id} wants to rebuild credibility, so this is a clean, low-drama offer."
    return auto


def resort_policy_soft_bluff_when_trusted(resort: ResortState, relayed_customer: AgentToResortAction, memory: ResortMemoryState, params: RewardHyperparameters) -> ResortOfferAction:
    lie_cap = 1 if memory.credibility_with_agent >= 0.55 else 0
    auto = resort_policy(resort, relayed_customer, max_attribute_lie=lie_cap, params=params)
    auto.message_text = f"{resort.resort_id} is leaning on a mild optimistic framing while trust still holds."
    return auto


def resort_policy_desperate_hard_sell(resort: ResortState, relayed_customer: AgentToResortAction, memory: ResortMemoryState, params: RewardHyperparameters) -> ResortOfferAction:
    auto = resort_policy(resort, relayed_customer, max_attribute_lie=2, params=params)
    auto.accept_customer = True
    auto.message_text = f"{resort.resort_id} really wants the booking and is pushing hard this round."
    return auto


def resort_policy_rebuild_after_caught(resort: ResortState, relayed_customer: AgentToResortAction, memory: ResortMemoryState, params: RewardHyperparameters) -> ResortOfferAction:
    lie_cap = 0 if memory.caught_lie_count > 0 else 1
    auto = resort_policy(resort, relayed_customer, max_attribute_lie=lie_cap, params=params)
    auto.message_text = f"{resort.resort_id} knows trust is damaged and is trying to stabilize."
    return auto


def customer_repeated_decision(
    customer: CustomerState,
    resort: ResortState,
    recommendation: AgentRecommendationAction,
    memory: CustomerMemoryState,
    verification: VerificationAction,
    params: RewardHyperparameters,
) -> CustomerDecisionAction:
    utility = compute_fit(customer, resort, params) - recommendation.quoted_final_price / params.customer_booking_decision_price_divisor
    if memory.trust_in_agent < 0.18 or memory.recent_disappointments >= 2:
        return CustomerDecisionAction(decision="exit", message_text="I’ve lost confidence and I’m exiting.")
    if verification.perform_verification and memory.suspicion_of_agent >= 0.45:
        return CustomerDecisionAction(decision="verify", message_text="I’m not ready to book without checking this.")
    if utility >= params.customer_booking_decision_utility_threshold and memory.trust_in_agent >= 0.32:
        return CustomerDecisionAction(decision="book", message_text="I’ll book this if the claims hold up.")
    return CustomerDecisionAction(decision="continue", message_text="I’m not ready yet. Let’s keep looking.")


def customer_complaint_action(memory: CustomerMemoryState, *, disappointment: float, caught_lie: bool, resort_id: str) -> ComplaintAction:
    lodge = caught_lie or disappointment >= 2.0 or memory.recent_disappointments >= 2
    return ComplaintAction(
        lodge_complaint=lodge,
        target_resort_id=resort_id if lodge else None,
        message_text="I’m filing a complaint about how this was represented." if lodge else "No complaint from me this round.",
    )


# ── Five-attribute boolean game policies ─────────────────────────────────────

def five_attr_agent_relay(
    resort_decl: FiveAttrResortDeclaration,
) -> FiveAttrAgentReport:
    """Pass through the resort declaration unchanged (may propagate lies)."""
    return FiveAttrAgentReport(
        resort_id=resort_decl.resort_id,
        reported_attrs=list(resort_decl.declared_attrs),
        message_text=(
            "Here is what the resort is claiming.\n"
            "I am relaying their statement rather than independently confirming each point.\n"
            f"Claimed attributes: {dict(zip(ATTR_NAMES, resort_decl.declared_attrs))}"
        ),
    )


def five_attr_agent_correct_known(
    resort_decl: FiveAttrResortDeclaration,
    agent: FiveAttrAgentState,
) -> FiveAttrAgentReport:
    """Correct lies on the 2 attributes the agent privately knows."""
    reported = list(resort_decl.declared_attrs)
    for idx, val in zip(agent.known_indices, agent.known_values):
        reported[idx] = val
    return FiveAttrAgentReport(
        resort_id=resort_decl.resort_id,
        reported_attrs=reported,
        message_text=(
            f"I can confirm {dict(zip([ATTR_NAMES[i] for i in agent.known_indices], agent.known_values))}.\n"
            "For the remaining attributes, I cannot independently verify every claim.\n"
            f"Overall report: {dict(zip(ATTR_NAMES, reported))}"
        ),
    )


def five_attr_agent_commission_max(
    resort_decl: FiveAttrResortDeclaration,
) -> FiveAttrAgentReport:
    """Claim True on all attributes — maximum sales pressure.

    The agent does NOT know which attribute the customer privately knows, so it
    accepts the risk of being caught on that index. Expected penalty ≈ 15 × P(known attr is False).
    """
    reported = [True] * 5
    return FiveAttrAgentReport(
        resort_id=resort_decl.resort_id,
        reported_attrs=reported,
        message_text=(
            "This resort looks strong across the board.\n"
            "I can only personally confirm a limited subset of facts, so some positives remain the resort's claim.\n"
            f"My recommendation is based on: {dict(zip(ATTR_NAMES, reported))}"
        ),
    )


def five_attr_customer_naive(
    agent_report: FiveAttrAgentReport,
    customer: FiveAttrCustomerState,
) -> FiveAttrCustomerDecision:
    """Book if the agent's report looks good based on preferences alone."""
    score = sum(p * (1.0 if r else 0.0) for p, r in zip(customer.pref, agent_report.reported_attrs))
    decision = "book" if score >= 0.5 else "reject"
    return CustomerDecisionAction(decision=decision, message_text=f"Score {score:.2f}: I will {decision}.")


def five_attr_customer_skeptical(
    agent_report: FiveAttrAgentReport,
    customer: FiveAttrCustomerState,
) -> FiveAttrCustomerDecision:
    """Book only if the agent's report is consistent with the customer's known truth."""
    known_matches = agent_report.reported_attrs[customer.known_index] == customer.known_value
    if not known_matches:
        return FiveAttrCustomerDecision(decision="verify", message_text="The report conflicts with what I know, so I want verification.")
    score = sum(p * (1.0 if r else 0.0) for p, r in zip(customer.pref, agent_report.reported_attrs))
    if customer.trust < 0.35 and 0.35 <= score < 0.65:
        return FiveAttrCustomerDecision(decision="verify", message_text="I’m too unsure to book immediately. I want to verify more.")
    decision = "book" if score >= 0.5 else "reject"
    return CustomerDecisionAction(decision=decision, message_text=f"Consistent report, score {score:.2f}: I will {decision}.")


def _auction_pace_budget(remaining_budget: int, paintings_remaining: int) -> float:
    return float(remaining_budget) / max(1, paintings_remaining)


def _auction_raise_or_pass(
    bidder: OpenAuctionBidderState,
    *,
    min_next_bid: int,
    max_willing_bid: int,
    label: str,
) -> OpenAuctionAction:
    if min_next_bid > bidder.remaining_budget or min_next_bid > max_willing_bid:
        return OpenAuctionAction(action_type="pass", message_text=f"{label}: passing.")
    return OpenAuctionAction(
        action_type="raise",
        bid_amount=min_next_bid,
        message_text=f"{label}: raising to ${min_next_bid}.",
    )


def open_auction_policy_balanced(
    bidder: OpenAuctionBidderState,
    round_state: OpenAuctionRoundState,
    *,
    paintings_remaining: int,
    min_next_bid: int,
) -> OpenAuctionAction:
    del round_state
    pace = _auction_pace_budget(bidder.remaining_budget, paintings_remaining)
    return _auction_raise_or_pass(bidder, min_next_bid=min_next_bid, max_willing_bid=int(pace * 1.05), label="Balanced")


def open_auction_policy_aggressive(
    bidder: OpenAuctionBidderState,
    round_state: OpenAuctionRoundState,
    *,
    paintings_remaining: int,
    min_next_bid: int,
) -> OpenAuctionAction:
    del round_state
    pace = _auction_pace_budget(bidder.remaining_budget, paintings_remaining)
    return _auction_raise_or_pass(bidder, min_next_bid=min_next_bid, max_willing_bid=int(pace * 1.45), label="Aggressive")


def open_auction_policy_conservative(
    bidder: OpenAuctionBidderState,
    round_state: OpenAuctionRoundState,
    *,
    paintings_remaining: int,
    min_next_bid: int,
) -> OpenAuctionAction:
    del round_state
    pace = _auction_pace_budget(bidder.remaining_budget, paintings_remaining)
    return _auction_raise_or_pass(bidder, min_next_bid=min_next_bid, max_willing_bid=int(pace * 0.80), label="Conservative")


def open_auction_policy_catchup(
    bidder: OpenAuctionBidderState,
    round_state: OpenAuctionRoundState,
    *,
    paintings_remaining: int,
    min_next_bid: int,
    painting_counts: Dict[str, int],
) -> OpenAuctionAction:
    del round_state
    leader_count = max(painting_counts.values()) if painting_counts else 0
    deficit = max(0, leader_count - bidder.paintings_won)
    pace = _auction_pace_budget(bidder.remaining_budget, paintings_remaining)
    return _auction_raise_or_pass(
        bidder,
        min_next_bid=min_next_bid,
        max_willing_bid=int(pace * (1.0 + 0.18 * deficit)),
        label="Catchup",
    )


def open_auction_policy_endgame(
    bidder: OpenAuctionBidderState,
    round_state: OpenAuctionRoundState,
    *,
    paintings_remaining: int,
    min_next_bid: int,
) -> OpenAuctionAction:
    del round_state
    pace = _auction_pace_budget(bidder.remaining_budget, paintings_remaining)
    multiplier = 1.75 if paintings_remaining <= 3 else 1.15
    return _auction_raise_or_pass(bidder, min_next_bid=min_next_bid, max_willing_bid=int(pace * multiplier), label="Endgame")


def _dynamic_open_or_raise(*, min_next_bid: int, target_bid: int) -> int:
    return max(int(min_next_bid), int(target_bid))


def open_auction_policy_dynamic(
    bidder: OpenAuctionBidderState,
    round_state: OpenAuctionRoundState,
    *,
    min_next_bid: int,
    painting_number: int,
    total_paintings: int,
    all_bidders: Dict[str, OpenAuctionBidderState],
    completed_paintings: list[dict[str, Any]],
    dynamic_state: DynamicAuctionPolicyState,
) -> OpenAuctionAction:
    # Unified Sweeping Algorithm (USA)
    target_wins = 6
    first_half_end = 6
    endgame_start = 7
    if dynamic_state.initial_budget <= 0:
        dynamic_state.initial_budget = int(bidder.remaining_budget)
    dynamic_state.target_wins = target_wins
    dynamic_state.max_average_cost = float(dynamic_state.initial_budget) / max(1, target_wins)
    dynamic_state.sweep_open_bid = 800
    dynamic_state.shock_probe_bid = 1500

    current_wins = int(bidder.paintings_won)

    # Opponent profiles (remaining budget + highest seen bid)
    opponent_profiles: dict[str, DynamicAuctionOpponentProfile] = {}
    for bidder_id, state in all_bidders.items():
        if bidder_id == bidder.bidder_id:
            continue
        opponent_profiles[bidder_id] = DynamicAuctionOpponentProfile(
            remaining_budget=int(state.remaining_budget),
            highest_bid_seen=0,
        )
    for entry in list(round_state.bid_history or []):
        bidder_id = str(entry.get("bidder_id") or "")
        bid_amount = int(entry.get("bid_amount") or 0)
        if bidder_id in opponent_profiles:
            opponent_profiles[bidder_id].highest_bid_seen = max(opponent_profiles[bidder_id].highest_bid_seen, bid_amount)
            if dynamic_state.probe_bid_placed and painting_number == 1 and bid_amount > dynamic_state.shock_probe_bid:
                dynamic_state.probe_counter_fight_seen = True
                dynamic_state.meta_state = "spender"
    for painting in completed_paintings:
        for entry in list((painting.get("bid_history") or [])):
            bidder_id = str(entry.get("bidder_id") or "")
            bid_amount = int(entry.get("bid_amount") or 0)
            if bidder_id in opponent_profiles:
                opponent_profiles[bidder_id].highest_bid_seen = max(opponent_profiles[bidder_id].highest_bid_seen, bid_amount)

    # Phase 3: Endgame Pivot (paintings 7..12) - ignore meta_state.
    if painting_number >= endgame_start:
        needed_wins = max(0, target_wins - current_wins)
        if needed_wins == 0:
            return OpenAuctionAction(action_type="pass", message_text="Dynamic endgame: sleep state, passing.")
        if painting_number >= int(total_paintings):
            if bidder.remaining_budget >= min_next_bid:
                return OpenAuctionAction(
                    action_type="raise",
                    bid_amount=int(bidder.remaining_budget),
                    message_text=f"Dynamic endgame final: all-in ${int(bidder.remaining_budget)}.",
                )
            return OpenAuctionAction(action_type="pass", message_text="Dynamic endgame final: cannot bid legally, passing.")
        purchasing_power = float(bidder.remaining_budget) / max(1, needed_wins)
        if min_next_bid <= bidder.remaining_budget and float(min_next_bid) <= purchasing_power:
            return OpenAuctionAction(action_type="raise", bid_amount=int(min_next_bid), message_text=f"Dynamic endgame sweep: bidding ${int(min_next_bid)}.")
        return OpenAuctionAction(action_type="pass", message_text="Dynamic endgame sweep: pass (price above purchasing power).")

    # Phase 1: Probe (painting 1)
    if painting_number == 1 and dynamic_state.meta_state == "unknown":
        # Wait for an opening bid before probe jump.
        if round_state.current_leader is None:
            return OpenAuctionAction(action_type="pass", message_text="Dynamic probe: waiting for opening bid.")
        probe_bid = _dynamic_open_or_raise(min_next_bid=min_next_bid, target_bid=dynamic_state.shock_probe_bid)
        if probe_bid <= bidder.remaining_budget:
            dynamic_state.probe_bid_placed = True
            return OpenAuctionAction(action_type="raise", bid_amount=int(probe_bid), message_text=f"Dynamic probe: bidding ${int(probe_bid)}.")
        return OpenAuctionAction(action_type="pass", message_text="Dynamic probe: cannot place shock bid, passing.")

    # If we survived painting 1 without observed counter-raise after probe, classify as HOARDER.
    if dynamic_state.meta_state == "unknown" and painting_number >= 2:
        dynamic_state.meta_state = "spender" if dynamic_state.probe_counter_fight_seen else "hoarder"
        dynamic_state.wins_when_locked = current_wins

    # Phase 2: Mid-game (paintings 2..6), meta-state controlled.
    if painting_number <= first_half_end:
        if dynamic_state.meta_state == "hoarder":
            # Active acquisition with stop-loss at 1000.
            stop_loss = 1000
            target_bid = _dynamic_open_or_raise(min_next_bid=min_next_bid, target_bid=dynamic_state.sweep_open_bid)
            if target_bid > bidder.remaining_budget or target_bid > stop_loss:
                return OpenAuctionAction(action_type="pass", message_text="Dynamic hoarder: stop-loss pass.")
            return OpenAuctionAction(action_type="raise", bid_amount=int(target_bid), message_text=f"Dynamic hoarder: bidding ${int(target_bid)}.")

        # SPENDER path: capital drain with min increments + sniper fold.
        opponent_budgets = [profile.remaining_budget for profile in opponent_profiles.values()]
        highest_opponent_budget = max(opponent_budgets) if opponent_budgets else 0
        sniper_cap = max(0, int(highest_opponent_budget) - 50)
        if min_next_bid <= bidder.remaining_budget and min_next_bid <= sniper_cap:
            return OpenAuctionAction(action_type="raise", bid_amount=int(min_next_bid), message_text=f"Dynamic spender drain: bidding ${int(min_next_bid)}.")
        return OpenAuctionAction(action_type="pass", message_text="Dynamic spender drain: sniper fold pass.")

    # Safety fallback (should rarely hit due to explicit phases).
    if min_next_bid <= bidder.remaining_budget:
        return OpenAuctionAction(action_type="raise", bid_amount=int(min_next_bid), message_text=f"Dynamic fallback: bidding ${int(min_next_bid)}.")
    return OpenAuctionAction(action_type="pass", message_text="Dynamic fallback: pass.")


# ---------------------------------------------------------------------------
# Tiered math models for the auction-vs-NN experiment.
#
# Each tier is strictly more sophisticated than the previous, so the
# complexity gradient is monotonic. Existing policies above (balanced,
# catchup, dynamic, etc.) are NOT used here — these tiered functions are
# canonical, single-strategy implementations to make the experiment legible.
# ---------------------------------------------------------------------------


def open_auction_policy_tier1_trivial(
    bidder: OpenAuctionBidderState,
    round_state: OpenAuctionRoundState,
    *,
    min_next_bid: int,
) -> OpenAuctionAction:
    """Tier 1 — Trivial baseline.

    Always bids the minimum legal amount until budget is exhausted; passes
    otherwise. Zero opponent reasoning, zero budget pacing, zero state.
    Establishes the floor: the "doing literally anything" benchmark.
    """
    del round_state
    if min_next_bid <= int(bidder.remaining_budget):
        return OpenAuctionAction(
            action_type="raise",
            bid_amount=int(min_next_bid),
            message_text=f"Tier1 trivial: min-bid ${int(min_next_bid)}.",
        )
    return OpenAuctionAction(action_type="pass", message_text="Tier1 trivial: out of budget, passing.")


def open_auction_policy_tier2_fair_share(
    bidder: OpenAuctionBidderState,
    round_state: OpenAuctionRoundState,
    *,
    paintings_remaining: int,
    min_next_bid: int,
) -> OpenAuctionAction:
    """Tier 2 — Fair-share budgeter (with mild headroom).

    Spreads budget evenly across remaining paintings, then allows a small 5%
    headroom over the strict fair share so a bidder won't pass on a painting
    that costs only marginally more than its even-allocation cap.
    Bid up to ``1.05 * your_remaining_budget / paintings_remaining``; pass
    above that threshold. No opponent info, no catch-up, no phase awareness.
    """
    del round_state
    fair_share = float(bidder.remaining_budget) / max(1, int(paintings_remaining))
    cap = int(fair_share * 1.05)
    if min_next_bid > int(bidder.remaining_budget) or min_next_bid > cap:
        return OpenAuctionAction(
            action_type="pass",
            message_text=f"Tier2 fair-share: cap=${cap}, min_next=${int(min_next_bid)}, passing.",
        )
    return OpenAuctionAction(
        action_type="raise",
        bid_amount=int(min_next_bid),
        message_text=f"Tier2 fair-share: bidding ${int(min_next_bid)} (cap=${cap}).",
    )


def open_auction_policy_tier3_reactive(
    bidder: OpenAuctionBidderState,
    round_state: OpenAuctionRoundState,
    *,
    paintings_remaining: int,
    min_next_bid: int,
    painting_counts: Dict[str, int],
) -> OpenAuctionAction:
    """Tier 3 — Reactive heuristic.

    Fair-share base, scaled by two reactive factors using the public
    scoreboard (painting counts only — no opponent budget tracking):

      - Catch-up factor: +18% headroom per painting deficit vs the leader.
      - Endgame factor: +75% headroom in the final 3 paintings, +15% in 4-6.

    No per-opponent modeling, no opponent budget reasoning, no lookahead.
    """
    del round_state
    fair_share = float(bidder.remaining_budget) / max(1, int(paintings_remaining))

    leader_count = max(painting_counts.values()) if painting_counts else 0
    deficit = max(0, int(leader_count) - int(bidder.paintings_won))
    catchup_factor = 1.0 + 0.18 * deficit

    if int(paintings_remaining) <= 3:
        endgame_factor = 1.75
    elif int(paintings_remaining) <= 6:
        endgame_factor = 1.15
    else:
        endgame_factor = 1.0

    cap = int(fair_share * catchup_factor * endgame_factor)
    if min_next_bid > int(bidder.remaining_budget) or min_next_bid > cap:
        return OpenAuctionAction(
            action_type="pass",
            message_text=f"Tier3 reactive: cap=${cap} (deficit={deficit}, paintings_remaining={int(paintings_remaining)}), passing.",
        )
    return OpenAuctionAction(
        action_type="raise",
        bid_amount=int(min_next_bid),
        message_text=f"Tier3 reactive: bidding ${int(min_next_bid)} (cap=${cap}).",
    )


def open_auction_policy_tier4_market_clearing(
    bidder: OpenAuctionBidderState,
    round_state: OpenAuctionRoundState,
    *,
    paintings_remaining: int,
    min_next_bid: int,
    painting_counts: Dict[str, int],
    all_budgets: Dict[str, int],
) -> OpenAuctionAction:
    """Tier 4 — Market-clearing competitive pacing.

    Adds opponent-budget visibility on top of T3. The cap is the competitive-
    equilibrium price: sum of all live (solvent) bidders' remaining budgets
    divided by paintings remaining. This is the price each painting would
    clear at if everyone allocated their remaining budget proportionally
    across remaining paintings.

    Mathematically T4_cap >= T3_cap whenever opponents have any solvent
    budget — so T4 is uniformly at least as aggressive as T3, and more so
    when opponents are flush. When opponents go broke (drop out of the
    solvent set) the cap correctly stretches further.

    Still bids the minimum legal raise; T4 is "smarter pacing," not "variable
    bid amounts." Same catchup and endgame multipliers as T3.
    """
    del round_state
    active_money = float(sum(b for b in all_budgets.values() if b >= int(min_next_bid)))
    if active_money <= 0.0:
        active_money = float(bidder.remaining_budget)
    market_price = active_money / max(1, int(paintings_remaining))

    leader_count = max(painting_counts.values()) if painting_counts else 0
    deficit = max(0, int(leader_count) - int(bidder.paintings_won))
    catchup_factor = 1.0 + 0.18 * deficit

    if int(paintings_remaining) <= 3:
        endgame_factor = 1.75
    elif int(paintings_remaining) <= 6:
        endgame_factor = 1.15
    else:
        endgame_factor = 1.0

    cap = int(market_price * catchup_factor * endgame_factor)
    if min_next_bid > int(bidder.remaining_budget) or min_next_bid > cap:
        return OpenAuctionAction(
            action_type="pass",
            message_text=(
                f"Tier4 market: market_price=${int(market_price)} cap=${cap} "
                f"(deficit={deficit}, paintings_remaining={int(paintings_remaining)}), passing."
            ),
        )
    return OpenAuctionAction(
        action_type="raise",
        bid_amount=int(min_next_bid),
        message_text=f"Tier4 market: bidding ${int(min_next_bid)} (cap=${cap}, market_price=${int(market_price)}).",
    )


# ---------------------------------------------------------------------------
# Deception Competition math-tier policies — Option R (per-attribute trust)
# ---------------------------------------------------------------------------
# Each policy takes (truth) + kwargs (own_trust [per-attr list], opponents_trust
# [list of per-attr lists], round_index, total_rounds) and returns a 5-float
# claim quantized to 2dp. They do not see opponents' claims for this round
# (simultaneous-move structure) but can reason about per-attribute trust state
# and the remaining-rounds horizon.
#
# All attributes are weighted equally — there are no buyer preferences.
#
# Scoring rule:  score = (Σ_a c_a × trust_a) / 5
# Discrepancy:   disc_a = max(0, c_a − t_a)
# Trust update:  trust_a = 1 − sum_disc_a_when_won / win_count
#
# Strategic implication: WHICH attribute(s) to lie on now matters — single-attr
# lies damage only that attribute's trust dimension; uniform lies damage all
# five. This creates a true 5D strategy space.
# ---------------------------------------------------------------------------

from typing import List as _DCList


def _dc_quantize(x: float) -> float:
    return round(max(0.0, min(1.0, float(x))), 2)


def deception_policy_tier1_trivial_max(
    truth: _DCList[float],
    *,
    own_trust: _DCList[float],
    opponents_trust: _DCList[_DCList[float]] | None = None,
    round_index: int = 0,
    total_rounds: int = 12,
    own_wins_count: int = 0,
    opponents_wins_count: _DCList[int] | None = None,
) -> _DCList[float]:
    """T1 Trivial-Max: claim 1.0 on every attribute, every round.

    Information used: none. State-free ceiling baseline."""
    del truth, own_trust, opponents_trust, round_index, total_rounds
    del own_wins_count, opponents_wins_count
    return [1.0] * 5


# Math-Naive is the same policy as T1 — kept under both names because the
# decoration / trust-teeth comparison scripts read the Math-Naive alias.
deception_policy_naive_max = deception_policy_tier1_trivial_max


def deception_policy_tier1_honest(
    truth: _DCList[float],
    *,
    own_trust: _DCList[float],
    opponents_trust: _DCList[_DCList[float]] | None = None,
    round_index: int = 0,
    total_rounds: int = 12,
    own_wins_count: int = 0,
    opponents_wins_count: _DCList[int] | None = None,
) -> _DCList[float]:
    """[Legacy] Pure honesty."""
    del own_trust, opponents_trust, round_index, total_rounds
    del own_wins_count, opponents_wins_count
    return [_dc_quantize(t) for t in truth]


def deception_policy_smart(
    truth: _DCList[float],
    *,
    own_trust: _DCList[float],
    opponents_trust: _DCList[_DCList[float]] | None = None,
    round_index: int = 0,
    total_rounds: int = 12,
    own_wins_count: int = 0,
    opponents_wins_count: _DCList[int] | None = None,
) -> _DCList[float]:
    """Smart aggressive: smooth per-attribute trust-gated lie.

    Per-attribute formula:

        c_a = clip(t_a + own_trust_a × (1 − t_a), 0, 1)
            = own_trust_a + (1 − own_trust_a) × t_a

    Interpretation: lie a fraction `own_trust_a` of the way from t_a to 1.0.
    When trust is fresh (=1), claim 1.0 (Naive). When trust is burned (=0),
    claim t_a (honest rest). Smooth interpolation between — backs off
    proportionally to how much trust you've already spent on that attribute.

    Each attribute is treated independently. Mean claim stays near 1.0
    early game (high trust); drops gradually as trust depletes.

    Contrast with the previous rest-only-the-worst variant: that one
    sacrificed full claim on one attribute every round, even when the others
    were healthy. This version distributes the backoff smoothly across all
    five attributes proportional to their actual trust state."""
    del opponents_trust, round_index, total_rounds
    del own_wins_count, opponents_wins_count
    n = len(truth)
    if len(own_trust) != n:
        raise ValueError("truth/own_trust length mismatch")
    return [
        _dc_quantize(
            float(truth[a])
            + max(0.0, min(1.0, float(own_trust[a]))) * (1.0 - float(truth[a]))
        )
        for a in range(n)
    ]


def deception_policy_calibrated(
    truth: _DCList[float],
    *,
    own_trust: _DCList[float],
    opponents_trust: _DCList[_DCList[float]] | None = None,
    round_index: int = 0,
    total_rounds: int = 12,
    own_wins_count: int = 0,
    opponents_wins_count: _DCList[int] | None = None,
    truth_threshold: float = 0.5,
    trust_floor: float = 0.6,
) -> _DCList[float]:
    """Calibrated: actual trust management.

    Two simultaneous rules (both per-attribute, independent across attrs):

      1. CHEAP-LIE RULE: claim 1.0 only on attrs where truth is already high
         (t_a > truth_threshold). Discrepancy from a 1.0 claim is then
         (1 - t_a) ≈ small, so trust damage per win is minimized.

      2. TRUST-FLOOR RULE: if own_trust_a < trust_floor, do not inflate that
         attribute at all — claim truth instead, so the buyer's running
         average can dilute back toward 1.0 over future rounds.

    Attributes that don't pass both gates are emitted as truth. This is what
    the original "Calibrated" specification meant: claim high only where it's
    cheap, back off on burned attrs to let them heal.

    Contrast with T3 (max-headroom concentrator), which slams the LOWEST-truth
    attribute to 1.0 — exactly the most expensive lie possible. T3 is
    misnamed for this experiment; it tests max-score concentration, not
    trust-preserving calibration."""
    del opponents_trust, round_index, total_rounds
    del own_wins_count, opponents_wins_count
    n = len(truth)
    if len(own_trust) != n:
        raise ValueError("truth/own_trust length mismatch")
    out: _DCList[float] = []
    for a in range(n):
        t_a = float(truth[a])
        trust_a = max(0.0, min(1.0, float(own_trust[a])))
        if t_a > truth_threshold and trust_a >= trust_floor:
            # Cheap to lie here, and trust budget can afford it.
            out.append(1.0)
        else:
            # Either too-expensive a lie (low truth) or trust too damaged
            # — back off and play honest on this attribute.
            out.append(_dc_quantize(t_a))
    return out


def deception_policy_tier2_truth_anchored(
    truth: _DCList[float],
    *,
    own_trust: _DCList[float],
    opponents_trust: _DCList[_DCList[float]] | None = None,
    round_index: int = 0,
    total_rounds: int = 12,
    own_wins_count: int = 0,
    opponents_wins_count: _DCList[int] | None = None,
) -> _DCList[float]:
    """T2 — truth-anchored cheap-lie selector. Info used: {truth}.

    First "thinking" tier: under track-record (denom = wins), claiming 1.0
    on a high-truth attribute spends almost no trust (disc = 1 - t ≈ small),
    while claiming 1.0 on a low-truth attribute spends a lot (disc = 1 -
    small_t ≈ large). T2 claims 1.0 only where the lie is cheap and stays
    moderate elsewhere:

        c_a = 1.0  if t_a >= 0.5  (cheap lie)
              clip(t_a + 0.3, 0.5, 1.0)  otherwise  (credible floor)

    State-free over time; reads only this round's truth. Aggressive — mean
    claim around 0.85, minimum 0.5."""
    del own_trust, opponents_trust, round_index, total_rounds
    del own_wins_count, opponents_wins_count
    out: _DCList[float] = []
    for t in truth:
        t = float(t)
        if t >= 0.5:
            out.append(1.0)
        else:
            out.append(_dc_quantize(max(0.5, min(1.0, t + 0.3))))
    return out


# Back-compat alias for old name.
deception_policy_tier2_uniform_backoff = deception_policy_tier2_truth_anchored


def deception_policy_tier3_self_aware(
    truth: _DCList[float],
    *,
    own_trust: _DCList[float],
    opponents_trust: _DCList[_DCList[float]] | None = None,
    round_index: int = 0,
    total_rounds: int = 12,
    own_wins_count: int = 0,
    opponents_wins_count: _DCList[int] | None = None,
) -> _DCList[float]:
    """T3 — self-aware. Info used: {truth, own_trust, own_wins, progress}.

    Adds three pieces of own-state on top of T2's truth-only baseline:

      1. PER-ATTR HEALING: if own_trust_a < 0.5, claim t_a verbatim. Under
         track-record this is the *only* way a damaged attribute heals: an
         honest win brings disc=0, which drops the running average toward
         zero. T2 has no concept of healing because it ignores own_trust.

      2. PROGRESS-AWARE THRESHOLD: the cheap-lie threshold shifts down
         (more attrs qualify as "cheap") as the round counter rises.
         Late-game lies cost less because there are fewer future rounds
         to benefit from preserved trust.

      3. WINS-AWARE INFLATION: the moderate-claim term scales with wins
         so far. Under track-record, marginal cost of a new disc point =
         disc / (wins + 1) — so each win you've already booked makes the
         next inflation cheaper.

    Formula:
        if own_trust_a < 0.5:
            c_a = t_a                         (heal)
        else:
            threshold = 0.5 − 0.3 × progress
            if t_a >= threshold:
                c_a = 1.0                     (cheap lie, easier to qualify late)
            else:
                c_a = clip(t_a + 0.3 + 0.3 × wins_norm, 0.5, 1.0)

    where progress = round_index/total_rounds and wins_norm = wins/total_rounds."""
    del opponents_trust
    del opponents_wins_count
    n = len(truth)
    if len(own_trust) != n:
        raise ValueError("truth/own_trust length mismatch")
    N = max(1, int(total_rounds))
    progress = max(0.0, min(1.0, float(round_index) / N))
    wins_norm = max(0.0, min(1.0, float(own_wins_count) / N))
    threshold = 0.5 - 0.3 * progress
    out: _DCList[float] = []
    for a in range(n):
        t = float(truth[a])
        trust = max(0.0, min(1.0, float(own_trust[a])))
        if trust < 0.5:
            out.append(_dc_quantize(t))
        elif t >= threshold:
            out.append(1.0)
        else:
            out.append(_dc_quantize(max(0.5, min(1.0, t + 0.3 + 0.3 * wins_norm))))
    return out


deception_policy_tier3_trust_proportional = deception_policy_tier3_self_aware


def deception_policy_tier4_pack_aware(
    truth: _DCList[float],
    *,
    own_trust: _DCList[float],
    opponents_trust: _DCList[_DCList[float]] | None = None,
    round_index: int = 0,
    total_rounds: int = 12,
    own_wins_count: int = 0,
    opponents_wins_count: _DCList[int] | None = None,
) -> _DCList[float]:
    """T4 — pack-aware. Info used: {T3 + opp_trust + opp_wins}.

    Builds on T3's self-aware logic and adds two opponent-state signals:

      A. OPP-WEAKNESS BONUS: when the strongest opponent on attribute a has
         damaged trust (max_opp_trust_a low), I can dominate that attribute
         at a cheaper claim. The T3 cheap-lie threshold is LOWERED by
         0.2 × (1 − max_opp_trust_a) on attrs where opps are weak, so more
         attributes qualify as cheap-lie targets.

      B. OPP-WINS PRESSURE: when the leading opponent has many wins, they
         are at peak strategic power (their trust ratio is most efficient).
         The non-cheap inflation term gets a bonus +0.2 × opp_max_wins_norm
         to push harder before their position consolidates further.

    Formula (extending T3):
        opp_weakness_a   = 1 − max(opp_trust_a)
        opp_pressure     = max(opp_wins) / total_rounds
        threshold_a      = 0.5 − 0.3 × progress − 0.2 × opp_weakness_a
        if own_trust_a < 0.5:
            c_a = t_a
        elif t_a >= threshold_a:
            c_a = 1.0
        else:
            c_a = clip(t_a + 0.3 + 0.3 × wins_norm + 0.2 × opp_pressure, 0.5, 1.0)

    Reduces to T3 when opps are fresh (opp_weakness=0, opp_pressure=0).
    Strict generalization in the auction-T4 sense — extra opp info never
    hurts, sometimes helps."""
    n = len(truth)
    if len(own_trust) != n:
        raise ValueError("truth/own_trust length mismatch")
    N = max(1, int(total_rounds))
    progress = max(0.0, min(1.0, float(round_index) / N))
    wins_norm = max(0.0, min(1.0, float(own_wins_count) / N))
    if opponents_trust:
        opp_max_per_attr = [
            max(max(0.0, min(1.0, v[a])) for v in opponents_trust)
            for a in range(n)
        ]
    else:
        opp_max_per_attr = [1.0] * n
    del opponents_wins_count
    # Strength-based push: push harder on attributes where opp is STRONG
    # (still in scoring territory) AND I have spare wins-capacity to spend.
    # Per-attribute marginal calculus under track-record:
    #   - opp_strength_a (= max_opp_trust_a) high → opp's per-attr ceiling is
    #     high → I need to push to compete on this attr.
    #   - opp_strength_a low → opp's per-attr score is bounded low → moderate
    #     play already wins this attr → don't waste own trust here.
    #   - wins_norm high → my own marginal cost-per-disc is tiny (denominator
    #     dilution) → cheap to push.
    # Push factor combines both: only push hard where competition is real AND
    # I can afford it.
    out: _DCList[float] = []
    for a in range(n):
        t = float(truth[a])
        trust = max(0.0, min(1.0, float(own_trust[a])))
        opp_strength = opp_max_per_attr[a]  # high = opp is strong on a
        push_a = wins_norm * opp_strength
        threshold = 0.5 - 0.3 * progress - 0.3 * push_a
        if trust < 0.5:
            out.append(_dc_quantize(t))
        elif t >= threshold:
            out.append(1.0)
        else:
            c = t + 0.3 + 0.3 * wins_norm
            out.append(_dc_quantize(max(0.5, min(1.0, c))))
    return out


deception_policy_tier4_pack_proportional = deception_policy_tier4_pack_aware


# Back-compat aliases — legacy imports reference older T2/T3/T4 names.
deception_policy_tier2_cheap_lie = deception_policy_tier2_uniform_backoff
deception_policy_tier3_trust_budgeter = deception_policy_tier3_trust_proportional
deception_policy_tier4_pack_shadower = deception_policy_tier4_pack_proportional
deception_policy_tier2_uniform_aggressive = deception_policy_tier2_uniform_backoff
deception_policy_tier3_trust_scaled_aggressive = deception_policy_tier3_trust_proportional
deception_policy_tier4_opp_adjusted_aggressive = deception_policy_tier4_pack_proportional
deception_policy_tier2_max_headroom = deception_policy_tier2_uniform_backoff
deception_policy_tier2_uniform = deception_policy_tier2_uniform_backoff
deception_policy_tier2_greedy_maxweight = deception_policy_tier2_uniform_backoff
deception_policy_tier3_trust_aware_concentrator = deception_policy_tier3_trust_proportional
deception_policy_tier3_per_attr_nash = deception_policy_tier3_trust_proportional
deception_policy_tier3_single_attr_lie = deception_policy_tier3_trust_proportional
deception_policy_tier4_opp_aware_concentrator = deception_policy_tier4_pack_proportional
deception_policy_tier4_symmetric_nash = deception_policy_tier4_pack_proportional
deception_policy_tier4_backloaded = deception_policy_tier4_pack_proportional
deception_policy_tier4_concentrator = deception_policy_tier4_pack_proportional


DECEPTION_TIER_POLICIES = {
    # Canonical semantic names — preferred for new code and outputs.
    "Math-Trivial-Max":   deception_policy_tier1_trivial_max,
    "Math-Truth-Anchored": deception_policy_tier2_truth_anchored,
    "Math-Self-Aware":    deception_policy_tier3_self_aware,
    "Math-Pack-Aware":    deception_policy_tier4_pack_aware,
    # Math-RL ("Math-T5") is dispatched via _deception_t5_claim in server.py,
    # not via this registry, since it loads a PPO checkpoint at runtime.
    # Back-compat aliases — older save slots and scripts use the T-numbered
    # form. Keep both so historical episode_log.json reads correctly and
    # existing run scripts don't break.
    "Math-T1": deception_policy_tier1_trivial_max,
    "Math-T2": deception_policy_tier2_truth_anchored,
    "Math-T3": deception_policy_tier3_self_aware,
    "Math-T4": deception_policy_tier4_pack_aware,
    # Other hand-coded benchmarks (not part of the info-ladder).
    "Math-Naive":      deception_policy_naive_max,
    "Math-Smart":      deception_policy_smart,
    "Math-Calibrated": deception_policy_calibrated,
}
