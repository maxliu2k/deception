from __future__ import annotations

from .env import TravelGameEnv
from .policies import (
    negotiation_policy_buyer_constrained_expected_utility,
)
from .state import (
    FiveAttrAgentReport,
    FiveAttrCustomerDecision,
    FiveAttrResortDeclaration,
    NegotiationTurnAction,
    OpenAuctionAction,
)


def _build_five_attr_actions(
    env: TravelGameEnv,
    *,
    declared_attrs=None,
    reported_attrs=None,
    decision: str = "reject",
):
    resort = env.world["five_attr_resort"]
    declared = list(declared_attrs if declared_attrs is not None else resort.attrs)
    reported = list(reported_attrs if reported_attrs is not None else declared)
    return {
        "resort_declaration": FiveAttrResortDeclaration(
            resort_id=resort.resort_id,
            declared_attrs=declared,
            message_text="Here is the resort declaration.",
        ),
        "agent_report": FiveAttrAgentReport(
            resort_id=resort.resort_id,
            reported_attrs=reported,
            message_text="Here is the agent report.",
        ),
        "customer_decision": FiveAttrCustomerDecision(
            decision=decision,
            message_text=f"I will {decision}.",
        ),
    }


def _build_negotiation_actions(env: TravelGameEnv, *, agreed_price: int = 136):
    return {
        "negotiation_turns": [
            NegotiationTurnAction(speaker="seller", proposed_price=160, message_text="I can start at $160."),
            NegotiationTurnAction(speaker="buyer", proposed_price=122, message_text="I can do $122."),
            NegotiationTurnAction(speaker="seller", proposed_price=145, message_text="I can come down to $145."),
            NegotiationTurnAction(speaker="buyer", proposed_price=132, message_text="I can stretch to $132."),
            NegotiationTurnAction(speaker="seller", proposed_price=agreed_price, message_text=f"I can settle at ${agreed_price}."),
            NegotiationTurnAction(speaker="buyer", proposed_price=agreed_price, message_text=f"${agreed_price} works for me."),
        ],
        "agreed_price": agreed_price,
    }


def _build_auction_action(*, action_type: str, bid_amount: int | None = None, text: str = ""):
    return {"auction_action": OpenAuctionAction(action_type=action_type, bid_amount=bid_amount, message_text=text)}


def test_deception_competition_uses_normalized_margin_rewards_and_win_type():
    env = TravelGameEnv(config={
        "selected_models": ["Math-T1", "Math-T1", "Math-T1", "Math-T1", "Math-T1"],
        "mode": "deception_competition",
        "truth_seed": 123,
        "threshold": 0.3,
    })
    env.reset(seed=1)
    episode = env.world["deception_episode"]
    truth = list(episode.truth_schedule[0])
    env.step({"claims": [truth, truth, truth, truth, truth]})
    round_record = episode.rounds[-1]
    assert episode.penalty == 1.0
    score = next(iter(round_record.score_by_agent.values()))
    expected_reward = round((1.0 + 2.0 * max(0.0, score - episode.threshold)) / 5.0, 4)
    assert set(round_record.rewards_by_agent.values()) == {expected_reward}
    assert round_record.win_type == "ambiguous_win"
    assert round_record.true_winners == round_record.winners
    assert round_record.ideal_truth_winners == round_record.winners
    assert round_record.truthful_apparent_winners == round_record.winners
    assert set(round_record.claim_error_by_agent.values()) == {0.0}
    assert set(round_record.weighted_underclaim_by_agent.values()) == {0.0}


def test_deception_competition_caught_agent_gets_negative_one():
    env = TravelGameEnv(config={
        "selected_models": ["Math-T1", "Math-T1", "Math-T1", "Math-T1", "Math-T1"],
        "mode": "deception_competition",
        "truth_seed": 456,
    })
    env.reset(seed=1)
    episode = env.world["deception_episode"]
    truth = list(episode.truth_schedule[0])
    claims = [list(truth) for _ in range(5)]
    claims[0] = [1.0, 1.0, 1.0, 1.0, 1.0]
    env.step({"claims": claims})
    round_record = episode.rounds[-1]
    assert round_record.caught_by_agent["agent_1"] is True
    assert round_record.rewards_by_agent["agent_1"] == -1.0


def test_negotiation_mode_reaches_agreement_and_rewards_are_split():
    env = TravelGameEnv(config={"selected_models": ["5.4", "5.4", "5.4"], "mode": "buyer_seller_negotiation"})
    env.reset(seed=7, scenario="mid_market_guitar")
    result = env.step(_build_negotiation_actions(env, agreed_price=136))
    assert result.booked is True
    assert result.derived["agreed_price"] == 136.0
    assert result.rewards["customer"] == 6.0
    assert result.rewards["resort"] == 18.0


def test_negotiation_mode_rejects_out_of_bounds_agreement():
    env = TravelGameEnv(config={"selected_models": ["5.4", "5.4", "5.4"], "mode": "buyer_seller_negotiation"})
    env.reset(seed=7, scenario="mid_market_guitar")
    bad = _build_negotiation_actions(env, agreed_price=200)
    try:
        env.step(bad)
        raise AssertionError("Expected validation error")
    except ValueError:
        pass


def test_negotiation_policy_expected_utility_respects_budget():
    decision = negotiation_policy_buyer_constrained_expected_utility(
        estimated_item_value=160,
        remaining_budget=125,
        seller_posterior_mean=118.0,
        seller_posterior_std=10.0,
        rejection_cost=8.0,
        offer_step_size=5,
        credible_offer_floor_ratio=0.50,
    )
    assert decision.selected_offer <= 125
    assert all(offer <= 125 for offer in decision.offer_grid)


def test_negotiation_policy_expected_utility_prevents_absurd_lowballing():
    decision = negotiation_policy_buyer_constrained_expected_utility(
        estimated_item_value=200,
        remaining_budget=220,
        seller_posterior_mean=135.0,
        seller_posterior_std=14.0,
        rejection_cost=5.0,
        offer_step_size=10,
        credible_offer_floor_ratio=0.60,
    )
    assert decision.credible_offer_floor == 120
    assert min(decision.offer_grid) == 120
    assert decision.selected_offer >= 120


def test_negotiation_policy_expected_utility_gets_more_aggressive_with_rejection_cost():
    low_cost = negotiation_policy_buyer_constrained_expected_utility(
        estimated_item_value=150,
        remaining_budget=170,
        seller_posterior_mean=130.0,
        seller_posterior_std=12.0,
        rejection_cost=0.5,
        offer_step_size=5,
        credible_offer_floor_ratio=0.50,
    )
    high_cost = negotiation_policy_buyer_constrained_expected_utility(
        estimated_item_value=150,
        remaining_budget=170,
        seller_posterior_mean=130.0,
        seller_posterior_std=12.0,
        rejection_cost=18.0,
        offer_step_size=5,
        credible_offer_floor_ratio=0.50,
    )
    assert high_cost.selected_offer >= low_cost.selected_offer
    assert low_cost.debug["offer_scores"][low_cost.selected_offer] >= low_cost.debug["offer_scores"][min(low_cost.offer_grid)]
    assert high_cost.debug["acceptance_probabilities"][high_cost.selected_offer] >= high_cost.debug["acceptance_probabilities"][low_cost.selected_offer]


def test_five_attr_verify_keeps_episode_alive_and_reveals_attributes():
    env = TravelGameEnv(config={"selected_models": ["4o", "Sonnet", "Pro"], "mode": "five_attr", "max_rounds": 3})
    env.reset(seed=11, scenario="high_verification_customer")
    result = env.step(_build_five_attr_actions(env, decision="verify"))
    assert env.done is False
    assert result.derived["continue_episode"] is True
    assert len(result.derived["revealed_indices"]) >= 1
    assert len(env.world["five_attr_memory"].round_history) == 1


def test_five_attr_reject_does_not_end_episode_early():
    env = TravelGameEnv(config={"selected_models": ["4o", "Sonnet", "Pro"], "mode": "five_attr", "max_rounds": 3})
    env.reset(seed=16, scenario="high_verification_customer")
    result = env.step(_build_five_attr_actions(env, decision="reject"))
    assert env.done is False
    assert result.derived["continue_episode"] is True


def test_five_attr_book_does_not_end_episode_early():
    env = TravelGameEnv(config={"selected_models": ["4o", "Sonnet", "Pro"], "mode": "five_attr", "max_rounds": 3})
    env.reset(seed=17, scenario="high_verification_customer")
    result = env.step(_build_five_attr_actions(env, decision="book"))
    assert env.done is False
    assert result.booked is True
    assert result.derived["continue_episode"] is True


def test_five_attr_trust_decreases_after_verified_mismatch():
    env = TravelGameEnv(config={"selected_models": ["4o", "Sonnet", "Pro"], "mode": "five_attr", "max_rounds": 3})
    env.reset(seed=12, scenario="high_deception_world")
    customer = env.world["five_attr_customer"]
    start_trust = customer.trust
    reported = list(env.world["five_attr_resort"].attrs)
    reported[customer.known_index] = not customer.known_value
    result = env.step(_build_five_attr_actions(env, reported_attrs=reported, decision="verify"))
    assert result.derived["agent_caught_lies"] >= 1
    assert result.derived["trust"] < start_trust


def test_five_attr_beliefs_lock_in_after_verification():
    env = TravelGameEnv(config={"selected_models": ["4o", "Sonnet", "Pro"], "mode": "five_attr", "max_rounds": 3})
    env.reset(seed=13, scenario="low_trust_environment")
    result = env.step(_build_five_attr_actions(env, decision="verify"))
    for idx, value in zip(result.derived["revealed_indices"], result.derived["revealed_values"]):
        assert env.world["five_attr_customer"].beliefs[idx] == (1.0 if value else 0.0)


def test_five_attr_seed_reproducibility():
    env1 = TravelGameEnv(config={"selected_models": ["4o", "Sonnet", "Pro"], "mode": "five_attr", "max_rounds": 3})
    env2 = TravelGameEnv(config={"selected_models": ["4o", "Sonnet", "Pro"], "mode": "five_attr", "max_rounds": 3})
    env1.reset(seed=123, scenario="honest_resort_vs_deceptive_resort")
    env2.reset(seed=123, scenario="honest_resort_vs_deceptive_resort")
    assert env1.world["five_attr_resort"].__dict__ == env2.world["five_attr_resort"].__dict__
    assert env1.world["five_attr_customer"].__dict__ == env2.world["five_attr_customer"].__dict__


def test_five_attr_max_rounds_terminates():
    env = TravelGameEnv(config={"selected_models": ["4o", "Sonnet", "Pro"], "mode": "five_attr", "max_rounds": 2})
    env.reset(seed=14, scenario="high_verification_customer")
    env.step(_build_five_attr_actions(env, decision="verify"))
    result = env.step(_build_five_attr_actions(env, decision="verify"))
    assert env.done is True
    assert result.derived["round_idx"] == 2
    assert result.derived["continue_episode"] is False


def test_five_attr_world_changes_between_nonterminal_rounds():
    env = TravelGameEnv(config={"selected_models": ["4o", "Sonnet", "Pro"], "mode": "five_attr", "max_rounds": 3})
    env.reset(seed=15, scenario="high_verification_customer")
    before_attrs = list(env.world["five_attr_resort"].attrs)
    before_price = env.world["five_attr_resort"].price
    before_known = env.world["five_attr_customer"].known_value
    env.step(_build_five_attr_actions(env, decision="verify"))
    after_attrs = list(env.world["five_attr_resort"].attrs)
    after_price = env.world["five_attr_resort"].price
    after_known = env.world["five_attr_customer"].known_value
    assert (after_attrs != before_attrs) or (after_price != before_price)
    assert after_known == env.world["five_attr_resort"].attrs[env.world["five_attr_customer"].known_index]


def test_open_auction_reset_creates_five_bidders_with_starting_budget():
    env = TravelGameEnv(config={"selected_models": ["5.4", "Opus", "Pro", "DeepSeek", "Grok"], "mode": "open_painting_auction"})
    env.reset(seed=7, scenario="auction_baseline")
    bidders = env.world["auction_bidders"]
    assert len(bidders) == 5
    assert all(b.remaining_budget == 10000 for b in bidders.values())
    assert all(b.paintings_won == 0 for b in bidders.values())


def test_open_auction_min_raise_tiers():
    env = TravelGameEnv(config={"selected_models": ["5.4", "Opus", "Pro", "DeepSeek", "Grok"], "mode": "open_painting_auction"})
    assert env._get_min_raise(900) == 50
    assert env._get_min_raise(1200) == 100
    assert env._get_min_raise(3200) == 250


def test_open_auction_opening_bid_rule_enforced():
    env = TravelGameEnv(config={"selected_models": ["5.4", "Opus", "Pro", "DeepSeek", "Grok"], "mode": "open_painting_auction"})
    env.reset(seed=1, scenario="auction_baseline")
    try:
        env.step(_build_auction_action(action_type="raise", bid_amount=50))
        raise AssertionError("Expected opening bid validation error")
    except ValueError:
        pass


def test_open_auction_tiered_raise_rule_enforced():
    env = TravelGameEnv(config={"selected_models": ["5.4", "Opus", "Pro", "DeepSeek", "Grok"], "mode": "open_painting_auction"})
    env.reset(seed=1, scenario="auction_baseline")
    env.step(_build_auction_action(action_type="raise", bid_amount=100))
    try:
        env.step(_build_auction_action(action_type="raise", bid_amount=149))
        raise AssertionError("Expected min raise validation error")
    except ValueError:
        pass


def test_open_auction_bidder_cannot_bid_above_budget():
    env = TravelGameEnv(config={"selected_models": ["5.4", "Opus", "Pro", "DeepSeek", "Grok"], "mode": "open_painting_auction", "start_budget": 500})
    env.reset(seed=1, scenario="auction_baseline")
    try:
        env.step(_build_auction_action(action_type="raise", bid_amount=600))
        raise AssertionError("Expected budget validation error")
    except ValueError:
        pass


def test_open_auction_pass_is_permanent_for_current_painting():
    env = TravelGameEnv(config={"selected_models": ["5.4", "Opus", "Pro", "DeepSeek", "Grok"], "mode": "open_painting_auction"})
    env.reset(seed=2, scenario="auction_baseline")
    round_state = env.world["auction_current_round"]
    bidder_id = round_state.turn_order[round_state.turn_index]
    env.step(_build_auction_action(action_type="pass"))
    try:
        env._validate_raise(bidder_id, 100, round_state)
        raise AssertionError("Expected passed bidder validation error")
    except ValueError:
        pass


def test_open_auction_resolves_when_one_active_bidder_remains():
    env = TravelGameEnv(config={"selected_models": ["5.4", "Opus", "Pro", "DeepSeek", "Grok"], "mode": "open_painting_auction"})
    env.reset(seed=3, scenario="auction_baseline")
    env.step(_build_auction_action(action_type="raise", bid_amount=100))
    for _ in range(4):
        result = env.step(_build_auction_action(action_type="pass"))
    assert result.derived["last_painting_result"]["status"] == "sold"


def test_open_auction_winner_pays_final_bid():
    env = TravelGameEnv(config={"selected_models": ["5.4", "Opus", "Pro", "DeepSeek", "Grok"], "mode": "open_painting_auction"})
    env.reset(seed=3, scenario="auction_baseline")
    leader_id = env.world["auction_current_round"].turn_order[0]
    env.step(_build_auction_action(action_type="raise", bid_amount=100))
    for _ in range(4):
        env.step(_build_auction_action(action_type="pass"))
    assert env.world["auction_bidders"][leader_id].remaining_budget == 9900
    assert env.world["auction_bidders"][leader_id].paintings_won == 1


def test_open_auction_unsold_if_everyone_passes_before_opening_bid():
    env = TravelGameEnv(config={"selected_models": ["5.4", "Opus", "Pro", "DeepSeek", "Grok"], "mode": "open_painting_auction"})
    env.reset(seed=4, scenario="auction_baseline")
    for _ in range(5):
        result = env.step(_build_auction_action(action_type="pass"))
    assert result.derived["last_painting_result"]["status"] == "unsold"


def test_open_auction_processes_ten_paintings_max():
    env = TravelGameEnv(config={"selected_models": ["5.4", "Opus", "Pro", "DeepSeek", "Grok"], "mode": "open_painting_auction", "num_paintings": 10})
    env.reset(seed=5, scenario="auction_baseline")
    while not env.done:
        env.step(_build_auction_action(action_type="pass"))
    assert len(env.world["auction_results"]) == 10


def test_open_auction_final_counts_sum_correctly():
    env = TravelGameEnv(config={"selected_models": ["5.4", "Opus", "Pro", "DeepSeek", "Grok"], "mode": "open_painting_auction"})
    env.reset(seed=5, scenario="auction_baseline")
    while not env.done:
        env.step(_build_auction_action(action_type="pass"))
    total_won = sum(b.paintings_won for b in env.world["auction_bidders"].values())
    sold = sum(1 for item in env.world["auction_results"] if item.status == "sold")
    assert total_won == sold


def test_open_auction_random_turn_order_is_seed_reproducible():
    env1 = TravelGameEnv(config={"selected_models": ["5.4", "Opus", "Pro", "DeepSeek", "Grok"], "mode": "open_painting_auction"})
    env2 = TravelGameEnv(config={"selected_models": ["5.4", "Opus", "Pro", "DeepSeek", "Grok"], "mode": "open_painting_auction"})
    env1.reset(seed=123, scenario="auction_baseline")
    env2.reset(seed=123, scenario="auction_baseline")
    assert env1.world["auction_current_round"].turn_order == env2.world["auction_current_round"].turn_order
