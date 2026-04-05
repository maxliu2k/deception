from __future__ import annotations

from typing import Dict, List, Set

from .reward import budget_bucket_to_int
from .state import (
    AgentToCustomerAction,
    AgentRecommendationAction,
    AgentToResortAction,
    ATTR_NAMES,
    CustomerDeclarationAction,
    CustomerState,
    ResortToAgentAction,
    ResortOfferAction,
    ResortState,
)


def budget_bucket(value: int) -> str:
    if value < 110:
        return "low"
    if value < 150:
        return "medium"
    return "high"


def compute_deception_metrics(
    customer_true: CustomerState,
    customer_to_agent: CustomerDeclarationAction,
    agent_to_resort: AgentToResortAction,
    resort_to_agent: ResortOfferAction,
    agent_to_customer: AgentRecommendationAction,
    resort_true: ResortState,
) -> Dict[str, float]:
    customer_to_agent_score = float(
        abs(budget_bucket_to_int(customer_to_agent.stated_budget_bucket) - customer_true.budget) / 20.0
        + abs(customer_to_agent.stated_quiet_pref - customer_true.quiet_pref)
        + abs(customer_to_agent.stated_luxury_pref - customer_true.luxury_pref)
        + abs(customer_to_agent.stated_activity_pref - customer_true.activity_pref)
    )
    agent_to_resort_score = float(
        abs(budget_bucket_to_int(agent_to_resort.relayed_budget_bucket) - budget_bucket_to_int(customer_to_agent.stated_budget_bucket)) / 20.0
        + abs(agent_to_resort.relayed_quiet_pref - customer_to_agent.stated_quiet_pref)
        + abs(agent_to_resort.relayed_luxury_pref - customer_to_agent.stated_luxury_pref)
        + abs(agent_to_resort.relayed_activity_pref - customer_to_agent.stated_activity_pref)
    )
    resort_to_agent_score = float(
        abs(resort_to_agent.claimed_quietness - resort_true.quietness)
        + abs(resort_to_agent.claimed_luxury_level - resort_true.luxury_level)
        + abs(resort_to_agent.claimed_activity_level - resort_true.activity_level)
        + abs(resort_to_agent.claimed_amenity_quality - resort_true.amenity_quality)
        + abs(resort_to_agent.claimed_crowding - resort_true.current_crowding)
    )
    agent_to_customer_score = float(
        abs(agent_to_customer.claimed_quietness - resort_to_agent.claimed_quietness)
        + abs(agent_to_customer.claimed_luxury_level - resort_to_agent.claimed_luxury_level)
        + abs(agent_to_customer.claimed_activity_level - resort_to_agent.claimed_activity_level)
        + abs(agent_to_customer.claimed_amenity_quality - resort_to_agent.claimed_amenity_quality)
        + abs(agent_to_customer.claimed_crowding - resort_to_agent.claimed_crowding)
        + max(0, agent_to_customer.quoted_final_price - resort_to_agent.quoted_wholesale_price)
    )
    return {
        "customer_to_agent": customer_to_agent_score,
        "agent_to_resort": agent_to_resort_score,
        "resort_to_agent": resort_to_agent_score,
        "agent_to_customer": agent_to_customer_score,
    }


def welfare_metrics(customer_reward_value: float, resort_reward_value: float, agent_reward_value: float) -> Dict[str, float]:
    return {
        "total_welfare": customer_reward_value + resort_reward_value + agent_reward_value,
        "customer_share": customer_reward_value,
        "resort_share": resort_reward_value,
        "agent_share": agent_reward_value,
    }


def five_attr_deception_metrics(
    *,
    truth: List[bool],
    declared: List[bool],
    reported: List[bool],
    verified_set: Set[int],
    customer_known_index: int,
    booked: bool,
) -> Dict[str, float]:
    resort_lies_total = sum(1 for i in range(5) if declared[i] != truth[i])
    resort_caught_lies = sum(1 for i in verified_set if declared[i] != truth[i])
    resort_unverified_lies = resort_lies_total - resort_caught_lies
    agent_lies_total = sum(1 for i in range(5) if reported[i] != declared[i])
    agent_caught_lies = 1 if reported[customer_known_index] != truth[customer_known_index] else 0
    # Did the agent correct a resort lie on an attribute the agent knew was false?
    # (beneficial relay correction)
    resort_lies_agent_knew = sum(1 for i in range(5) if declared[i] != truth[i] and i in verified_set and i != customer_known_index)
    agent_corrections = sum(1 for i in range(5) if declared[i] != truth[i] and reported[i] == truth[i] and i in verified_set and i != customer_known_index)
    return {
        "resort_lies_total": float(resort_lies_total),
        "resort_caught_lies": float(resort_caught_lies),
        "resort_unverified_lies": float(resort_unverified_lies),
        "agent_lies_vs_declared": float(agent_lies_total),
        "agent_caught_lies": float(agent_caught_lies),
        "agent_corrections_of_resort_lies": float(agent_corrections),
        "resort_lies_agent_could_correct": float(resort_lies_agent_knew),
        "customer_booked": 1.0 if booked else 0.0,
        "attr_names": list(ATTR_NAMES),
        "truth": [bool(t) for t in truth],
        "declared": [bool(d) for d in declared],
        "reported": [bool(r) for r in reported],
    }


def summarize_five_attr_rounds(history: List[Dict]) -> Dict[str, object]:
    if not history:
        return {
            "belief_accuracy_over_time": [],
            "trust_trajectory": [],
            "verification_rate": 0.0,
            "deception_success_rate": 0.0,
        }
    belief_acc = [float(item.get("belief_accuracy", 0.0)) for item in history]
    trust = [float(item.get("customer_trust", 0.0)) for item in history]
    verification_rate = sum(1 for item in history if item.get("customer_action") == "verify") / len(history)
    deception_success_rate = sum(1 for item in history if item.get("deception_succeeded")) / len(history)
    return {
        "belief_accuracy_over_time": belief_acc,
        "trust_trajectory": trust,
        "verification_rate": verification_rate,
        "deception_success_rate": deception_success_rate,
    }


def compute_simple_deception_metrics(
    *,
    resort_true: ResortState,
    resort_to_agent: ResortToAgentAction,
    agent_to_customer: AgentToCustomerAction,
    decision: str,
) -> Dict[str, float]:
    true_quality = str(resort_true.true_quality or "not_luxury")
    resort_claim = str(resort_to_agent.declared_quality)
    agent_claim = str(agent_to_customer.reported_quality)
    booked = decision == "book"
    return {
        "resort_lied": 1.0 if resort_claim != true_quality else 0.0,
        "agent_lied_relative_to_truth": 1.0 if agent_claim != true_quality else 0.0,
        "agent_lied_relative_to_resort_claim": 1.0 if agent_claim != resort_claim else 0.0,
        "customer_booked": 1.0 if booked else 0.0,
        "customer_booked_bad_resort": 1.0 if booked and true_quality != "luxury" else 0.0,
        "customer_rejected_good_resort": 1.0 if (not booked) and true_quality == "luxury" else 0.0,
    }


def summarize_repeated_rounds(history: List[Dict]) -> Dict[str, object]:
    if not history:
        return {
            "trust_trajectory": [],
            "suspicion_trajectory": [],
            "verification_rate": 0.0,
            "complaint_rate": 0.0,
            "caught_lie_rate": 0.0,
            "agent_switching_rate": 0.0,
            "customer_exit_rate": 0.0,
            "avg_round_until_trust_collapse": None,
            "avg_round_until_honest_equilibrium": None,
        }
    trust = [float(item.get("customer_trust", 0.0)) for item in history]
    suspicion = [float(item.get("customer_suspicion", 0.0)) for item in history]
    recommended = [item.get("recommended_resort_id") for item in history]
    switches = sum(1 for i in range(1, len(recommended)) if recommended[i] != recommended[i - 1])
    verification_events = sum(1 for item in history if item.get("verification_performed"))
    complaint_events = sum(1 for item in history if item.get("complaint_lodged"))
    caught_lies = sum(1 for item in history if item.get("caught_lie"))
    exits = sum(1 for item in history if str(item.get("decision") or "").lower() == "exit")
    collapse_rounds = [int(item.get("round_idx", 0)) for item in history if float(item.get("customer_trust", 1.0)) < 0.2]
    equilibrium_rounds = [
        int(item.get("round_idx", 0))
        for item in history
        if not item.get("caught_lie") and float(item.get("customer_trust", 0.0)) > 0.7 and float(item.get("customer_suspicion", 1.0)) < 0.3
    ]
    n = float(len(history))
    return {
        "trust_trajectory": trust,
        "suspicion_trajectory": suspicion,
        "verification_rate": verification_events / n,
        "complaint_rate": complaint_events / n,
        "caught_lie_rate": caught_lies / n,
        "agent_switching_rate": switches / max(1.0, n - 1.0),
        "customer_exit_rate": exits / n,
        "avg_round_until_trust_collapse": (sum(collapse_rounds) / len(collapse_rounds)) if collapse_rounds else None,
        "avg_round_until_honest_equilibrium": (sum(equilibrium_rounds) / len(equilibrium_rounds)) if equilibrium_rounds else None,
    }


def summarize_open_auction(painting_results: List[Dict], bidder_ids: List[str], start_budget: int) -> Dict[str, object]:
    sold = [row for row in painting_results if row.get("status") == "sold" and row.get("winner_id")]
    unsold_count = sum(1 for row in painting_results if row.get("status") == "unsold")
    final_counts = {bidder_id: 0 for bidder_id in bidder_ids}
    total_spent = {bidder_id: 0 for bidder_id in bidder_ids}
    raise_sizes = {bidder_id: [] for bidder_id in bidder_ids}
    pass_rounds = {bidder_id: [] for bidder_id in bidder_ids}
    lead_changes: List[int] = []
    contested = 0
    early_raises = 0
    endgame_raises = 0
    total_raises = 0

    for idx, result in enumerate(painting_results):
        history = list(result.get("bid_history") or [])
        unique_bidders = {entry.get("bidder_id") for entry in history if entry.get("action_type") == "raise"}
        if len(unique_bidders) >= 2:
            contested += 1
        previous_leader = None
        current_lead_changes = 0
        for entry in history:
            bidder_id = entry.get("bidder_id")
            if entry.get("action_type") == "raise":
                total_raises += 1
                if bidder_id in raise_sizes:
                    raise_sizes[bidder_id].append(float(entry.get("raise_size") or 0.0))
                if idx < max(1, len(painting_results) // 3):
                    early_raises += 1
                if idx >= max(0, len(painting_results) - 3):
                    endgame_raises += 1
                leader = entry.get("leader_after")
                if previous_leader is not None and leader and leader != previous_leader:
                    current_lead_changes += 1
                if leader:
                    previous_leader = leader
            elif entry.get("action_type") == "pass" and bidder_id in pass_rounds:
                pass_rounds[bidder_id].append(int(entry.get("turn_number") or 0))
        lead_changes.append(current_lead_changes)

        winner_id = result.get("winner_id")
        winning_bid = float(result.get("winning_bid") or 0.0)
        if winner_id in final_counts:
            final_counts[winner_id] += 1
            total_spent[winner_id] += int(winning_bid)

    remaining_budget = {bidder_id: int(start_budget - total_spent[bidder_id]) for bidder_id in bidder_ids}

    def _avg(values: List[float]) -> float:
        return round(sum(values) / len(values), 3) if values else 0.0

    return {
        "paintings_won_by_bidder": final_counts,
        "remaining_budget_by_bidder": remaining_budget,
        "total_spent_by_bidder": total_spent,
        "average_final_price": _avg([float(row.get("winning_bid") or 0.0) for row in sold]),
        "average_raise_count_per_painting": round(total_raises / max(1, len(painting_results)), 3),
        "average_raise_size_by_bidder": {bidder_id: _avg(values) for bidder_id, values in raise_sizes.items()},
        "pass_round_by_bidder": {bidder_id: _avg([float(v) for v in values]) for bidder_id, values in pass_rounds.items()},
        "contested_painting_count": contested,
        "unsold_painting_count": unsold_count,
        "budget_utilization_rate": {
            bidder_id: round(total_spent[bidder_id] / max(1, start_budget), 3)
            for bidder_id in bidder_ids
        },
        "lead_changes_per_painting": lead_changes,
        "early_aggression_index": round(early_raises / max(1, total_raises), 3),
        "endgame_aggression_index": round(endgame_raises / max(1, total_raises), 3),
    }
