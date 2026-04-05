from __future__ import annotations

import argparse
import json

from .env import TravelGameEnv
from .policies import (
    agent_policy_commission_max,
    customer_booking_decision,
    customer_policy_budget_hiding,
    customer_policy_truthful,
    resort_policy,
    agent_to_resort_policy,
)
from .reward import compute_fit
from .state import ResortState, AgentToResortAction, CustomerDeclarationAction
from .reward import RewardHyperparameters


def resort_policy_truthful(resort: ResortState, relayed_customer: AgentToResortAction, params: RewardHyperparameters):
    return resort_policy(resort, relayed_customer, 0, params)

def resort_policy_moderate_exaggeration(resort: ResortState, relayed_customer: AgentToResortAction, max_attribute_lie: int, params: RewardHyperparameters):
    return resort_policy(resort, relayed_customer, max_attribute_lie, params)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scenario", type=str, default="A", choices=["A", "B", "C"])
    parser.add_argument("--models", type=str, default="4o,Sonnet,Pro", help="Exactly 3 of: 4o,5.4,Flash,Pro,Haiku,Sonnet,Opus,Grok,Kimi,DeepSeek,Llama,GLM")
    args = parser.parse_args()

    selected_models = [x.strip() for x in args.models.split(",") if x.strip()]
    env = TravelGameEnv(config={"selected_models": selected_models})
    reset_info = env.reset(seed=args.seed, scenario=args.scenario)

    customer = env.world["customer_true"]
    customer_decl = customer_policy_truthful(customer) if args.scenario != "C" else customer_policy_budget_hiding(customer)
    
    agent_to_resort_actions = {}
    for rid, resort in env.world["resorts_true"].items():
        agent_to_resort_actions[rid] = agent_to_resort_policy(customer_decl, resort, maximize_close=False)

    resort_offers = {}
    for rid, resort in env.world["resorts_true"].items():
        if args.scenario == "A":
            resort_offers[rid] = resort_policy_moderate_exaggeration(resort, agent_to_resort_actions[rid], env.config["max_attribute_lie"], env.reward_params)
        else:
            resort_offers[rid] = resort_policy_truthful(resort, agent_to_resort_actions[rid], env.reward_params)

    rec = agent_policy_commission_max(
        customer_decl=customer_decl,
        offers=resort_offers,
        commission_rate_by_resort=env.world["agent_true"].commission_rate_by_resort,
    )

    decision = customer_booking_decision(customer, env.world["resorts_true"][rec.recommended_resort_id], rec, env.reward_params)

    result = env.step(
        {
            "customer_to_agent": customer_decl,
            "agent_to_resort": agent_to_resort_actions,
            "resort_to_agent": resort_offers,
            "agent_to_customer": rec,
            "customer_decision": decision,
        }
    )
    payload = {
        "reset": reset_info,
        "booked": result.booked,
        "booked_resort_id": result.booked_resort_id,
        "rewards": result.rewards,
        "derived": result.derived,
        "deception_metrics": result.deception_metrics,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
