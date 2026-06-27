"""PPO for negotiation Math-RL — learner plays one role, mimic plays the other.

Simplified hybrid action space:
    output 2 values:
        accept_prob   ∈ [0,1] (sigmoid)
        price_ratio   ∈ ℝ     (Gaussian sample, applied as price = ratio * own_value)

At each turn:
    1. Sample accept_prob → Bernoulli(accept_prob)
        - if accept and standing legal → accept
    2. Otherwise propose price = clip(ratio * own_value, legal_range)

Reward = surplus on agreement (own_value - price for buyer, price - own_value for seller).
Penalty for no-deal at message_limit = 0 reward (default zero reward — no extra cost).

Both roles (buyer + seller) train in one policy by feeding role_is_seller in the obs.
Opponent samples uniformly from the 5 mimics each episode.

Usage:
    python -m simulation.train_rl_negotiation --episodes 3000 --label v1
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal

from simulation.env import TravelGameEnv
from simulation.negotiation_mimic import build_feature_vector, negotiation_mimic_action
from simulation.state import NegotiationTurnAction


MODELS_DIR = Path(__file__).parent / "models" / "rl_negotiation"
OBS_DIM = 32
MIMIC_OPPONENTS = ["Mimic-GPT-5.4", "Mimic-Grok", "Mimic-Opus", "Mimic-Pro", "Mimic-Llama"]


class NegotiationPolicy(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        # Heads: accept_prob (sigmoid) + price_ratio (Gaussian mean)
        self.accept_head = nn.Linear(hidden, 1)
        self.price_mean_head = nn.Linear(hidden, 1)
        self.price_log_std = nn.Parameter(torch.full((1,), -1.0))   # std≈0.37

    def forward(self, obs):
        h = self.trunk(obs)
        accept_logit = self.accept_head(h).squeeze(-1)
        price_mean = self.price_mean_head(h).squeeze(-1)
        log_std = self.price_log_std.expand_as(price_mean).clamp(-4.0, 1.0)
        return accept_logit, price_mean, log_std

    def sample_action(self, obs):
        a_logit, p_mean, p_log_std = self(obs)
        ber = Bernoulli(logits=a_logit)
        accept = ber.sample()
        accept_logp = ber.log_prob(accept)
        gauss = Normal(p_mean, p_log_std.exp())
        price_ratio = gauss.rsample()
        price_logp = gauss.log_prob(price_ratio)
        logp = accept_logp + price_logp
        return accept, price_ratio, logp

    def log_prob(self, obs, accept, price_ratio):
        a_logit, p_mean, p_log_std = self(obs)
        ber = Bernoulli(logits=a_logit)
        gauss = Normal(p_mean, p_log_std.exp())
        return ber.log_prob(accept) + gauss.log_prob(price_ratio)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


def _run_episode(policy: NegotiationPolicy, opponent_alias: str, *, seed: int,
                 learner_role: str, message_limit: int = 10):
    """Run one episode with learner in `learner_role`, mimic opponent in the other."""
    env = TravelGameEnv({
        "mode": "buyer_seller_negotiation",
        "selected_models": ["GPT-5.4", "GPT-5.4", "GPT-5.4"],  # ignored — we direct-dispatch
        "seed": seed,
        "negotiation_message_limit": message_limit,
    })
    env.reset(seed=seed)
    buyer = env.world["buyer_true"]
    seller = env.world["seller_true"]

    obs_buf, accept_buf, price_buf, logp_buf, rew_buf, done_buf = [], [], [], [], [], []
    turns = []
    standing_price = None

    # Seller opens. If learner is seller, sample from policy; else mimic.
    if learner_role == "seller":
        own_value = float(seller.baseline_value); own_target = float(seller.asking_price)
        x = build_feature_vector(role="seller", own_private_value=own_value, own_target_price=own_target,
                                 turn_history=[], standing_price=None, turn_index=0, message_limit=message_limit)
        xt = torch.tensor([x], dtype=torch.float32)
        with torch.no_grad():
            accept, ratio, logp = policy.sample_action(xt)
        a = float(accept.item()); pr = float(ratio.item()); lp = float(logp.item())
        opening_price = max(int(seller.baseline_value), int(round(max(1.0, pr) * own_value)))
        turns.append(NegotiationTurnAction(speaker="seller", proposed_price=opening_price, message_text=""))
        standing_price = float(opening_price)
        obs_buf.append(x); accept_buf.append(a); price_buf.append(pr); logp_buf.append(lp); rew_buf.append(0.0); done_buf.append(False)
    else:
        mres = negotiation_mimic_action(opponent_alias, role="seller",
                                        own_private_value=float(seller.baseline_value),
                                        own_target_price=float(seller.asking_price),
                                        turn_history=[], standing_price=None,
                                        turn_index=0, message_limit=message_limit)
        opening_price = max(int(seller.baseline_value), int(mres["proposed_price"]))
        turns.append(NegotiationTurnAction(speaker="seller", proposed_price=opening_price, message_text=""))
        standing_price = float(opening_price)

    agreed = None
    for turn_idx in range(1, message_limit):
        buyer_turn = (turn_idx % 2) == 1
        is_learner_turn = (buyer_turn and learner_role == "buyer") or (not buyer_turn and learner_role == "seller")
        role = "buyer" if buyer_turn else "seller"
        own_value = float(seller.baseline_value) if role == "seller" else float(buyer.budget)
        own_target = float(seller.asking_price) if role == "seller" else float(buyer.target_price)
        history = [{"speaker": t.speaker, "price": int(t.proposed_price)} for t in turns]

        if is_learner_turn:
            x = build_feature_vector(role=role, own_private_value=own_value, own_target_price=own_target,
                                     turn_history=history, standing_price=standing_price,
                                     turn_index=turn_idx, message_limit=message_limit)
            xt = torch.tensor([x], dtype=torch.float32)
            with torch.no_grad():
                accept_t, ratio_t, logp_t = policy.sample_action(xt)
            a = float(accept_t.item()); pr = float(ratio_t.item()); lp = float(logp_t.item())
            # Decide action
            legal = ((role == "buyer" and standing_price <= own_value) or
                     (role == "seller" and standing_price >= own_value))
            if a > 0.5 and legal:
                agreed = int(standing_price)
                turns.append(NegotiationTurnAction(speaker=role, proposed_price=int(standing_price), message_text=""))
                # Reward = surplus
                surplus = (own_value - agreed) if role == "buyer" else (agreed - own_value)
                obs_buf.append(x); accept_buf.append(a); price_buf.append(pr); logp_buf.append(lp)
                rew_buf.append(float(surplus)); done_buf.append(True)
                break
            # Counter
            new_price = int(round(max(1.0, pr) * own_value))
            if role == "buyer":
                new_price = max(1, min(int(buyer.budget), new_price))
            else:
                new_price = max(int(seller.baseline_value), new_price)
            turns.append(NegotiationTurnAction(speaker=role, proposed_price=new_price, message_text=""))
            standing_price = float(new_price)
            obs_buf.append(x); accept_buf.append(a); price_buf.append(pr); logp_buf.append(lp)
            rew_buf.append(0.0); done_buf.append(False)
        else:
            mres = negotiation_mimic_action(opponent_alias, role=role,
                                            own_private_value=own_value,
                                            own_target_price=own_target,
                                            turn_history=history, standing_price=standing_price,
                                            turn_index=turn_idx, message_limit=message_limit)
            if mres["action"] == "accept":
                legal_opp = ((role == "buyer" and standing_price <= own_value) or
                             (role == "seller" and standing_price >= own_value))
                if legal_opp:
                    agreed = int(standing_price)
                    turns.append(NegotiationTurnAction(speaker=role, proposed_price=int(standing_price), message_text=""))
                    break
            new_price = mres["proposed_price"]
            if role == "buyer":
                new_price = max(1, min(int(buyer.budget), int(new_price)))
            else:
                new_price = max(int(seller.baseline_value), int(new_price))
            turns.append(NegotiationTurnAction(speaker=role, proposed_price=int(new_price), message_text=""))
            standing_price = float(new_price)

    if not done_buf or not done_buf[-1]:
        # Episode ended without agreement (either we exhausted turns)
        if obs_buf and not done_buf[-1]:
            done_buf[-1] = True  # mark last learner step done
    return {
        "obs": obs_buf, "accepts": accept_buf, "prices": price_buf, "log_probs": logp_buf,
        "rewards": rew_buf, "dones": done_buf, "agreed": agreed,
    }


def compute_gae(rewards, values, dones, *, gamma=0.99, lam=0.95):
    advs = [0.0] * len(rewards)
    last_adv = 0.0
    next_v = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        nv = 0.0 if dones[t] else next_v
        delta = rewards[t] + gamma * nv - values[t]
        last_adv = delta + gamma * lam * (0.0 if dones[t] else last_adv)
        advs[t] = last_adv
        next_v = values[t]
    returns = [a + v for a, v in zip(advs, values)]
    return advs, returns


def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    policy = NegotiationPolicy()
    value_net = ValueNet()
    opt_p = torch.optim.Adam(policy.parameters(), lr=args.lr, weight_decay=1e-5)
    opt_v = torch.optim.Adam(value_net.parameters(), lr=args.lr_value, weight_decay=1e-5)

    reward_hist = []
    best_avg = -math.inf

    for ep in range(args.episodes):
        opponent = random.choice(MIMIC_OPPONENTS)
        role = random.choice(["buyer", "seller"])
        roll = _run_episode(policy, opponent, seed=ep + 1, learner_role=role,
                            message_limit=args.message_limit)
        if not roll["obs"]:
            continue
        total_reward = sum(roll["rewards"])
        reward_hist.append(total_reward)

        obs_t = torch.tensor(roll["obs"], dtype=torch.float32)
        with torch.no_grad():
            values = value_net(obs_t).tolist()
        adv, ret = compute_gae(roll["rewards"], values, roll["dones"], gamma=args.gamma, lam=args.gae_lambda)
        adv_t = torch.tensor(adv, dtype=torch.float32)
        ret_t = torch.tensor(ret, dtype=torch.float32)
        acc_t = torch.tensor(roll["accepts"], dtype=torch.float32)
        pr_t = torch.tensor(roll["prices"], dtype=torch.float32)
        oldlp_t = torch.tensor(roll["log_probs"], dtype=torch.float32)
        if adv_t.numel() > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(args.ppo_epochs):
            new_lp = policy.log_prob(obs_t, acc_t, pr_t)
            ratio = (new_lp - oldlp_t).exp()
            cr = ratio.clamp(1.0 - args.clip_eps, 1.0 + args.clip_eps)
            policy_loss = -torch.min(ratio * adv_t, cr * adv_t).mean()
            v_pred = value_net(obs_t)
            v_loss = F.mse_loss(v_pred, ret_t)
            opt_p.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt_p.step()
            opt_v.zero_grad()
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
            opt_v.step()

        if (ep + 1) % args.log_every == 0:
            recent = reward_hist[-100:]
            avg = sum(recent) / max(1, len(recent))
            print(f"  ep {ep+1:5d}/{args.episodes}  avg_reward_100={avg:+.2f}  total_this={total_reward:+.1f}  agreed={roll['agreed'] is not None}", flush=True)
            if avg > best_avg:
                best_avg = avg
                torch.save({"policy": policy.state_dict(), "value": value_net.state_dict(), "best_avg": best_avg},
                           MODELS_DIR / f"negotiation_rl_{args.label}.best.pt")

    torch.save({"policy": policy.state_dict(), "value": value_net.state_dict(),
                "final_avg": sum(reward_hist[-100:]) / max(1, len(reward_hist[-100:]))},
               MODELS_DIR / f"negotiation_rl_{args.label}.pt")
    print(f"\nDone. final_avg={sum(reward_hist[-100:])/max(1,len(reward_hist[-100:])):+.2f}  best_avg={best_avg:+.2f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-value", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--message-limit", type=int, default=10)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--label", default="v1")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
