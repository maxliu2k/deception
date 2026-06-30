"""Symmetric PPO for negotiation — role-agnostic policy.

Key idea: the policy is TRULY symmetric across buyer/seller via two choices:
  1. INPUT: 32-dim role-agnostic feature vector (no role flag). Every feature
     is framed in "own" / "opponent" terms, so it has the same MEANING
     regardless of which role I play.
  2. OUTPUT: predict a log-symmetric "extraction" e ∈ [0, ∞)
     = how much surplus I want, in log-units of my own_value.
       - Buyer's price = own_value × exp(-e)
       - Seller's price = own_value × exp(+e)
     Same number, same interpretation — the policy never sees role. The two
     role prices are exact multiplicative inverses for a given e.

This guarantees the policy plays identically when faced with mirror-image
states. The role only matters at the price-conversion wrapper.

Training:
  - Each episode: random role for the learner; opponent is uniformly random mimic
  - Reward = own surplus on agreement (zero on no-deal)
  - Role rotation balanced: alternate strictly buyer/seller to avoid sampling bias
"""
from __future__ import annotations

import argparse
import math
import random
import sys
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
OBS_DIM = 32   # v5-sym schema: 32 role-agnostic features (no role flag)
MIMIC_OPPONENTS = ["Mimic-GPT-5.4", "Mimic-Grok", "Mimic-Opus", "Mimic-Pro", "Mimic-Llama"]


def _symmetric_obs(role: str, own_value: float, own_target: float | None,
                   turn_history: list[dict], standing_price: float | None,
                   turn_index: int, message_limit: int) -> list[float]:
    """Build 31-dim symmetric feature vector (already role-agnostic in v4-sym)."""
    return build_feature_vector(
        role=role, own_private_value=own_value, own_target_price=own_target,
        turn_history=turn_history, standing_price=standing_price,
        turn_index=turn_index, message_limit=message_limit,
    )


class SymmetricPolicy(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.accept_head = nn.Linear(hidden, 1)
        self.extraction_head = nn.Linear(hidden, 1)
        self.extraction_log_std = nn.Parameter(torch.full((1,), -1.0))

    def forward(self, obs):
        h = self.trunk(obs)
        accept_logit = self.accept_head(h).squeeze(-1)
        # Log-symmetric extraction: squash output via sigmoid to keep the
        # Gaussian mean in [0, 1.5] log-units. This covers the full math-tier
        # range (T4 opens at e=0.7 → price ratio exp(0.7)≈2.0; cap of 1.5
        # corresponds to price ratio exp(1.5)≈4.5, giving PPO headroom to
        # outperform T4 by being either more or less aggressive than e=0.7).
        # Cap prevents PPO from collapsing to e=∞ (rare big lucky deals).
        ext_mean = torch.sigmoid(self.extraction_head(h).squeeze(-1)) * 1.5
        log_std = self.extraction_log_std.expand_as(ext_mean).clamp(-4.0, 0.0)
        return accept_logit, ext_mean, log_std

    def sample_action(self, obs):
        a_logit, e_mean, e_log_std = self(obs)
        ber = Bernoulli(logits=a_logit)
        accept = ber.sample()
        a_lp = ber.log_prob(accept)
        gauss = Normal(e_mean, e_log_std.exp())
        ext = gauss.rsample()
        e_lp = gauss.log_prob(ext)
        return accept, ext, a_lp + e_lp

    def log_prob(self, obs, accept, ext):
        a_logit, e_mean, e_log_std = self(obs)
        ber = Bernoulli(logits=a_logit)
        gauss = Normal(e_mean, e_log_std.exp())
        return ber.log_prob(accept) + gauss.log_prob(ext)


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


def _extraction_to_price(role: str, own_value: float, extraction: float) -> int:
    """Log-symmetric extraction → raw price.

    Buyer:  price = own_value · exp(-e)   (price ≤ own_value)
    Seller: price = own_value · exp(+e)   (price ≥ own_value)
    """
    import math as _m
    ext = max(0.0, min(5.0, float(extraction)))  # log-space cap
    if role == "buyer":
        return max(1, int(round(own_value * _m.exp(-ext))))
    return int(round(own_value * _m.exp(ext)))


def _run_episode(policy: SymmetricPolicy, opponent_alias: str, *, seed: int,
                 learner_role: str, message_limit: int = 10):
    env = TravelGameEnv({
        "mode": "buyer_seller_negotiation",
        "selected_models": ["GPT-5.4", "GPT-5.4", "GPT-5.4"],
        "seed": seed,
        "negotiation_message_limit": message_limit,
    })
    env.reset(seed=seed)
    buyer = env.world["buyer_true"]
    seller = env.world["seller_true"]

    obs_buf, acc_buf, ext_buf, lp_buf, rew_buf, done_buf = [], [], [], [], [], []
    turns = []
    standing_price = None

    # Opener determined by seed parity (matches server.py + eval).
    opener_role = "buyer" if (int(seed) % 2 == 1) else "seller"
    other_role = "seller" if opener_role == "buyer" else "buyer"

    def _own_value_target(role: str):
        if role == "buyer":
            return float(buyer.budget), float(buyer.target_price)
        return float(seller.baseline_value), float(getattr(seller, "target_price", 0) or seller.asking_price)

    # Opener acts (learner OR opponent depending on role assignment)
    op_own, op_target = _own_value_target(opener_role)
    if learner_role == opener_role:
        x = _symmetric_obs(opener_role, op_own, op_target, [], None, 0, message_limit)
        xt = torch.tensor([x], dtype=torch.float32)
        with torch.no_grad():
            accept_t, ext_t, lp_t = policy.sample_action(xt)
        price = _extraction_to_price(opener_role, op_own, float(ext_t.item()))
        if opener_role == "buyer":
            price = max(1, min(int(buyer.budget), price))
        else:
            price = max(int(seller.baseline_value), price)
        turns.append(NegotiationTurnAction(speaker=opener_role, proposed_price=price, message_text=""))
        standing_price = float(price)
        obs_buf.append(x); acc_buf.append(float(accept_t.item())); ext_buf.append(float(ext_t.item()))
        lp_buf.append(float(lp_t.item())); rew_buf.append(0.0); done_buf.append(False)
    else:
        mres = negotiation_mimic_action(opponent_alias, role=opener_role,
                                        own_private_value=op_own, own_target_price=op_target,
                                        turn_history=[], standing_price=None,
                                        turn_index=0, message_limit=message_limit)
        opening = int(mres["proposed_price"])
        if opener_role == "buyer":
            opening = max(1, min(int(buyer.budget), opening))
        else:
            opening = max(int(seller.baseline_value), opening)
        turns.append(NegotiationTurnAction(speaker=opener_role, proposed_price=opening, message_text=""))
        standing_price = float(opening)

    agreed = None
    for turn_idx in range(1, message_limit):
        role = opener_role if (turn_idx % 2 == 0) else other_role
        is_learner = (role == learner_role)
        own_value = float(seller.baseline_value) if role == "seller" else float(buyer.budget)
        own_target = float(getattr(seller, "target_price", 0) or seller.asking_price) if role == "seller" else float(buyer.target_price)
        history = [{"speaker": t.speaker, "price": int(t.proposed_price)} for t in turns]

        if is_learner:
            x = _symmetric_obs(role, own_value, own_target, history, standing_price, turn_idx, message_limit)
            xt = torch.tensor([x], dtype=torch.float32)
            with torch.no_grad():
                accept_t, ext_t, lp_t = policy.sample_action(xt)
            a = float(accept_t.item()); e = float(ext_t.item()); lp = float(lp_t.item())
            legal = ((role == "buyer" and standing_price <= own_value) or
                     (role == "seller" and standing_price >= own_value))
            if a > 0.5 and legal:
                agreed = int(standing_price)
                turns.append(NegotiationTurnAction(speaker=role, proposed_price=int(standing_price), message_text=""))
                surplus = (own_value - agreed) if role == "buyer" else (agreed - own_value)
                obs_buf.append(x); acc_buf.append(a); ext_buf.append(e); lp_buf.append(lp)
                rew_buf.append(float(surplus)); done_buf.append(True)
                break
            price = _extraction_to_price(role, own_value, e)
            if role == "buyer":
                price = max(1, min(int(buyer.budget), price))
            else:
                price = max(int(seller.baseline_value), price)
            turns.append(NegotiationTurnAction(speaker=role, proposed_price=price, message_text=""))
            standing_price = float(price)
            obs_buf.append(x); acc_buf.append(a); ext_buf.append(e); lp_buf.append(lp)
            rew_buf.append(0.0); done_buf.append(False)
        else:
            mres = negotiation_mimic_action(opponent_alias, role=role,
                                            own_private_value=own_value, own_target_price=own_target,
                                            turn_history=history, standing_price=standing_price,
                                            turn_index=turn_idx, message_limit=message_limit)
            if mres["action"] == "accept":
                legal_opp = ((role == "buyer" and standing_price <= own_value) or
                             (role == "seller" and standing_price >= own_value))
                if legal_opp:
                    agreed = int(standing_price)
                    turns.append(NegotiationTurnAction(speaker=role, proposed_price=int(standing_price), message_text=""))
                    break
            new_price = int(mres["proposed_price"])
            if role == "buyer":
                new_price = max(1, min(int(buyer.budget), new_price))
            else:
                new_price = max(int(seller.baseline_value), new_price)
            turns.append(NegotiationTurnAction(speaker=role, proposed_price=new_price, message_text=""))
            standing_price = float(new_price)

    if obs_buf and not done_buf[-1]:
        done_buf[-1] = True
    return {"obs": obs_buf, "accepts": acc_buf, "exts": ext_buf, "log_probs": lp_buf,
            "rewards": rew_buf, "dones": done_buf, "agreed": agreed, "role": learner_role}


def compute_gae(rewards, values, dones, *, gamma=0.99, lam=0.95):
    advs = [0.0] * len(rewards); last_adv = 0.0; next_v = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        nv = 0.0 if dones[t] else next_v
        delta = rewards[t] + gamma * nv - values[t]
        last_adv = delta + gamma * lam * (0.0 if dones[t] else last_adv)
        advs[t] = last_adv; next_v = values[t]
    returns = [a + v for a, v in zip(advs, values)]
    return advs, returns


def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    policy = SymmetricPolicy()
    value_net = ValueNet()
    opt_p = torch.optim.Adam(policy.parameters(), lr=args.lr, weight_decay=1e-5)
    opt_v = torch.optim.Adam(value_net.parameters(), lr=args.lr_value, weight_decay=1e-5)

    reward_hist = []
    role_rewards = {"buyer": [], "seller": []}
    best_avg = -math.inf

    for ep in range(args.episodes):
        # ALTERNATE roles strictly so neither side is under-trained.
        role = "buyer" if ep % 2 == 0 else "seller"
        opponent = random.choice(MIMIC_OPPONENTS)
        roll = _run_episode(policy, opponent, seed=ep + 1, learner_role=role,
                            message_limit=args.message_limit)
        if not roll["obs"]:
            continue
        total_reward = sum(roll["rewards"])
        reward_hist.append(total_reward)
        role_rewards[role].append(total_reward)

        obs_t = torch.tensor(roll["obs"], dtype=torch.float32)
        with torch.no_grad():
            values = value_net(obs_t).tolist()
        adv, ret = compute_gae(roll["rewards"], values, roll["dones"], gamma=args.gamma, lam=args.gae_lambda)
        adv_t = torch.tensor(adv, dtype=torch.float32)
        ret_t = torch.tensor(ret, dtype=torch.float32)
        acc_t = torch.tensor(roll["accepts"], dtype=torch.float32)
        ext_t = torch.tensor(roll["exts"], dtype=torch.float32)
        oldlp_t = torch.tensor(roll["log_probs"], dtype=torch.float32)
        if adv_t.numel() > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(args.ppo_epochs):
            new_lp = policy.log_prob(obs_t, acc_t, ext_t)
            ratio = (new_lp - oldlp_t).exp()
            cr = ratio.clamp(1.0 - args.clip_eps, 1.0 + args.clip_eps)
            policy_loss = -torch.min(ratio * adv_t, cr * adv_t).mean()
            v_loss = F.mse_loss(value_net(obs_t), ret_t)
            opt_p.zero_grad(); policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5); opt_p.step()
            opt_v.zero_grad(); v_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5); opt_v.step()

        if (ep + 1) % args.log_every == 0:
            recent = reward_hist[-200:]
            avg = sum(recent) / max(1, len(recent))
            br = role_rewards["buyer"][-100:]
            sr = role_rewards["seller"][-100:]
            b_avg = sum(br)/max(1,len(br)); s_avg = sum(sr)/max(1,len(sr))
            print(f"  ep {ep+1:5d}/{args.episodes}  avg_reward={avg:+.2f}  "
                  f"buyer_avg={b_avg:+.2f}  seller_avg={s_avg:+.2f}  agreed={roll['agreed'] is not None}", flush=True)
            if avg > best_avg:
                best_avg = avg
                torch.save({"policy": policy.state_dict(), "value": value_net.state_dict(),
                            "best_avg": best_avg, "obs_dim": OBS_DIM,
                            "symmetric": True},
                           MODELS_DIR / f"negotiation_rl_{args.label}.best.pt")

    torch.save({"policy": policy.state_dict(), "value": value_net.state_dict(),
                "final_avg": sum(reward_hist[-200:]) / max(1, len(reward_hist[-200:])),
                "obs_dim": OBS_DIM, "symmetric": True},
               MODELS_DIR / f"negotiation_rl_{args.label}.pt")
    print(f"\nFinal avg = {sum(reward_hist[-200:])/max(1,len(reward_hist[-200:])):+.2f}, "
          f"best = {best_avg:+.2f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=6000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-value", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--message-limit", type=int, default=10)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--label", default="v2-sym")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
