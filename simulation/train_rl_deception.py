"""PPO training for Math-T5 in the Deception Competition (Option R).

Continuous-action policy: outputs a Gaussian over the 5-vector claim (mean +
log_std), sampled and squashed to [0, 1] via sigmoid. Trained against the v1
mimic pool by default. Optional BC warm-start from T4 trajectories.

Usage:
    python -m simulation.train_rl_deception --episodes 3000 --label deception-v1
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
from torch.distributions import Normal

from .rl_deception_env import (
    DEFAULT_MIMIC_ALIASES,
    RLDeceptionEnv,
    collect_t4_imitation_episode,
)


MODELS_DIR = Path(__file__).parent / "models" / "rl_deception"
DEFAULT_CKPT = MODELS_DIR / "t5d_ppo.pt"

OBS_DIM = 19
ACTION_DIM = 5


# ── Policy + value networks ────────────────────────────────────────────────

class ContinuousPolicy(nn.Module):
    """5-D Gaussian policy with state-independent log_std."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 64, action_dim: int = ACTION_DIM):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))   # initial std ≈ 0.6

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std.expand_as(mean).clamp(-4.0, 1.0)
        return mean, log_std

    def sample_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action; return (sigmoid(action), log_prob_of_pre_squash)."""
        mean, log_std = self(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        u = dist.rsample()
        logp = dist.log_prob(u).sum(dim=-1)
        squashed = torch.sigmoid(u)
        # Squash correction: -log|d sigmoid/du| = -log(sigmoid(u)*(1-sigmoid(u)))
        logp = logp - (F.softplus(u) + F.softplus(-u)).sum(dim=-1)
        return squashed, logp

    def log_prob_for(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """log_prob of a previously-sampled squashed action."""
        mean, log_std = self(obs)
        std = log_std.exp()
        action_clamped = action.clamp(1e-5, 1.0 - 1e-5)
        u = torch.log(action_clamped) - torch.log1p(-action_clamped)
        dist = Normal(mean, std)
        logp = dist.log_prob(u).sum(dim=-1) - (F.softplus(u) + F.softplus(-u)).sum(dim=-1)
        return logp


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ── Episode rollout ────────────────────────────────────────────────────────

def rollout_episode(
    policy: ContinuousPolicy,
    env: RLDeceptionEnv,
    opponents: list[str],
    seed: int,
    *,
    deterministic: bool = False,
) -> dict:
    obs, _ = env.reset(learner_seat=0, opponent_aliases=opponents, seed=seed)
    obs_buf, act_buf, logp_buf, rew_buf, done_buf = [], [], [], [], []
    with torch.no_grad():
        while not env.done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            if deterministic:
                mean, _ = policy(obs_t)
                action = torch.sigmoid(mean).squeeze(0)
                logp = torch.tensor(0.0)
            else:
                squashed, lp = policy.sample_action(obs_t)
                action = squashed.squeeze(0)
                logp = lp.squeeze(0)
            claim = [round(float(v), 2) for v in action.tolist()]
            obs_next, r, done, _ = env.step(claim)
            obs_buf.append(obs)
            act_buf.append(action.tolist())
            logp_buf.append(float(logp.item()))
            rew_buf.append(float(r))
            done_buf.append(bool(done))
            obs = obs_next
    return {
        "obs": obs_buf,
        "actions": act_buf,
        "log_probs": logp_buf,
        "rewards": rew_buf,
        "dones": done_buf,
        "total_reward": sum(rew_buf),
        "wins": env.agents[env.learner_id].win_count,
    }


def compute_gae(rewards: list[float], values: list[float], dones: list[bool], *, gamma: float, lam: float) -> tuple[list[float], list[float]]:
    advantages = [0.0] * len(rewards)
    last_adv = 0.0
    next_value = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        next_v = 0.0 if dones[t] else next_value
        delta = rewards[t] + gamma * next_v - values[t]
        last_adv = delta + gamma * lam * (0.0 if dones[t] else last_adv)
        advantages[t] = last_adv
        next_value = values[t]
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


# ── Training loop ──────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    runs_dir = MODELS_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = int(time.time())
    metrics_path = runs_dir / f"t5d_{run_id}.jsonl"
    status_path = MODELS_DIR / f"t5d_training_status_{run_id}.json"

    policy = ContinuousPolicy(hidden=args.hidden)
    value_net = ValueNet(hidden=args.hidden)
    opt_p = torch.optim.Adam(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_v = torch.optim.Adam(value_net.parameters(), lr=args.lr_value, weight_decay=args.weight_decay)

    label = args.label or "default"

    # ── Optional BC warm-start ────────────────────────────────────────────
    if args.bc_warmstart_episodes > 0:
        print(f"BC warm-start: collecting {args.bc_warmstart_episodes} T4 episodes...")
        env_bc = RLDeceptionEnv(num_rounds=args.num_rounds)
        bc_obs: list[list[float]] = []
        bc_actions: list[list[float]] = []
        for ep in range(args.bc_warmstart_episodes):
            samples = collect_t4_imitation_episode(env_bc, DEFAULT_MIMIC_ALIASES[:4], seed=ep)
            for o, a in samples:
                bc_obs.append(o)
                bc_actions.append(a)
        bc_obs_t = torch.tensor(bc_obs, dtype=torch.float32)
        bc_act_t = torch.tensor(bc_actions, dtype=torch.float32).clamp(1e-5, 1.0 - 1e-5)
        print(f"BC dataset: {len(bc_obs)} (obs, action) pairs. Pretraining {args.bc_epochs} epochs...")
        for ep in range(args.bc_epochs):
            mean, log_std = policy(bc_obs_t)
            # We supervise the SQUASHED action: target = sigmoid(mean) ≈ bc_act_t
            pred = torch.sigmoid(mean)
            loss = F.mse_loss(pred, bc_act_t)
            opt_p.zero_grad()
            loss.backward()
            opt_p.step()
            if (ep + 1) % max(1, args.bc_epochs // 5) == 0:
                print(f"  BC epoch {ep + 1}/{args.bc_epochs}  mse={loss.item():.4f}")

    # ── PPO loop ──────────────────────────────────────────────────────────
    print(f"Training PPO for {args.episodes} episodes; opponents = mimic pool")
    env = RLDeceptionEnv(num_rounds=args.num_rounds)
    opponents = DEFAULT_MIMIC_ALIASES[:4]

    reward_history: list[float] = []
    best_avg = -math.inf

    for ep in range(args.episodes):
        ro = rollout_episode(policy, env, opponents, seed=ep + 1)
        reward_history.append(ro["total_reward"])

        # Compute value estimates + GAE
        obs_t = torch.tensor(ro["obs"], dtype=torch.float32)
        with torch.no_grad():
            values = value_net(obs_t).tolist()
        advantages, returns = compute_gae(
            ro["rewards"], values, ro["dones"], gamma=args.gamma, lam=args.gae_lambda
        )
        adv_t = torch.tensor(advantages, dtype=torch.float32)
        ret_t = torch.tensor(returns, dtype=torch.float32)
        act_t = torch.tensor(ro["actions"], dtype=torch.float32)
        old_logp_t = torch.tensor(ro["log_probs"], dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # PPO update (a few epochs of full-batch since episodes are short)
        for _ in range(args.ppo_epochs):
            new_logp = policy.log_prob_for(obs_t, act_t)
            ratio = (new_logp - old_logp_t).exp()
            clip_ratio = ratio.clamp(1.0 - args.clip_eps, 1.0 + args.clip_eps)
            policy_loss = -torch.min(ratio * adv_t, clip_ratio * adv_t).mean()

            # Value loss
            value_pred = value_net(obs_t)
            value_loss = F.mse_loss(value_pred, ret_t)

            # Entropy bonus (use log_std for explicit entropy estimate)
            _, log_std = policy(obs_t)
            entropy_bonus = log_std.mean() + 0.5 * math.log(2.0 * math.pi * math.e)

            if args.entropy_decay and args.episodes > 0:
                ent_coef = args.entropy_coef * max(0.0, 1.0 - ep / args.episodes)
            else:
                ent_coef = args.entropy_coef

            opt_p.zero_grad()
            (policy_loss - ent_coef * entropy_bonus).backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt_p.step()

            opt_v.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
            opt_v.step()

        # Periodic logging
        if (ep + 1) % args.log_every == 0 or ep == 0:
            recent = reward_history[-100:]
            avg = sum(recent) / len(recent) if recent else 0.0
            print(f"  ep {ep + 1:5d}/{args.episodes}  reward_last100={avg:+.3f}  "
                  f"wins={ro['wins']:2d}  p_loss={float(policy_loss.item()):+.4f}  "
                  f"v_loss={float(value_loss.item()):.4f}")
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "episode": ep + 1, "label": label,
                    "avg_reward_last100": avg, "wins": ro["wins"],
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                }) + "\n")
            if avg > best_avg:
                best_avg = avg
                best_path = DEFAULT_CKPT.with_name(f"t5d_ppo_{label}.best.pt")
                torch.save({"policy": policy.state_dict(), "value": value_net.state_dict(),
                            "label": label, "best_avg_reward": best_avg,
                            "hidden": args.hidden}, best_path)

    # Final checkpoint
    out_path = DEFAULT_CKPT.with_name(f"t5d_ppo_{label}.pt")
    torch.save({"policy": policy.state_dict(), "value": value_net.state_dict(),
                "label": label, "hidden": args.hidden,
                "final_avg_reward": sum(reward_history[-100:]) / max(1, len(reward_history[-100:]))}, out_path)
    status_path.write_text(json.dumps({
        "run_id": run_id, "label": label, "episodes": args.episodes,
        "final_avg_reward": float(sum(reward_history[-100:]) / max(1, len(reward_history[-100:]))),
        "best_avg_reward": float(best_avg),
    }, indent=2), encoding="utf-8")
    print(f"Trained {args.episodes} episodes. Final avg reward (last 100) = "
          f"{sum(reward_history[-100:]) / max(1, len(reward_history[-100:])):.3f}")
    print(f"Saved to {out_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Status: {status_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=1500)
    p.add_argument("--num-rounds", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-value", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--bc-warmstart-episodes", type=int, default=20)
    p.add_argument("--bc-epochs", type=int, default=50)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label", default="v1")
    p.add_argument("--hidden", type=int, default=64,
                   help="Hidden width for policy + value MLPs.")
    p.add_argument("--entropy-decay", action="store_true",
                   help="Linearly decay entropy_coef from initial to 0 across training.")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
