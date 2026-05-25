"""Train the Math-T5 PPO agent against the v7 mimic ensemble.

Usage:
    python -m simulation.train_rl_t5 --episodes 5000

The trained policy is saved to ``simulation/models/rl/t5_ppo.pt`` and is
loaded by ``simulation/rl_agent.py:rl_bid`` for inference during auctions.

PPO with:
  - Clipped surrogate objective
  - GAE advantage estimation
  - Shared input normalisation buffers (updated online from collected obs)
  - Random learner seat + random 4-of-5 mimic opponents each episode
  - Reward = paintings_won this transition (dense 0/1 per painting)
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from .rl_agent import DEFAULT_CKPT, MODELS_DIR, N_ACTIONS, OBS_DIM, PolicyNet, ValueNet
from .rl_env import MIMIC_ALIASES, RLAuctionEnv


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------

def collect_episode(
    env: RLAuctionEnv,
    policy: PolicyNet,
    value: ValueNet,
    *,
    learner_seat: int,
    opponent_aliases: list[str],
    device: torch.device,
) -> dict:
    obs, _ = env.reset(learner_seat=learner_seat, opponent_aliases=opponent_aliases)
    if env.done:
        return {
            "states": [], "actions": [], "log_probs": [],
            "rewards": [], "values": [], "final_wins": 0,
        }
    states: list[list[float]] = []
    actions: list[int] = []
    log_probs: list[float] = []
    rewards: list[float] = []
    values: list[float] = []
    while not env.done:
        s_tensor = torch.tensor([obs], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = policy(s_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            v = value(s_tensor)
        a_idx = int(action.item())
        next_obs, reward, done, _info = env.step(a_idx)
        states.append(obs)
        actions.append(a_idx)
        log_probs.append(float(log_prob.item()))
        rewards.append(float(reward))
        values.append(float(v.item()))
        if done:
            break
        obs = next_obs
    final_wins = env.bidders[env.learner_id].paintings_won if env.learner_id else 0
    return {
        "states": states, "actions": actions, "log_probs": log_probs,
        "rewards": rewards, "values": values, "final_wins": final_wins,
    }


def compute_gae(rewards: list[float], values: list[float], *, gamma: float = 0.99, lam: float = 0.95) -> tuple[list[float], list[float]]:
    """Generalised advantage estimation. Terminal value assumed 0."""
    advs: list[float] = []
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advs.insert(0, gae)
        next_value = values[t]
    returns = [a + v for a, v in zip(advs, values)]
    return advs, returns


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(
    policy: PolicyNet,
    value: ValueNet,
    optim_p: optim.Optimizer,
    optim_v: optim.Optimizer,
    batch: dict,
    *,
    clip_eps: float = 0.2,
    n_epochs: int = 4,
    batch_size: int = 64,
    entropy_coef: float = 0.01,
    device: torch.device = torch.device("cpu"),
) -> dict:
    states = torch.tensor(batch["states"], dtype=torch.float32, device=device)
    actions = torch.tensor(batch["actions"], dtype=torch.int64, device=device)
    old_log_probs = torch.tensor(batch["log_probs"], dtype=torch.float32, device=device)
    advs = torch.tensor(batch["advs"], dtype=torch.float32, device=device)
    returns = torch.tensor(batch["returns"], dtype=torch.float32, device=device)
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    n = len(states)
    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0}
    update_count = 0
    for _ in range(n_epochs):
        idx = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            if len(b) == 0:
                continue
            logits = policy(states[b])
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions[b])
            ratio = torch.exp(new_log_probs - old_log_probs[b])
            surr1 = ratio * advs[b]
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advs[b]
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss - entropy_coef * entropy
            optim_p.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optim_p.step()

            v_pred = value(states[b])
            value_loss = F.mse_loss(v_pred, returns[b])
            optim_v.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value.parameters(), 0.5)
            optim_v.step()

            with torch.no_grad():
                clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()
            metrics["policy_loss"] += float(policy_loss.item())
            metrics["value_loss"] += float(value_loss.item())
            metrics["entropy"] += float(entropy.item())
            metrics["clip_frac"] += float(clip_frac)
            update_count += 1
    if update_count > 0:
        for k in metrics:
            metrics[k] /= update_count
    return metrics


# ---------------------------------------------------------------------------
# Online observation normalisation
# ---------------------------------------------------------------------------

class RunningNorm:
    """Welford-style running mean/std for input normalisation."""

    def __init__(self, dim: int):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)
        self.count = 0

    def update(self, x: np.ndarray) -> None:
        # x can be (n, d). Update with batch.
        if x.ndim == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        if n == 0:
            return
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        total = self.count + n
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (n / total)
        new_M2 = self.M2 + batch_var * n + (delta ** 2) * self.count * n / total
        self.mean = new_mean
        self.M2 = new_M2
        self.count = total

    def std(self) -> np.ndarray:
        if self.count < 2:
            return np.ones_like(self.mean)
        return np.sqrt(self.M2 / max(1, self.count - 1)) + 1e-8

    def write_to(self, *modules: torch.nn.Module) -> None:
        m = torch.tensor(self.mean, dtype=torch.float32)
        s = torch.tensor(self.std(), dtype=torch.float32)
        for mod in modules:
            if hasattr(mod, "input_mean"):
                mod.input_mean.copy_(m)
            if hasattr(mod, "input_std"):
                mod.input_std.copy_(s)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="PPO training for Math-T5 against v7 mimics")
    p.add_argument("--episodes", type=int, default=5000, help="Total episodes to train for.")
    p.add_argument("--episodes-per-update", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num-paintings", type=int, default=12)
    p.add_argument("--start-budget", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--output", default=str(DEFAULT_CKPT))
    p.add_argument("--metrics-out", default=str(MODELS_DIR / "t5_training_metrics.jsonl"))
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    policy = PolicyNet(obs_dim=OBS_DIM, hidden=args.hidden, n_actions=N_ACTIONS).to(device)
    value = ValueNet(obs_dim=OBS_DIM, hidden=args.hidden).to(device)
    optim_p = optim.Adam(policy.parameters(), lr=args.lr)
    optim_v = optim.Adam(value.parameters(), lr=args.lr)

    env = RLAuctionEnv(num_paintings=args.num_paintings, start_budget=args.start_budget)
    norm = RunningNorm(OBS_DIM)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_path.open("a", encoding="utf-8")

    print(
        f"Training T5 PPO: episodes={args.episodes}  per_update={args.episodes_per_update}  "
        f"lr={args.lr}  hidden={args.hidden}  device=cpu"
    )
    win_history: list[int] = []
    t0 = time.time()

    eps_done = 0
    while eps_done < args.episodes:
        # Collect a batch of episodes
        all_states: list[list[float]] = []
        all_actions: list[int] = []
        all_log_probs: list[float] = []
        all_advs: list[float] = []
        all_returns: list[float] = []
        batch_wins: list[int] = []
        for _ in range(args.episodes_per_update):
            learner_seat = random.randint(0, env.NUM_BIDDERS - 1)
            opps = random.sample(MIMIC_ALIASES, env.NUM_BIDDERS - 1)
            ep = collect_episode(
                env, policy, value,
                learner_seat=learner_seat,
                opponent_aliases=opps,
                device=device,
            )
            if not ep["states"]:
                batch_wins.append(ep["final_wins"])
                continue
            advs, returns = compute_gae(
                ep["rewards"], ep["values"],
                gamma=args.gamma, lam=args.gae_lambda,
            )
            all_states.extend(ep["states"])
            all_actions.extend(ep["actions"])
            all_log_probs.extend(ep["log_probs"])
            all_advs.extend(advs)
            all_returns.extend(returns)
            batch_wins.append(ep["final_wins"])
        eps_done += args.episodes_per_update
        win_history.extend(batch_wins)
        if not all_states:
            continue
        # Update normalisation buffers from this batch and propagate to nets.
        norm.update(np.asarray(all_states, dtype=np.float64))
        norm.write_to(policy, value)
        update_metrics = ppo_update(
            policy, value, optim_p, optim_v,
            {
                "states": all_states, "actions": all_actions,
                "log_probs": all_log_probs, "advs": all_advs, "returns": all_returns,
            },
            clip_eps=args.clip_eps,
            n_epochs=args.ppo_epochs,
            batch_size=args.minibatch_size,
            entropy_coef=args.entropy_coef,
            device=device,
        )
        recent = win_history[-100:]
        avg_wins = sum(recent) / len(recent) if recent else 0.0
        elapsed = time.time() - t0
        row = {
            "episode": eps_done,
            "avg_wins_last100": round(avg_wins, 3),
            "policy_loss": round(update_metrics["policy_loss"], 4),
            "value_loss": round(update_metrics["value_loss"], 4),
            "entropy": round(update_metrics["entropy"], 4),
            "clip_frac": round(update_metrics["clip_frac"], 4),
            "elapsed_s": round(elapsed, 1),
        }
        metrics_file.write(json.dumps(row) + "\n")
        metrics_file.flush()
        print(
            f"[ep {eps_done:>5d}/{args.episodes}]  avg_wins={avg_wins:.2f}  "
            f"ploss={update_metrics['policy_loss']:+.3f}  vloss={update_metrics['value_loss']:.3f}  "
            f"H={update_metrics['entropy']:.3f}  clip={update_metrics['clip_frac']:.2f}  "
            f"({elapsed:.0f}s)"
        )
        if eps_done % args.checkpoint_every == 0 or eps_done >= args.episodes:
            torch.save(
                {
                    "model_state": policy.state_dict(),
                    "value_state": value.state_dict(),
                    "obs_dim": OBS_DIM,
                    "hidden": args.hidden,
                    "n_actions": N_ACTIONS,
                    "episodes_trained": eps_done,
                    "norm_mean": norm.mean.tolist(),
                    "norm_std": norm.std().tolist(),
                },
                output_path,
            )
        if eps_done % args.eval_every == 0:
            # Held-out evaluation: deterministic policy (argmax) vs fixed mimic rotations.
            eval_wins: list[int] = []
            policy.eval()
            for k in range(args.eval_episodes):
                ls = k % env.NUM_BIDDERS
                opps = [a for i, a in enumerate(MIMIC_ALIASES) if i != ls]
                obs, _ = env.reset(learner_seat=ls, opponent_aliases=opps)
                while not env.done:
                    s_tensor = torch.tensor([obs], dtype=torch.float32)
                    with torch.no_grad():
                        a = int(torch.argmax(policy(s_tensor), dim=-1).item())
                    obs, _r, done, _info = env.step(a)
                    if done:
                        break
                eval_wins.append(env.bidders[env.learner_id].paintings_won)
            policy.train()
            avg_eval = sum(eval_wins) / max(1, len(eval_wins))
            share = avg_eval / args.num_paintings * 100.0
            print(
                f"  [eval n={args.eval_episodes}]  avg_wins={avg_eval:.2f}  "
                f"share={share:.1f}%  (uniform baseline = {100.0 / env.NUM_BIDDERS:.1f}%)"
            )

    metrics_file.close()
    print(f"\nTraining complete. Policy saved to {output_path}.")
    print(f"Use Math-T5 in a loadout to deploy. Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
