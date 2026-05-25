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
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Force UTF-8 stdout/stderr so non-ASCII chars in log lines (→, ★, ⚠) don't
# crash print() on Windows cp1252 consoles. Same fix as step_worker.py.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

from .rl_agent import DEFAULT_CKPT, HybridPolicyNet, MODELS_DIR, N_ACTIONS, OBS_DIM, PolicyNet, ValueNet
from .rl_env import MIMIC_ALIASES, RLAuctionEnv, collect_t4_imitation_episode


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


def collect_episode_hybrid(
    env: RLAuctionEnv,
    policy: HybridPolicyNet,
    value: ValueNet,
    *,
    learner_seat: int,
    opponent_aliases: list[str],
    device: torch.device,
) -> dict:
    """Episode collection for the continuous (hybrid) action space.

    Returns extra per-step data: discrete action, bid fraction (NaN if PASS).
    """
    obs, _ = env.reset(learner_seat=learner_seat, opponent_aliases=opponent_aliases)
    if env.done:
        return {
            "states": [], "discrete_actions": [], "bid_fractions": [], "log_probs": [],
            "rewards": [], "values": [], "final_wins": 0,
        }
    states, d_actions, bid_fracs, log_probs, rewards, values = [], [], [], [], [], []
    while not env.done:
        s_tensor = torch.tensor([obs], dtype=torch.float32, device=device)
        with torch.no_grad():
            d_logits, bid_mean, bid_log_std = policy(s_tensor)
            d_dist = Categorical(logits=d_logits)
            d_action = d_dist.sample()
            d_log_prob = d_dist.log_prob(d_action)
            if int(d_action.item()) == 1:
                # RAISE: sample bid_fraction in [0, 1]
                c_dist = Normal(bid_mean, bid_log_std.exp())
                bid_f = c_dist.sample().clamp(0.0, 1.0)
                c_log_prob = c_dist.log_prob(bid_f)
                total_log_prob = d_log_prob + c_log_prob
                bid_frac_val = float(bid_f.item())
            else:
                total_log_prob = d_log_prob
                bid_frac_val = float("nan")
            v = value(s_tensor)
        next_obs, reward, done, _info = env.step_continuous(int(d_action.item()), bid_frac_val)
        states.append(obs)
        d_actions.append(int(d_action.item()))
        bid_fracs.append(bid_frac_val)
        log_probs.append(float(total_log_prob.item()))
        rewards.append(float(reward))
        values.append(float(v.item()))
        if done:
            break
        obs = next_obs
    final_wins = env.bidders[env.learner_id].paintings_won if env.learner_id else 0
    return {
        "states": states, "discrete_actions": d_actions, "bid_fractions": bid_fracs,
        "log_probs": log_probs, "rewards": rewards, "values": values, "final_wins": final_wins,
    }


def ppo_update_hybrid(
    policy: HybridPolicyNet,
    value: ValueNet,
    optim_p: optim.Optimizer,
    optim_v: optim.Optimizer,
    batch: dict,
    *,
    clip_eps: float = 0.2,
    n_epochs: int = 4,
    batch_size: int = 64,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    value_clip_eps: float | None = None,
    target_kl: float | None = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """PPO update for the hybrid action space. log_prob includes both the
    discrete and continuous components (continuous part is masked to 0 for PASS).
    """
    states = torch.tensor(batch["states"], dtype=torch.float32, device=device)
    d_actions = torch.tensor(batch["discrete_actions"], dtype=torch.int64, device=device)
    bid_fracs_raw = torch.tensor([0.0 if math.isnan(b) else b for b in batch["bid_fractions"]],
                                  dtype=torch.float32, device=device)
    is_raise = (d_actions == 1)
    old_log_probs = torch.tensor(batch["log_probs"], dtype=torch.float32, device=device)
    advs = torch.tensor(batch["advs"], dtype=torch.float32, device=device)
    returns = torch.tensor(batch["returns"], dtype=torch.float32, device=device)
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    n = states.shape[0]
    if value_clip_eps is not None:
        with torch.no_grad():
            old_values = value(states)
    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
               "clip_frac": 0.0, "approx_kl": 0.0, "kl_early_stops": 0.0}
    update_count = 0
    for _epoch in range(n_epochs):
        idx = torch.randperm(n, device=device)
        epoch_kl_total = 0.0
        epoch_batches = 0
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            if len(b) == 0:
                continue
            d_logits, bid_mean, bid_log_std = policy(states[b])
            d_dist = Categorical(logits=d_logits)
            d_new_lp = d_dist.log_prob(d_actions[b])
            c_dist = Normal(bid_mean, bid_log_std.exp())
            c_new_lp = c_dist.log_prob(bid_fracs_raw[b])
            # Mask continuous log-prob to 0 when action was PASS.
            c_new_lp = c_new_lp * is_raise[b].float()
            new_log_probs = d_new_lp + c_new_lp
            ratio = torch.exp(new_log_probs - old_log_probs[b])
            surr1 = ratio * advs[b]
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advs[b]
            policy_loss = -torch.min(surr1, surr2).mean()
            # Combined entropy: discrete entropy + (per-state) continuous entropy * is_raise
            ent_discrete = d_dist.entropy().mean()
            ent_cont = (c_dist.entropy() * is_raise[b].float()).mean()
            entropy = ent_discrete + ent_cont
            loss = policy_loss - entropy_coef * entropy
            optim_p.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optim_p.step()

            v_pred = value(states[b])
            if value_clip_eps is not None:
                v_old = old_values[b]
                v_clip = v_old + torch.clamp(v_pred - v_old, -value_clip_eps, value_clip_eps)
                vl_unclipped = (v_pred - returns[b]) ** 2
                vl_clipped = (v_clip - returns[b]) ** 2
                value_loss = 0.5 * torch.max(vl_unclipped, vl_clipped).mean()
            else:
                value_loss = F.mse_loss(v_pred, returns[b])
            optim_v.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value.parameters(), max_grad_norm)
            optim_v.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - (new_log_probs - old_log_probs[b])).mean().item()
                clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()
            metrics["policy_loss"] += float(policy_loss.item())
            metrics["value_loss"] += float(value_loss.item())
            metrics["entropy"] += float(entropy.item())
            metrics["clip_frac"] += float(clip_frac)
            metrics["approx_kl"] += float(approx_kl)
            epoch_kl_total += float(approx_kl)
            epoch_batches += 1
            update_count += 1
        if target_kl is not None and epoch_batches > 0:
            mean_kl = epoch_kl_total / epoch_batches
            if mean_kl > 1.5 * target_kl:
                metrics["kl_early_stops"] += 1.0
                break
    if update_count > 0:
        for k in ("policy_loss", "value_loss", "entropy", "clip_frac", "approx_kl"):
            metrics[k] /= update_count
    return metrics


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
    max_grad_norm: float = 0.5,
    value_clip_eps: float | None = None,
    target_kl: float | None = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    states = torch.tensor(batch["states"], dtype=torch.float32, device=device)
    actions = torch.tensor(batch["actions"], dtype=torch.int64, device=device)
    old_log_probs = torch.tensor(batch["log_probs"], dtype=torch.float32, device=device)
    advs = torch.tensor(batch["advs"], dtype=torch.float32, device=device)
    returns = torch.tensor(batch["returns"], dtype=torch.float32, device=device)
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    n = len(states)
    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
               "clip_frac": 0.0, "approx_kl": 0.0, "kl_early_stops": 0.0}
    # Cache old value predictions for value clipping (computed before any update).
    if value_clip_eps is not None:
        with torch.no_grad():
            old_values = value(states)
    update_count = 0
    epoch_count = 0
    for _epoch in range(n_epochs):
        idx = torch.randperm(n, device=device)
        epoch_kl_total = 0.0
        epoch_batches = 0
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
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optim_p.step()

            v_pred = value(states[b])
            if value_clip_eps is not None:
                # PPO value clipping: pred is allowed to move from old_pred by
                # at most ±value_clip_eps; we use the WORSE of the clipped and
                # unclipped losses (matches the policy clip).
                v_old = old_values[b]
                v_clipped = v_old + torch.clamp(v_pred - v_old, -value_clip_eps, value_clip_eps)
                v_loss_unclipped = (v_pred - returns[b]) ** 2
                v_loss_clipped = (v_clipped - returns[b]) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                value_loss = F.mse_loss(v_pred, returns[b])
            optim_v.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value.parameters(), max_grad_norm)
            optim_v.step()

            with torch.no_grad():
                # Schulman's approx_kl = E[(ratio - 1) - log(ratio)] ≈ E[KL].
                approx_kl = ((ratio - 1.0) - (new_log_probs - old_log_probs[b])).mean().item()
                clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()
            metrics["policy_loss"] += float(policy_loss.item())
            metrics["value_loss"] += float(value_loss.item())
            metrics["entropy"] += float(entropy.item())
            metrics["clip_frac"] += float(clip_frac)
            metrics["approx_kl"] += float(approx_kl)
            epoch_kl_total += float(approx_kl)
            epoch_batches += 1
            update_count += 1
        epoch_count += 1
        # KL early stop within the inner loop: if the average KL across this
        # epoch's minibatches exceeded the target, abort remaining epochs.
        if target_kl is not None and epoch_batches > 0:
            mean_epoch_kl = epoch_kl_total / epoch_batches
            if mean_epoch_kl > 1.5 * target_kl:
                metrics["kl_early_stops"] += 1.0
                break
    if update_count > 0:
        for k in ("policy_loss", "value_loss", "entropy", "clip_frac", "approx_kl"):
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
    # Learning rates. Use --lr to set both, or override individually.
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-policy", type=float, default=None, help="Override policy LR (defaults to --lr).")
    p.add_argument("--lr-value", type=float, default=None, help="Override value-net LR (defaults to --lr).")
    p.add_argument("--lr-schedule", choices=["constant", "linear"], default="linear",
                   help="LR schedule. 'linear' decays from initial to 0 over --episodes.")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.05,
                   help="Initial entropy coefficient. Was 0.01 in v1 — bumped because v1 collapsed entropy by ep 500.")
    p.add_argument("--entropy-coef-final", type=float, default=0.005,
                   help="Final entropy coef after linear decay (set equal to --entropy-coef to disable decay).")
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=None,
                   help="If set, abort remaining PPO epochs when batch approx_kl > 1.5×target_kl. "
                        "Typical values: 0.01-0.03. Off (None) by default.")
    p.add_argument("--value-clip-eps", type=float, default=None,
                   help="PPO-style value clip: limit value updates to ±this from the cached "
                        "pre-update prediction. Off (None) by default. Try 0.2 if value loss diverges.")
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="L2 regularization (Adam weight_decay). 0.0 disables.")
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--num-collect-workers", type=int, default=1,
                   help="Threads used to collect episodes in parallel each update batch. "
                        "1 = serial. >1 speeds up wall-clock at the cost of more memory.")
    p.add_argument("--opponent-pool", default="mimics",
                   help="Comma-separated alias list to sample 4 opponents from each episode. "
                        "Default 'mimics' = the 5 v7 mimic aliases. Use 'T5-frozen:<ckpt_path>' "
                        "to include a frozen RL policy. Example for pure self-play: "
                        "'T5-frozen:simulation/models/rl/t5_v1_frozen.pt' (will be replicated to 4 slots).")
    p.add_argument("--bc-warmstart-episodes", type=int, default=0,
                   help="Run N episodes with T4 in the learner seat, then supervised-pretrain "
                        "the policy on those (obs, T4-action) pairs before PPO. 0 disables (default).")
    p.add_argument("--bc-epochs", type=int, default=20,
                   help="Supervised epochs over the BC dataset.")
    p.add_argument("--bc-batch-size", type=int, default=256)
    p.add_argument("--bc-lr", type=float, default=1e-3)
    p.add_argument("--reward-norm", action="store_true",
                   help="Normalize advantages by running reward std. Off by default.")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--action-space", choices=["discrete", "continuous"], default="discrete",
                   help="discrete: {PASS, RAISE-min, JUMP}. continuous: hybrid PASS or "
                        "Gaussian-sampled bid amount in [min_next, budget]. "
                        "Continuous is a separate experiment — incompatible with BC warm-start "
                        "(T4's discrete labels can't supervise a Gaussian head).")
    p.add_argument("--num-paintings", type=int, default=12)
    p.add_argument("--start-budget", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-episodes", type=int, default=50)
    # Early stopping.
    p.add_argument("--early-stop-patience", type=int, default=4,
                   help="Stop training if held-out eval avg_wins hasn't improved in this many evals. 0 disables.")
    p.add_argument("--early-stop-min-delta", type=float, default=0.05,
                   help="Minimum eval-wins improvement to reset the patience counter.")
    p.add_argument("--output", default=str(DEFAULT_CKPT))
    p.add_argument("--best-output", default=None,
                   help="Optional path for the best-eval checkpoint. Defaults to '<output>.best.pt'.")
    p.add_argument("--metrics-out", default=None,
                   help="Per-run metrics JSONL. Default: models/rl/runs/t5_<unix_ts>.jsonl")
    p.add_argument("--label", default="",
                   help="Human-readable tag (e.g. 'bc-discrete') for the UI progress bar. "
                        "When set with default --output, also suffixes the checkpoint filename "
                        "so parallel runs don't overwrite each other.")
    args = p.parse_args()
    run_id = int(time.time())
    # Per-run metrics file so older runs survive as separate files and the
    # UI chart can overlay multiple recent runs with opacity fading.
    if args.metrics_out is None:
        args.metrics_out = str(MODELS_DIR / "runs" / f"t5_{run_id}.jsonl")
    # Per-run checkpoint paths when --label is provided + --output is default,
    # so two parallel runs don't clobber each other.
    if args.label and args.output == str(DEFAULT_CKPT):
        safe_label = "".join(c if c.isalnum() or c in "-_" else "-" for c in args.label)
        args.output = str(DEFAULT_CKPT.with_name(f"t5_ppo_{safe_label}.pt"))

    lr_policy_init = args.lr_policy if args.lr_policy is not None else args.lr
    lr_value_init = args.lr_value if args.lr_value is not None else args.lr

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    if args.action_space == "continuous":
        if int(args.bc_warmstart_episodes) > 0:
            print("WARN: --action-space=continuous incompatible with --bc-warmstart-episodes; ignoring BC.")
            args.bc_warmstart_episodes = 0
        policy = HybridPolicyNet(obs_dim=OBS_DIM, hidden=args.hidden).to(device)
        print("Action space: CONTINUOUS (hybrid PASS / Gaussian bid_fraction)")
    else:
        policy = PolicyNet(obs_dim=OBS_DIM, hidden=args.hidden, n_actions=N_ACTIONS).to(device)
        print("Action space: DISCRETE {PASS, RAISE-min, JUMP}")
    value = ValueNet(obs_dim=OBS_DIM, hidden=args.hidden).to(device)
    optim_p = optim.Adam(
        policy.parameters(),
        lr=lr_policy_init,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )
    optim_v = optim.Adam(
        value.parameters(),
        lr=lr_value_init,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )

    # Eval reuses a single env; collection workers each get a freshly-built
    # env per task to avoid concurrent-access collisions (two tasks sharing an
    # env would race on its mutable round state and inflate win counts).
    num_workers = max(1, int(args.num_collect_workers))
    env = RLAuctionEnv(num_paintings=args.num_paintings, start_budget=args.start_budget)
    norm = RunningNorm(OBS_DIM)
    pool = ThreadPoolExecutor(max_workers=num_workers) if num_workers > 1 else None
    def _make_env() -> RLAuctionEnv:
        return RLAuctionEnv(num_paintings=args.num_paintings, start_budget=args.start_budget)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_output_path = Path(args.best_output) if args.best_output else output_path.with_suffix(".best.pt")
    best_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    # Truncate (not append) so each run starts a fresh metrics history that the
    # UI chart modal can render without splicing concatenated past runs.
    metrics_file = metrics_path.open("w", encoding="utf-8")
    # Per-run sidecar status file. Multiple parallel trainings each write their
    # own status JSON; the UI lists all and renders side-by-side bars.
    status_path = MODELS_DIR / f"t5_training_status_{run_id}.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    auto_label = args.label or args.action_space
    run_label = auto_label[:48]

    def _write_status(*, eps_done: int, avg_wins: float, best_eval: float,
                      finished: bool, early_stopped: bool) -> None:
        try:
            status_path.write_text(json.dumps({
                "run_id": run_id,
                "label": run_label,
                "action_space": args.action_space,
                "episode": int(eps_done),
                "target_episodes": int(args.episodes),
                "avg_wins_last100": float(round(avg_wins, 3)),
                "best_eval_wins": (None if best_eval == float("-inf") else float(round(best_eval, 3))),
                "started_at": float(started_at),
                "updated_at": float(time.time()),
                "elapsed_s": float(round(time.time() - started_at, 1)),
                "finished": bool(finished),
                "early_stopped": bool(early_stopped),
                "checkpoint_path": str(output_path),
                "best_checkpoint_path": str(best_output_path),
                "pid": int(os.getpid()),
            }, indent=2), encoding="utf-8")
        except Exception:
            pass

    _write_status(eps_done=0, avg_wins=0.0, best_eval=float("-inf"), finished=False, early_stopped=False)

    print(
        f"Training T5 PPO: episodes={args.episodes}  per_update={args.episodes_per_update}  "
        f"lr_p={lr_policy_init}  lr_v={lr_value_init}  schedule={args.lr_schedule}  "
        f"H={args.entropy_coef}→{args.entropy_coef_final}  hidden={args.hidden}  "
        f"early_stop_patience={args.early_stop_patience}  device=cpu"
    )

    # ──────────────────────────────────────────────────────────────────────
    # Behavior-cloning warm-start (optional). Pretrains the policy to imitate
    # T4 on (obs, action) pairs from T4-driven auctions, then PPO refines.
    # ──────────────────────────────────────────────────────────────────────
    if int(args.bc_warmstart_episodes) > 0:
        print(f"\nBC warm-start: collecting {args.bc_warmstart_episodes} episodes with T4 in learner seat...")
        bc_states: list[list[float]] = []
        bc_labels: list[int] = []
        bc_t0 = time.time()
        for k in range(int(args.bc_warmstart_episodes)):
            seat = random.randint(0, env.NUM_BIDDERS - 1)
            opps = random.sample(MIMIC_ALIASES, env.NUM_BIDDERS - 1)
            ep = collect_t4_imitation_episode(env, learner_seat=seat, opponent_aliases=opps)
            bc_states.extend(ep["states"])
            bc_labels.extend(ep["labels"])
        print(f"  collected {len(bc_states)} (obs, label) pairs in {time.time() - bc_t0:.1f}s")
        # Update observation normalization from the BC dataset before SL training.
        norm.update(np.asarray(bc_states, dtype=np.float64))
        norm.write_to(policy, value)
        states_t = torch.tensor(bc_states, dtype=torch.float32, device=device)
        labels_t = torch.tensor(bc_labels, dtype=torch.int64, device=device)
        # Class balance — useful diagnostic since labels are typically PASS-heavy at endgame
        label_counts = torch.bincount(labels_t, minlength=N_ACTIONS).tolist()
        print(f"  label distribution: PASS={label_counts[0]}, RAISE={label_counts[1]}, JUMP={label_counts[2]}")
        bc_optim = optim.Adam(policy.parameters(), lr=float(args.bc_lr))
        n_bc = states_t.shape[0]
        for epoch in range(int(args.bc_epochs)):
            idx = torch.randperm(n_bc, device=device)
            ep_loss = 0.0
            ep_batches = 0
            for start in range(0, n_bc, int(args.bc_batch_size)):
                b = idx[start:start + int(args.bc_batch_size)]
                logits = policy(states_t[b])
                loss = F.cross_entropy(logits, labels_t[b])
                bc_optim.zero_grad()
                loss.backward()
                bc_optim.step()
                ep_loss += float(loss.item())
                ep_batches += 1
            with torch.no_grad():
                pred = torch.argmax(policy(states_t), dim=-1)
                acc = (pred == labels_t).float().mean().item()
            if epoch % max(1, args.bc_epochs // 5) == 0 or epoch == args.bc_epochs - 1:
                print(f"  BC epoch {epoch + 1:>2d}/{args.bc_epochs}  ce_loss={ep_loss / max(1, ep_batches):.4f}  train_acc={acc:.3f}")
        # Smoke-test the warm-started policy via a quick deterministic eval.
        policy.eval()
        warm_wins = []
        for k in range(20):
            ls = k % env.NUM_BIDDERS
            opps = [a for i, a in enumerate(MIMIC_ALIASES) if i != ls]
            obs, _ = env.reset(learner_seat=ls, opponent_aliases=opps)
            while not env.done:
                s_tensor = torch.tensor([obs], dtype=torch.float32)
                with torch.no_grad():
                    a = int(torch.argmax(policy(s_tensor), dim=-1).item())
                obs, _r, done, _info = env.step(a)
                if done: break
            warm_wins.append(env.bidders[env.learner_id].paintings_won)
        policy.train()
        warm_avg = sum(warm_wins) / max(1, len(warm_wins))
        print(f"  Warm-start eval (n=20): avg_wins={warm_avg:.2f}  share={warm_avg / args.num_paintings * 100:.1f}%")
        print(f"  (PPO fine-tuning begins now; expect this to climb above the warm-start baseline.)\n")

    win_history: list[int] = []
    t0 = time.time()

    # Early-stopping state.
    best_eval = float("-inf")
    evals_since_improvement = 0
    early_stopped = False

    # Running reward stats for optional reward normalization.
    reward_running_var = 1.0
    reward_running_mean = 0.0
    reward_running_count = 0

    eps_done = 0
    while eps_done < args.episodes:
        # Compute LR + entropy coef for this update according to schedule.
        progress = min(1.0, eps_done / max(1, args.episodes))
        if args.lr_schedule == "linear":
            lr_p_now = lr_policy_init * (1.0 - progress)
            lr_v_now = lr_value_init * (1.0 - progress)
        else:
            lr_p_now = lr_policy_init
            lr_v_now = lr_value_init
        for pg in optim_p.param_groups:
            pg["lr"] = lr_p_now
        for pg in optim_v.param_groups:
            pg["lr"] = lr_v_now
        ent_coef_now = args.entropy_coef + (args.entropy_coef_final - args.entropy_coef) * progress
        # Collect a batch of episodes
        all_states: list[list[float]] = []
        all_actions: list[int] = []
        all_log_probs: list[float] = []
        all_advs: list[float] = []
        all_returns: list[float] = []
        batch_wins: list[int] = []
        # Pre-sample seats/opponents for the whole batch so worker order is
        # irrelevant and reproducibility holds.
        if args.opponent_pool == "mimics":
            opponent_pool = list(MIMIC_ALIASES)
        else:
            opponent_pool = [s.strip() for s in args.opponent_pool.split(",") if s.strip()]
        def _sample_opps() -> list[str]:
            n_needed = env.NUM_BIDDERS - 1
            if len(opponent_pool) >= n_needed:
                return random.sample(opponent_pool, n_needed)
            # Pool smaller than seats — replicate (e.g. one T5-frozen alias → fill all 4 slots).
            return random.choices(opponent_pool, k=n_needed)
        ep_specs = [
            (random.randint(0, env.NUM_BIDDERS - 1), _sample_opps())
            for _ in range(args.episodes_per_update)
        ]
        collect_fn = collect_episode_hybrid if args.action_space == "continuous" else collect_episode
        if pool is None:
            eps = []
            for seat, opps in ep_specs:
                eps.append(collect_fn(env, policy, value,
                                      learner_seat=seat, opponent_aliases=opps, device=device))
        else:
            def _run(spec):
                seat, opps = spec
                return collect_fn(_make_env(), policy, value,
                                  learner_seat=seat, opponent_aliases=opps, device=device)
            futures = [pool.submit(_run, spec) for spec in ep_specs]
            eps = [f.result() for f in futures]
        all_d_actions: list[int] = []        # hybrid: discrete part of action
        all_bid_fracs: list[float] = []      # hybrid: continuous part
        for ep in eps:
            if not ep["states"]:
                batch_wins.append(ep["final_wins"])
                continue
            advs, returns = compute_gae(
                ep["rewards"], ep["values"],
                gamma=args.gamma, lam=args.gae_lambda,
            )
            all_states.extend(ep["states"])
            if args.action_space == "continuous":
                all_d_actions.extend(ep["discrete_actions"])
                all_bid_fracs.extend(ep["bid_fractions"])
            else:
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
        # Optional reward normalization on advantages before PPO update.
        if args.reward_norm and all_advs:
            arr = np.asarray(all_advs, dtype=np.float64)
            batch_mean = float(arr.mean())
            batch_var = float(arr.var())
            n = len(arr)
            total = reward_running_count + n
            if total > 0:
                delta = batch_mean - reward_running_mean
                reward_running_mean += delta * n / total
                reward_running_var = (reward_running_var * reward_running_count + batch_var * n
                                       + (delta ** 2) * reward_running_count * n / total) / total
                reward_running_count = total
            running_std = max(1e-6, math.sqrt(reward_running_var))
            all_advs = [a / running_std for a in all_advs]
        if args.action_space == "continuous":
            update_metrics = ppo_update_hybrid(
                policy, value, optim_p, optim_v,
                {
                    "states": all_states, "discrete_actions": all_d_actions,
                    "bid_fractions": all_bid_fracs, "log_probs": all_log_probs,
                    "advs": all_advs, "returns": all_returns,
                },
                clip_eps=args.clip_eps, n_epochs=args.ppo_epochs,
                batch_size=args.minibatch_size, entropy_coef=ent_coef_now,
                max_grad_norm=args.max_grad_norm, value_clip_eps=args.value_clip_eps,
                target_kl=args.target_kl, device=device,
            )
        else:
            update_metrics = ppo_update(
                policy, value, optim_p, optim_v,
                {
                    "states": all_states, "actions": all_actions,
                    "log_probs": all_log_probs, "advs": all_advs, "returns": all_returns,
                },
                clip_eps=args.clip_eps, n_epochs=args.ppo_epochs,
                batch_size=args.minibatch_size, entropy_coef=ent_coef_now,
                max_grad_norm=args.max_grad_norm, value_clip_eps=args.value_clip_eps,
                target_kl=args.target_kl, device=device,
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
            "lr_policy": round(lr_p_now, 6),
            "lr_value": round(lr_v_now, 6),
            "entropy_coef": round(ent_coef_now, 4),
            "elapsed_s": round(elapsed, 1),
        }
        metrics_file.write(json.dumps(row) + "\n")
        metrics_file.flush()
        _write_status(eps_done=eps_done, avg_wins=avg_wins, best_eval=best_eval,
                      finished=False, early_stopped=False)
        print(
            f"[ep {eps_done:>5d}/{args.episodes}]  avg_wins={avg_wins:.2f}  "
            f"ploss={update_metrics['policy_loss']:+.3f}  vloss={update_metrics['value_loss']:.3f}  "
            f"H={update_metrics['entropy']:.3f}  clip={update_metrics['clip_frac']:.2f}  "
            f"lr={lr_p_now:.5f}  ({elapsed:.0f}s)"
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
            # Held-out deterministic evaluation vs fixed mimic rotation.
            eval_wins: list[int] = []
            policy.eval()
            for k in range(args.eval_episodes):
                ls = k % env.NUM_BIDDERS
                opps = [a for i, a in enumerate(MIMIC_ALIASES) if i != ls]
                obs, _ = env.reset(learner_seat=ls, opponent_aliases=opps)
                while not env.done:
                    s_tensor = torch.tensor([obs], dtype=torch.float32)
                    with torch.no_grad():
                        if args.action_space == "continuous":
                            d_logits, bid_mean, _bid_log_std = policy(s_tensor)
                            d_a = int(torch.argmax(d_logits, dim=-1).item())
                            bf = float(bid_mean.item())
                            obs, _r, done, _info = env.step_continuous(d_a, bf)
                        else:
                            a = int(torch.argmax(policy(s_tensor), dim=-1).item())
                            obs, _r, done, _info = env.step(a)
                    if done:
                        break
                eval_wins.append(env.bidders[env.learner_id].paintings_won)
            policy.train()
            avg_eval = sum(eval_wins) / max(1, len(eval_wins))
            share = avg_eval / args.num_paintings * 100.0
            # Track best-eval checkpoint + early stopping.
            improved = avg_eval > best_eval + args.early_stop_min_delta
            if improved:
                best_eval = avg_eval
                evals_since_improvement = 0
                torch.save(
                    {
                        "model_state": policy.state_dict(),
                        "value_state": value.state_dict(),
                        "obs_dim": OBS_DIM,
                        "hidden": args.hidden,
                        "n_actions": N_ACTIONS,
                        "action_space": args.action_space,
                        "episodes_trained": eps_done,
                        "eval_avg_wins": avg_eval,
                        "norm_mean": norm.mean.tolist(),
                        "norm_std": norm.std().tolist(),
                    },
                    best_output_path,
                )
                tag = " ★ new best (saved)"
            else:
                evals_since_improvement += 1
                tag = f" (no improvement, patience {evals_since_improvement}/{args.early_stop_patience})"
            print(
                f"  [eval n={args.eval_episodes}]  avg_wins={avg_eval:.2f}  "
                f"share={share:.1f}%  (uniform = {100.0 / env.NUM_BIDDERS:.1f}%, best = {best_eval:.2f}){tag}"
            )
            if args.early_stop_patience > 0 and evals_since_improvement >= args.early_stop_patience:
                print(
                    f"  ⚠ Early stopping: eval avg_wins hasn't improved by ≥{args.early_stop_min_delta} "
                    f"in {args.early_stop_patience} consecutive evals."
                )
                early_stopped = True
                break

    metrics_file.close()
    _write_status(eps_done=eps_done, avg_wins=(sum(win_history[-100:]) / max(1, len(win_history[-100:]))),
                  best_eval=best_eval, finished=True, early_stopped=early_stopped)
    print(
        f"\nTraining {'STOPPED EARLY' if early_stopped else 'complete'} at episode {eps_done}/{args.episodes}."
    )
    print(f"Last checkpoint: {output_path}")
    print(f"Best-eval checkpoint ({best_eval:.2f} wins): {best_output_path}")
    print(f"Metrics log: {metrics_path}")
    print("Use Math-T5 in a loadout to deploy.")


if __name__ == "__main__":
    main()
