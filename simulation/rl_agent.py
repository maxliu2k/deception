"""Policy + value networks for the Math-T5 RL bidder, and an inference adapter
(``rl_bid``) mirroring ``mimic_agent.mimic_bid`` so the server can call it
from auction turns once a checkpoint exists at ``models/rl/t5_ppo.pt``.

The policy is a small MLP over the same 32-dim feature vector mimics use, so
T4 (heuristic) and T5 (learned) have identical information sets. Action space
is binary: 0 = PASS, 1 = RAISE-min. (Variable bid sizes are intentionally out
of scope so T5 only adds "learned policy" on top of T4's information.)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mimic_agent import build_feature_vector
from .state import OpenAuctionAction


MODELS_DIR = Path(__file__).parent / "models" / "rl"
DEFAULT_CKPT = MODELS_DIR / "t5_ppo.pt"

OBS_DIM = 32
N_ACTIONS = 2  # 0 = PASS, 1 = RAISE-min


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class PolicyNet(nn.Module):
    """Discrete-action policy network. Outputs 2 logits (PASS, RAISE)."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 64, n_actions: int = N_ACTIONS):
        super().__init__()
        self.register_buffer("input_mean", torch.zeros(obs_dim))
        self.register_buffer("input_std", torch.ones(obs_dim))
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.input_mean) / self.input_std
        return self.net(x)


class ValueNet(nn.Module):
    """State-value baseline for PPO."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 64):
        super().__init__()
        self.register_buffer("input_mean", torch.zeros(obs_dim))
        self.register_buffer("input_std", torch.ones(obs_dim))
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.input_mean) / self.input_std
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Inference adapter (used by the server when bidder_alias == "Math-T5")
# ---------------------------------------------------------------------------

_policy_cache: dict[str, PolicyNet] = {}


def _load_policy(ckpt_path: Path = DEFAULT_CKPT) -> PolicyNet:
    key = str(ckpt_path)
    cached = _policy_cache.get(key)
    if cached is not None:
        return cached
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Math-T5 policy not trained yet — expected checkpoint at {ckpt_path}. "
            f"Run: python -m simulation.train_rl_t5 --output {ckpt_path}"
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = PolicyNet(
        obs_dim=int(ckpt.get("obs_dim", OBS_DIM)),
        hidden=int(ckpt.get("hidden", 64)),
        n_actions=int(ckpt.get("n_actions", N_ACTIONS)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    _policy_cache[key] = model
    return model


def _temperature() -> float:
    """Stochasticity knob for inference. 0 -> argmax; >0 -> Bernoulli sample."""
    try:
        return max(0.0, float(os.environ.get("RL_TEMPERATURE", "0")))
    except Exception:
        return 0.0


def rl_bid(
    *,
    bidder_id: str,
    your_budget: int,
    your_count: int,
    current_bid: int,
    current_leader: str | None,
    active_bidders: list[str],
    bid_history: list[dict],
    all_budgets: dict[str, int],
    all_counts: dict[str, int],
    public_bid_table: dict[str, dict],
    painting_number: int,
    total_paintings: int,
    paintings_remaining: int,
    is_last_painting: bool,
    min_next_bid: int,
    start_budget: int = 10000,
) -> OpenAuctionAction:
    """Math-T5 bid function. Mirrors ``mimic_bid``'s signature."""
    feat = build_feature_vector(
        bidder_id=bidder_id,
        your_budget=your_budget,
        your_count=your_count,
        current_bid=current_bid,
        current_leader=current_leader,
        active_bidders=active_bidders,
        bid_history=bid_history,
        all_budgets=all_budgets,
        all_counts=all_counts,
        public_bid_table=public_bid_table,
        painting_number=painting_number,
        total_paintings=total_paintings,
        paintings_remaining=paintings_remaining,
        is_last_painting=is_last_painting,
        min_next_bid=min_next_bid,
        start_budget=start_budget,
    )
    policy = _load_policy()
    x = torch.from_numpy(np.array([feat], dtype=np.float32))
    with torch.no_grad():
        logits = policy(x)[0]
    T = _temperature()
    if T <= 0.0:
        action_idx = int(torch.argmax(logits).item())
    else:
        probs = F.softmax(logits / T, dim=-1)
        action_idx = int(torch.multinomial(probs, 1).item())

    if action_idx == 0 or int(min_next_bid) > int(your_budget):
        return OpenAuctionAction(action_type="pass", bid_amount=None, message_text="PASS")
    return OpenAuctionAction(
        action_type="raise",
        bid_amount=int(min_next_bid),
        message_text=f"BID {int(min_next_bid)}",
    )
