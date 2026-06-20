"""
Neural-net mimic agents for the open painting auction.

Each mimic loads a pair of small PyTorch networks:
  - classifier: predicts RAISE/PASS
  - regressor:  predicts overbid above minimum legal (active only on RAISE)

Models are loaded lazily and cached for the lifetime of the process.
"""
from __future__ import annotations

import os
import random as _stdlib_random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .state import OpenAuctionAction


def _deception_temperature() -> float:
    """Scale applied to each LLM's learned residual std for stochastic dispatch.

    Default 1.75 — "sample with the model's natural variance plus a small
    smoothing buffer". The actual additive Gaussian noise std per attribute
    is `T * residual_std_per_attr`, where residual_std is fit at training
    time and stored in the .pt checkpoint.

    T = 1.75 was selected by sweeping {0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
    2.0, 2.5, 3.0, 4.0} on the v3 LLM episodes against 150 matched-seed
    all-mimic episodes (3 replicates per truth seed). T = 1.75 minimized
    Cramer's V at 0.017 with chi-sq p = 0.95 (compared to auction's 0.041
    benchmark) while keeping per-LLM strategic fingerprints intact;
    higher T values start washing out the per-LLM signature.

    Useful overrides via DECEPTION_MIMIC_TEMPERATURE env var:
      T = 0      → fully deterministic argmax-style emission
      T = 1.0    → exactly the source LLM's learned variance
      T = 1.75   → default (smoothed natural variance, best fidelity)
      T = 3.0+   → over-stochastic; per-LLM fingerprint degrades
    """
    import os
    try:
        return max(0.0, float(os.environ.get("DECEPTION_MIMIC_TEMPERATURE", "0.50")))
    except (TypeError, ValueError):
        return 1.75


def _temperature() -> float:
    """Global stochasticity knob, read from MIMIC_TEMPERATURE env var.

    0    → deterministic argmax (committed default).
    >0   → sample from sharpened Bernoulli on the classifier probability,
           and add Gaussian noise to the regressor's step count.

    Default 0.3 reproduces the mimic-auction behaviour that the original
    chi-squared test (p=0.146) was measured against — slight sampling that
    breaks ties between two confident mimics without making them erratic.
    """
    try:
        return max(0.0, float(os.environ.get("MIMIC_TEMPERATURE", "1.0")))
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Network definitions — must match train_auction_nn.py
# ---------------------------------------------------------------------------

class _Classifier(nn.Module):
    def __init__(self, input_dim: int = 32, hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.input_mean) / self.input_std
        return self.net(x).squeeze(-1)


class _Regressor(nn.Module):
    def __init__(self, input_dim: int = 32, hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.input_mean) / self.input_std
        return self.softplus(self.trunk(x)).squeeze(-1)

# ---------------------------------------------------------------------------
# Model paths / cache
# ---------------------------------------------------------------------------

_MODELS_DIR = Path(__file__).parent / "models" / "v7"

_MIMIC_NAMES = {"Grok", "Opus", "GPT-5.4", "Pro", "Llama"}

_clf_cache: dict[str, _Classifier] = {}
_reg_cache: dict[str, _Regressor | None] = {}


def _safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-.]", "_", name)


def _load_clf(name: str) -> _Classifier:
    if name not in _clf_cache:
        path = _MODELS_DIR / f"auction_clf_v6_{_safe_filename(name)}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Mimic classifier not found: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = _Classifier(
            input_dim=int(ckpt.get("input_dim", 32)),
            hidden=int(ckpt.get("hidden", 64)),
            dropout=float(ckpt.get("dropout", 0.2)),
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        _clf_cache[name] = model
    return _clf_cache[name]


def _load_reg(name: str) -> _Regressor | None:
    if name not in _reg_cache:
        path = _MODELS_DIR / f"auction_reg_v6_{_safe_filename(name)}.pt"
        if not path.exists():
            _reg_cache[name] = None
        else:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model = _Regressor(
                input_dim=int(ckpt.get("input_dim", 32)),
                hidden=int(ckpt.get("hidden", 64)),
                dropout=float(ckpt.get("dropout", 0.2)),
            )
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            _reg_cache[name] = model
    return _reg_cache[name]


# ---------------------------------------------------------------------------
# Feature extraction — mirrors build_nn_rows exactly
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0.0 else 0.0


def _stats(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    n = float(len(values))
    mean = sum(values) / n
    vmin = min(values)
    vmax = max(values)
    std = (sum((x - mean) ** 2 for x in values) / n) ** 0.5
    return mean, std, vmin, vmax


def _topk(values: list[float], k: int) -> list[float]:
    out = list(values[:k])
    out.extend([0.0] * max(0, k - len(out)))
    return out


def build_feature_vector(
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
) -> list[float]:
    self_start = max(1.0, float(start_budget))
    total_p = max(1, total_paintings)
    pr = max(1.0, float(paintings_remaining))

    opp_budgets = sorted(
        [float(all_budgets[k]) for k in all_budgets if k != bidder_id],
        reverse=True,
    )
    opp_counts = sorted(
        [float(all_counts[k]) for k in all_counts if k != bidder_id],
        reverse=True,
    )
    opp_b_mean, opp_b_std, opp_b_min, opp_b_max = _stats(opp_budgets)
    opp_c_mean, opp_c_std, _opp_c_min, opp_c_max = _stats(opp_counts)
    top_opp_budgets = _topk(opp_budgets, 4)
    top_opp_counts = _topk(opp_counts, 4)

    all_counts_vals = list(all_counts.values())
    global_count_max = max(all_counts_vals) if all_counts_vals else 0.0

    last_hist = bid_history[-1] if bid_history else {}
    own_current_bid = float(
        (public_bid_table.get(bidder_id) or {}).get("current_bid_this_painting") or 0.0
    )

    cb = float(current_bid)
    ml = float(min_next_bid)
    yb = float(your_budget)
    yc = float(your_count)

    return [
        painting_number / total_p,
        pr,
        1.0 if is_last_painting else 0.0,
        _safe_div(cb, self_start),
        _safe_div(ml, self_start),
        float(len(active_bidders)),
        float(len(bid_history)),
        _safe_div(yb, self_start),
        _safe_div(yb, opp_b_mean),
        _safe_div(yb, ml),
        yc,
        yc - global_count_max,
        _safe_div(_safe_div(yb, pr), self_start),
        _safe_div(own_current_bid, self_start),
        1.0 if str(last_hist.get("action_type") or "") == "raise" else 0.0,
        _safe_div(float(last_hist.get("bid_amount") or 0.0), self_start),
        1.0 if str(last_hist.get("bidder_id") or "") == bidder_id else 0.0,
        _safe_div(opp_b_mean, self_start),
        _safe_div(opp_b_std, self_start),
        _safe_div(opp_b_min, self_start),
        _safe_div(opp_b_max, self_start),
        opp_c_mean,
        opp_c_std,
        opp_c_max,
        *[_safe_div(b, self_start) for b in top_opp_budgets],
        *top_opp_counts,
    ]


# ---------------------------------------------------------------------------
# Bid increment rules (must match _min_raise in build_auction_dataset.py
# and the auction env)
# ---------------------------------------------------------------------------

def _min_raise_step(current_bid: int) -> int:
    if current_bid < 1000:
        return 50
    if current_bid < 3000:
        return 100
    return 250


def _snap_to_legal_bid(proposed: int, current_bid: int, min_next_bid: int) -> int:
    """Round `proposed` to the nearest legal multiple of the bid step,
    then bump up to `min_next_bid` if it falls below."""
    step = _min_raise_step(current_bid)
    snapped = int(round(proposed / step) * step)
    if snapped < min_next_bid:
        # Round up to the next multiple of `step` that is at least min_next_bid
        snapped = ((min_next_bid + step - 1) // step) * step
    return snapped


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

MIMIC_PREFIX = "Mimic-"


def is_mimic(alias: str) -> bool:
    return str(alias or "").startswith(MIMIC_PREFIX)


def mimic_name(alias: str) -> str:
    """Strip the 'Mimic-' prefix to get the underlying LLM name."""
    return alias[len(MIMIC_PREFIX):]


def mimic_bid(
    *,
    alias: str,
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
    name = mimic_name(alias)
    clf = _load_clf(name)
    reg = _load_reg(name)

    x = build_feature_vector(
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
    X = torch.from_numpy(np.array([x], dtype=np.float32))
    T = _temperature()

    with torch.no_grad():
        logit = clf(X).item()
    raise_prob = 1.0 / (1.0 + float(np.exp(-logit)))

    if T > 0:
        # Temperature-sharpened Bernoulli sample. Lower T → closer to argmax;
        # T=1 samples raw probability. Prevents deterministic ties between two
        # equally confident mimics that would otherwise fight to budget exhaustion.
        sharp = raise_prob ** (1.0 / T)
        sharp_neg = (1.0 - raise_prob) ** (1.0 / T)
        sampled_p = sharp / max(1e-9, sharp + sharp_neg)
        action_pred = 1 if _stdlib_random.random() < sampled_p else 0
    else:
        action_pred = 1 if raise_prob >= 0.5 else 0

    if action_pred == 0:
        return OpenAuctionAction(action_type="pass", bid_amount=None, message_text="PASS")

    # Raise: regressor predicts # of legal increments above min_next_bid.
    step = _min_raise_step(current_bid)
    extra_steps = 0
    if reg is not None:
        with torch.no_grad():
            raw = float(reg(X).item())
        if T > 0:
            raw += _stdlib_random.gauss(0.0, 0.7 * T)
        extra_steps = max(0, int(round(raw)))

    bid_amount = min_next_bid + extra_steps * step
    bid_amount = min(your_budget, bid_amount)

    if bid_amount < min_next_bid or bid_amount > your_budget:
        return OpenAuctionAction(action_type="pass", bid_amount=None, message_text="PASS")

    return OpenAuctionAction(
        action_type="raise",
        bid_amount=bid_amount,
        message_text=f"BID {bid_amount}",
    )


# ---------------------------------------------------------------------------
# Deception Competition mimic dispatch (D9: two-head architecture)
# ---------------------------------------------------------------------------

DECEPTION_MIMIC_INPUT_DIM = 32


def build_deception_mimic_input(
    *,
    truth: list[float],
    own_trust: list[float],
    opponents_trust: list[list[float]] | None,
    round_index: int,
    total_rounds: int,
    own_wins_count: int = 0,
    opponents_wins_count: list[int] | None = None,
) -> list[float]:
    """Build the 32-dim mimic input vector.

    Layout:
        [ 0: 5]  truth                              — this round's truth
        [ 5:10]  own_trust                          — per-attribute own trust
        [10:15]  opp_trust_max_per_attr             — strongest opp on each attr
        [15:20]  opp_trust_mean_per_attr            — average opp on each attr
        [20:25]  opp_trust_min_per_attr             — weakest opp on each attr
        [25:29]  opp_strengths_sorted_desc          — each opp's mean trust, sorted
        [29:30]  round_progress                     — round_index / total_rounds
        [30:31]  own_wins_norm                      — own_wins_count / total_rounds
        [31:32]  opp_wins_max_norm                  — max(opp_wins) / total_rounds

    Under track-record, trust = cum_disc / max(1, wins_count). The marginal cost
    of an over-claim THIS round depends on wins_count: a 0.5 disc lie at
    wins=1 → trust drops by 0.5; at wins=6 → trust drops by 0.07. Same trust
    state, very different future-cost calculus. own_wins_norm and the
    strongest-opp's wins (max) are the two scalar additions; opp_wins for the
    weaker opponents are correlated enough with opp_trust aggregates to skip.

    Permutation-invariant in opponent order.
    """
    n_attrs = 5
    n_opps = 4
    t = list(truth[:n_attrs])
    if len(t) < n_attrs:
        t = t + [0.0] * (n_attrs - len(t))
    own = list(own_trust[:n_attrs])
    if len(own) < n_attrs:
        own = own + [1.0] * (n_attrs - len(own))
    opps = []
    for vec in (opponents_trust or []):
        v = list(vec[:n_attrs])
        if len(v) < n_attrs:
            v = v + [1.0] * (n_attrs - len(v))
        opps.append(v)
    while len(opps) < n_opps:
        opps.append([1.0] * n_attrs)
    opps = opps[:n_opps]
    opp_max = [max(opps[i][a] for i in range(n_opps)) for a in range(n_attrs)]
    opp_mean = [sum(opps[i][a] for i in range(n_opps)) / float(n_opps) for a in range(n_attrs)]
    opp_min = [min(opps[i][a] for i in range(n_opps)) for a in range(n_attrs)]
    opp_strengths = sorted(
        [sum(opps[i]) / float(n_attrs) for i in range(n_opps)],
        reverse=True,
    )
    progress = float(round_index) / max(1, int(total_rounds))
    progress = max(0.0, min(1.0, progress))
    own_wins_norm = max(0.0, min(1.0, float(own_wins_count) / max(1, int(total_rounds))))
    opp_wins = list(opponents_wins_count or [])
    while len(opp_wins) < n_opps:
        opp_wins.append(0)
    opp_wins = opp_wins[:n_opps]
    opp_wins_max_norm = max(
        max(0.0, min(1.0, float(w) / max(1, int(total_rounds)))) for w in opp_wins
    )
    return (
        t + own + opp_max + opp_mean + opp_min + opp_strengths
        + [progress, own_wins_norm, opp_wins_max_norm]
    )


class _DeceptionMimic(nn.Module):
    """Single-head direct-claim deception mimic — mirrors train_deception_nn.py.

    Sigmoid output gives claim_a in [0, 1] directly.
    """

    def __init__(self, input_dim: int = DECEPTION_MIMIC_INPUT_DIM, hidden: int = 32, dropout: float = 0.2, num_attrs: int = 5):
        super().__init__()
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.claim_head = nn.Linear(hidden, num_attrs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.input_mean) / self.input_std
        h = self.trunk(x)
        return torch.sigmoid(self.claim_head(h))


_DECEPTION_MODELS_DIR = Path(__file__).parent / "models" / "deception_v5"
_DECEPTION_NET_CACHE: dict[str, _DeceptionMimic] = {}


def _strip_mimic_prefix(alias: str) -> str:
    return re.sub(r"^Mimic-", "", alias)


def _deception_model_path(alias: str) -> Path:
    """Map alias to the .pt file. 'Mimic-Opus' → mimic_Opus.pt."""
    base = _strip_mimic_prefix(alias)
    return _DECEPTION_MODELS_DIR / f"mimic_{base}.pt"


_DECEPTION_RESIDUAL_STD_CACHE: dict[str, list[float]] = {}


def _load_deception_mimic(alias: str) -> _DeceptionMimic | None:
    if alias in _DECEPTION_NET_CACHE:
        return _DECEPTION_NET_CACHE[alias]
    path = _deception_model_path(alias)
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = _DeceptionMimic(
        input_dim=int(payload.get("input_dim", DECEPTION_MIMIC_INPUT_DIM)),
        hidden=int(payload.get("hidden", 32)),
        num_attrs=int(payload.get("num_attrs", 5)),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    _DECEPTION_NET_CACHE[alias] = model
    # Cache residual std (may be absent on legacy checkpoints; default to 0
    # which makes stochastic dispatch a no-op for old models).
    rs = payload.get("residual_std")
    if isinstance(rs, list) and len(rs) == 5:
        _DECEPTION_RESIDUAL_STD_CACHE[alias] = [float(x) for x in rs]
    else:
        _DECEPTION_RESIDUAL_STD_CACHE[alias] = [0.0] * 5
    return model


def deception_mimic_claim(
    alias: str,
    truth: list[float],
    own_trust: list[float],
    opponents_trust: list[list[float]] | None = None,
    *,
    round_index: int = 0,
    total_rounds: int = 12,
    own_wins_count: int = 0,
    opponents_wins_count: list[int] | None = None,
) -> list[float]:
    """Return a 5-float claim vector for the named mimic.

    Input is the 35-dim vector built by `build_deception_mimic_input` (see
    that function for the layout): truth + own_trust + opp_trust aggregates
    + round_progress + own_wins_norm + opp_wins_norm. Permutation-invariant
    in opponent order.
    """
    model = _load_deception_mimic(alias)
    if model is None:
        return [round(float(t), 2) for t in truth]

    x = build_deception_mimic_input(
        truth=list(truth),
        own_trust=list(own_trust),
        opponents_trust=opponents_trust,
        round_index=round_index,
        total_rounds=total_rounds,
        own_wins_count=own_wins_count,
        opponents_wins_count=opponents_wins_count,
    )
    x_tensor = torch.tensor([x], dtype=torch.float32)
    with torch.no_grad():
        claim_pred = model(x_tensor)
    raw = claim_pred.squeeze(0).cpu().numpy()

    # Stochastic dispatch: noise std per attribute = T * residual_std_per_attr
    # (residual_std is the LLM's natural variance learned at training time).
    # T=1.0 (default) reproduces the LLM's variance; T=0 is deterministic.
    # Per-mimic-call RNG seeded from the full input state hash so noise is
    # reproducible from the env_seed (via truth schedule + own_trust history)
    # rather than dependent on Python's global random state (which drifts
    # unpredictably across HTTP server activity).
    T = _deception_temperature()
    rs = _DECEPTION_RESIDUAL_STD_CACHE.get(alias, [0.0] * 5)
    import hashlib as _hashlib
    h = _hashlib.sha256()
    h.update(alias.encode("utf-8"))
    h.update(str(int(round_index)).encode("utf-8"))
    h.update(str(int(total_rounds)).encode("utf-8"))
    h.update(str(tuple(round(float(t), 4) for t in truth)).encode("utf-8"))
    h.update(str(tuple(round(float(t), 4) for t in own_trust)).encode("utf-8"))
    h.update(str(int(own_wins_count or 0)).encode("utf-8"))
    seed = int.from_bytes(h.digest()[:8], "big")
    local_rng = _stdlib_random.Random(seed)
    out: list[float] = []
    for a in range(5):
        v = float(raw[a])
        noise_std = T * float(rs[a])
        if noise_std > 0:
            v = v + local_rng.gauss(0.0, noise_std)
        v = max(0.0, min(1.0, v))
        out.append(round(v, 2))
    return out
