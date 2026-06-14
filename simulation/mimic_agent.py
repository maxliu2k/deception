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

class _DeceptionMimic(nn.Module):
    """Two-head deception mimic — must mirror simulation/train_deception_nn.py.

    Default input_dim is the 23-dim leakage-safe v3 schema; the loader passes the
    checkpoint's actual input_dim, which also supports legacy 10-dim models.
    """

    def __init__(self, input_dim: int = 23, hidden: int = 32, dropout: float = 0.2, num_attrs: int = 5):
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
        self.lie_head = nn.Linear(hidden, num_attrs)
        self.claim_head = nn.Linear(hidden, num_attrs)
        # Per-attribute regressor residual std, populated from the checkpoint by
        # _load_deception_mimic. Plain attribute (not a buffer / not in state_dict)
        # so legacy checkpoints without it load cleanly; zeros → deterministic.
        self.claim_resid_std = torch.zeros(num_attrs)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = (x - self.input_mean) / self.input_std
        h = self.trunk(x)
        return self.lie_head(h), torch.sigmoid(self.claim_head(h))


_DECEPTION_MODELS_DIR = Path(__file__).parent / "models" / "deception_v2"
_DECEPTION_NET_CACHE: dict[str, _DeceptionMimic] = {}


def _strip_mimic_prefix(alias: str) -> str:
    return re.sub(r"^Mimic-", "", alias)


def _deception_model_path(alias: str) -> Path:
    """Map alias to the .pt file. 'Mimic-Opus' → mimic_Opus.pt."""
    base = _strip_mimic_prefix(alias)
    return _DECEPTION_MODELS_DIR / f"mimic_{base}.pt"


def _load_deception_mimic(alias: str) -> _DeceptionMimic | None:
    if alias in _DECEPTION_NET_CACHE:
        return _DECEPTION_NET_CACHE[alias]
    path = _deception_model_path(alias)
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = _DeceptionMimic(
        input_dim=int(payload.get("input_dim", 10)),
        hidden=int(payload.get("hidden", 32)),
        num_attrs=int(payload.get("num_attrs", 5)),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    rs = payload.get("claim_resid_std")
    if rs is not None:
        try:
            model.claim_resid_std = torch.tensor([float(v) for v in rs], dtype=torch.float32)
        except Exception:
            pass
    _DECEPTION_NET_CACHE[alias] = model
    return model


def _build_deception_features_v3(
    *,
    observed_truth: list[float | None],
    visible_mask: list[int],
    preferences: list[float],
    own_trust: float,
    opponents_trust: list[float],
    round_fraction: float,
    threshold: float,
    penalty: float,
) -> list[float]:
    """23-dim leakage-safe feature vector. MUST match the feature order produced
    by extract_deception_dataset.FEATURE_NAMES."""
    observed_for_x = [0.5 if v is None else float(v) for v in observed_truth[:5]]
    while len(observed_for_x) < 5:
        observed_for_x.append(0.5)
    mask = [float(m) for m in visible_mask[:5]]
    while len(mask) < 5:
        mask.append(1.0)
    prefs = [float(p) for p in (preferences or [])[:5]]
    while len(prefs) < 5:
        prefs.append(0.0)
    opp = [float(o) for o in (opponents_trust or [])[:4]]
    while len(opp) < 4:
        opp.append(0.0)
    return (
        observed_for_x
        + mask
        + prefs
        + [float(own_trust)]
        + opp
        + [float(round_fraction), float(threshold), float(penalty)]
    )


def deception_mimic_claim(
    alias: str,
    truth: list[float],
    own_trust: float,
    opponents_trust: list[float],
    *,
    information_mode: str = "full",
    observed_truth: list[float | None] | None = None,
    visible_attrs: list[int] | None = None,
    preferences: list[float] | None = None,
    threshold: float = 0.0,
    penalty: float = 0.0,
    round_index: int = 0,
    total_rounds: int = 12,
) -> list[float]:
    """Return a 5-float claim vector for the named mimic.

    Two-head NN: P(lie per attr) + raw claim per attr. The emitted claim depends
    on whether the attribute's truth is observable:
      - visible attr: sample/argmax the lie decision. If no lie, snap to the known
        truth (avoids float-noise spurious catches); if lie, use the regressor.
      - hidden attr (partial info): the truth is unknown, so always use the
        regressor — there is no truth to snap to.

    The feature vector is chosen to match the loaded checkpoint's input width:
    23-dim leakage-safe (v3) or the legacy 10-dim (truth + trust) for older
    scripted-tier mimics.
    """
    model = _load_deception_mimic(alias)
    # `truth` here is the policy/observed truth (0.5 placeholder for hidden);
    # prefer the explicit observed_truth list (with None for hidden) when given.
    obs = list(observed_truth) if observed_truth is not None else [float(t) for t in truth[:5]]
    while len(obs) < 5:
        obs.append(0.5)
    if visible_attrs is not None:
        visible_set = {int(a) for a in visible_attrs}
    elif information_mode == "partial":
        visible_set = {a for a, v in enumerate(obs) if v is not None}
    else:
        visible_set = set(range(5))
    visible_mask = [1 if a in visible_set else 0 for a in range(5)]
    # Per-attribute snap value for visible attrs (the known truth).
    snap_truth = [0.5 if obs[a] is None else float(obs[a]) for a in range(5)]

    if model is None:
        # Mimic not yet trained — honest play on visible attrs, placeholder elsewhere.
        return [round(snap_truth[a], 2) for a in range(5)]

    input_dim = int(model.input_mean.numel())
    if input_dim >= 23:
        round_fraction = float(round_index) / max(1.0, float(total_rounds) - 1.0)
        x = _build_deception_features_v3(
            observed_truth=obs,
            visible_mask=visible_mask,
            preferences=preferences or [0.0] * 5,
            own_trust=own_trust,
            opponents_trust=opponents_trust,
            round_fraction=round_fraction,
            threshold=threshold,
            penalty=penalty,
        )
    else:
        # Legacy 10-dim: truth(5) + own_trust(1) + opponents_trust(4).
        opp = [float(o) for o in (opponents_trust or [])[:4]]
        while len(opp) < 4:
            opp.append(1.0)
        x = [snap_truth[a] for a in range(5)] + [float(own_trust)] + opp

    x_tensor = torch.tensor([x], dtype=torch.float32)
    with torch.no_grad():
        lie_logits, raw_claim = model(x_tensor)
    p_lie = torch.sigmoid(lie_logits).squeeze(0).cpu().numpy()
    raw = raw_claim.squeeze(0).cpu().numpy()

    T = _temperature()
    resid_std = getattr(model, "claim_resid_std", None)

    def _regressor_val(a: int) -> float:
        # Point estimate + calibrated Gaussian noise (std = training residual
        # spread, scaled by temperature). Reproduces the real LLM's claim spread,
        # which the bare point estimate under-disperses. T=0 → deterministic.
        base = float(raw[a])
        if T > 0 and resid_std is not None and a < int(resid_std.numel()):
            s = float(resid_std[a])
            if s > 0.0:
                base += _stdlib_random.gauss(0.0, s * T)
        return round(max(0.0, min(1.0, base)), 2)

    out: list[float] = []
    for a in range(5):
        if a not in visible_set:
            # Hidden attribute: cannot snap to an unknown truth → always regressor.
            out.append(_regressor_val(a))
            continue
        if T > 0:
            lie_a = _stdlib_random.random() < float(p_lie[a])
        else:
            lie_a = float(p_lie[a]) > 0.5
        out.append(_regressor_val(a) if lie_a else round(snap_truth[a], 2))
    return out
