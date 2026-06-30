"""Negotiation mimic NN inference (fully symmetric, role-agnostic).

The mimic processes both roles identically:
  * 32-dim input vector built from role-agnostic features (no role flag)
  * reservation-threshold head (accept iff standing_ext >= reservation_ext)
  * counter-extraction head applied log-symmetrically downstream

The action space is binary — continue (counter) or accept. A negotiation ends
only by acceptance or by hitting the message limit (no-deal); there is no
explicit "reject" action, so none is modeled.

Log-symmetric extraction mapping (multiplicative-symmetric across roles):
    buyer  price = round(own_value * exp(-e))     clipped to [1, own_value]
    seller price = round(own_value * exp(+e))     clipped to [own_value, inf)

All numeric features are expressed in log-extraction units relative to my
own_value, so the same value distribution applies regardless of role.
"""
from __future__ import annotations

import hashlib
import math
import os as _os
import random as _random
from pathlib import Path

import torch
import torch.nn as nn


INPUT_DIM = 32
HIDDEN = 64
MODELS_DIR = Path(_os.environ.get(
    "NEG_MIMIC_DIR",
    str(Path(__file__).parent / "models" / "negotiation_v6_resv"),
))
# --- Stochasticity (matches the auction/deception mimic recipe) --------------
# MIMIC_TEMPERATURE T:  0 → deterministic; 1 → sample at the trained
#   probabilities (temperature-Bernoulli accept + Gaussian counter noise).
# PRICE_NOISE: std-dev (log-extraction units) of the Gaussian added to the
#   counter offer, scaled by T — analogous to the auction bid regressor noise.
# ACCEPT_TEMP: sigmoid sharpness of the reservation-threshold accept classifier
#   (must match the value used in training).
MIMIC_TEMPERATURE = float(_os.environ.get("NEG_MIMIC_TEMPERATURE", "1.0"))
PRICE_NOISE = float(_os.environ.get("NEG_PRICE_NOISE", "0.04"))
ACCEPT_TEMP = float(_os.environ.get("NEG_ACCEPT_TEMP", "10.0"))
# Reservation margin (log-extraction units): shifts the accept operating point.
# 0.5 calibrated on TRAINING close-rate (matches the LLM's), validated on the
# held-out eval (agreement 0/5 and extraction 0/5 rejected under the paired
# McNemar/Wilcoxon tests).
RESERVATION_MARGIN = float(_os.environ.get("NEG_RESV_MARGIN", "0.5"))
_NET_CACHE: dict[str, "_NegotiationMimic"] = {}


class _NegotiationMimic(nn.Module):
    """Reservation-threshold mimic (classic feedforward).

    Two coupled scalar outputs per turn, both in log-extraction units:
      * counter_ext     — the extraction I demand if I counter (the offer I make)
      * reservation_ext — the worst extraction I'd still ACCEPT this turn

    Inference: accept the standing offer iff standing_ext >= reservation_ext
    (a deterministic threshold crossing), else counter at counter_ext. This
    replaces the old 3-class softmax action head, which could not express
    "accept an offer below my counter-target" — the concession behavior that
    drove the mimic's under-closing in free-run duels.
    """

    def __init__(self, input_dim: int = INPUT_DIM, hidden: int = HIDDEN):
        super().__init__()
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.ext_head = nn.Linear(hidden, 1)         # counter extraction
        self.reservation_head = nn.Linear(hidden, 1)  # accept threshold

    def forward(self, x: torch.Tensor):
        x = (x - self.input_mean) / (self.input_std + 1e-6)
        h = self.trunk(x)
        return self.ext_head(h).squeeze(-1), self.reservation_head(h).squeeze(-1)


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


# Ensemble size: if > 1, load K members from MODELS_DIR/seed{k}/mimic_{alias}.pt
# and average their outputs (variance reduction → better held-out fidelity).
ENSEMBLE_K = int(_os.environ.get("NEG_MIMIC_K", "1"))


class _EnsembleMimic:
    """Averages the (counter_ext, reservation_ext) outputs of K member nets.
    Drop-in for a single _NegotiationMimic at the model(xt) call site."""

    def __init__(self, members: list["_NegotiationMimic"]):
        self.members = members

    def __call__(self, x: torch.Tensor):
        cs, rs = [], []
        for m in self.members:
            c, r = m(x)
            cs.append(c); rs.append(r)
        return torch.stack(cs).mean(0), torch.stack(rs).mean(0)


def _load_one(path: Path) -> "_NegotiationMimic | None":
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sd = payload.get("state_dict") or payload.get("policy")
    if sd is None:
        return None
    model = _NegotiationMimic(
        input_dim=int(payload.get("input_dim", INPUT_DIM)),
        hidden=int(payload.get("hidden", HIDDEN)),
    )
    model.load_state_dict(sd)
    model.eval()
    return model


def load_mimic(alias: str):
    bare = alias[len("Mimic-"):] if alias.startswith("Mimic-") else alias
    if bare in _NET_CACHE:
        return _NET_CACHE[bare]
    fname = f"mimic_{_safe_filename(bare)}.pt"
    if ENSEMBLE_K > 1:
        members = [m for k in range(ENSEMBLE_K)
                   if (m := _load_one(MODELS_DIR / f"seed{k}" / fname)) is not None]
        if not members:
            return None
        net = _EnsembleMimic(members)
    else:
        net = _load_one(MODELS_DIR / fname)
        if net is None:
            return None
    _NET_CACHE[bare] = net
    return net


def _price_to_extraction(price: float, own_value: float, role: str) -> float:
    """Convert an absolute price to a role-agnostic log-symmetric extraction.

    Buyer:  e = log(own_value / price)    (positive when price < own_value)
    Seller: e = log(price / own_value)    (positive when price > own_value)

    Both formulations give e ∈ [0, ∞) when the price gives me surplus, and
    are exactly mirror images in log-price space: swapping (role, v, price)
    in any structurally symmetric way produces the same e magnitude. This is
    the "truly multiplicative-symmetric" extraction (1±e is its first-order
    Taylor approximation around e=0).
    """
    if own_value <= 0 or price <= 0:
        return 0.0
    import math as _math
    if role == "buyer":
        return _math.log(own_value / price)
    return _math.log(price / own_value)


def build_feature_vector(
    *,
    role: str,
    own_private_value: float,
    own_target_price: float | None,
    turn_history: list[dict],
    standing_price: float | None,
    turn_index: int,
    message_limit: int,
) -> list[float]:
    """Build the 32-dim role-agnostic feature vector.

    All price-derived features are in extraction units (positive = favors me).
    No role flag, no opponent-private-value leakage — only my own_value, my
    own target hint, and the visible price history feed in. Same vector
    produced for either role given the same abstract state.
    """
    n_turns = len(turn_history)
    msg_limit = max(1, int(message_limit))
    own_value = float(own_private_value) if own_private_value else 1.0
    own_target = float(own_target_price) if own_target_price is not None else own_value

    def _ext(price: float) -> float:
        return _price_to_extraction(price, own_value, role)

    # --- Time / turn structure (4) ---
    turn_number_frac = min(1.0, float(turn_index) / msg_limit)
    history_count_frac = min(1.0, float(n_turns) / msg_limit)
    turns_remaining = max(0, msg_limit - n_turns)
    is_first_turn = 1.0 if n_turns == 0 else 0.0
    # My turns remaining ≈ ceil(turns_remaining / 2). Last turn for me iff <= 1.
    my_turns_remaining = (turns_remaining + 1) // 2
    is_my_last_turn = 1.0 if my_turns_remaining <= 1 else 0.0
    is_overall_last_turn = 1.0 if turns_remaining <= 1 else 0.0

    # --- Standing offer (4) ---
    has_standing = 1.0 if standing_price is not None else 0.0
    standing = float(standing_price or 0.0)
    standing_ext = _ext(standing) if standing else 0.0
    accept_surplus = max(0.0, standing_ext)
    own_target_ext = _ext(own_target)
    standing_beats_my_target = 1.0 if (has_standing and standing_ext >= own_target_ext) else 0.0

    # --- My target context (2) ---
    # How far the current standing is from my goal (positive = need to push more).
    own_target_distance_to_standing = own_target_ext - standing_ext if has_standing else own_target_ext

    # --- Self/opp price slices ---
    self_offers = [t for t in turn_history if t.get("speaker") == role]
    opp_offers = [t for t in turn_history if t.get("speaker") != role]

    # --- Self offer history (6) ---
    self_count_frac = len(self_offers) / msg_limit
    self_last_ext = _ext(float(self_offers[-1]["price"])) if self_offers else 0.0
    self_best_ext = max((_ext(float(o["price"])) for o in self_offers), default=0.0)
    if len(self_offers) >= 2:
        first_self_ext = _ext(float(self_offers[0]["price"]))
        prev_self_ext = _ext(float(self_offers[-2]["price"]))
        self_total_concession_ext = first_self_ext - self_last_ext
        self_last_concession_ext = prev_self_ext - self_last_ext
        self_concession_rate = self_total_concession_ext / max(1, len(self_offers) - 1)
    else:
        self_total_concession_ext = 0.0
        self_last_concession_ext = 0.0
        self_concession_rate = 0.0

    # --- Opp offer history (6) ---
    opp_count_frac = len(opp_offers) / msg_limit
    opp_last_ext = _ext(float(opp_offers[-1]["price"])) if opp_offers else 0.0
    opp_best_ext = max((_ext(float(o["price"])) for o in opp_offers), default=0.0)
    if len(opp_offers) >= 2:
        first_opp_ext = _ext(float(opp_offers[0]["price"]))
        prev_opp_ext = _ext(float(opp_offers[-2]["price"]))
        # opp_concession positive = opp gives me MORE extraction than before.
        opp_total_concession_ext = opp_last_ext - first_opp_ext
        opp_last_concession_ext = opp_last_ext - prev_opp_ext
        opp_concession_rate = opp_total_concession_ext / max(1, len(opp_offers) - 1)
    else:
        opp_total_concession_ext = 0.0
        opp_last_concession_ext = 0.0
        opp_concession_rate = 0.0

    # --- Concession dynamics (4) ---
    self_minus_opp_concession_rate = self_concession_rate - opp_concession_rate
    # Acceleration = change in per-turn concession (need at least 3 own offers)
    if len(self_offers) >= 3:
        c_recent = _ext(float(self_offers[-2]["price"])) - self_last_ext
        c_older  = _ext(float(self_offers[-3]["price"])) - _ext(float(self_offers[-2]["price"]))
        self_concession_accel = c_recent - c_older
    else:
        self_concession_accel = 0.0
    if len(opp_offers) >= 3:
        co_recent = opp_last_ext - _ext(float(opp_offers[-2]["price"]))
        co_older  = _ext(float(opp_offers[-2]["price"])) - _ext(float(opp_offers[-3]["price"]))
        opp_concession_accel = co_recent - co_older
    else:
        opp_concession_accel = 0.0

    # --- Latest turn + last-price-change (3) ---
    if turn_history:
        last_turn = turn_history[-1]
        last_by_self = 1.0 if last_turn.get("speaker") == role else 0.0
        last_was_opening = 1.0 if n_turns == 1 else 0.0
        if n_turns >= 2:
            cur_ext = _ext(float(last_turn["price"]))
            prev_ext = _ext(float(turn_history[-2]["price"]))
            last_price_change_ext = cur_ext - prev_ext
        else:
            last_price_change_ext = 0.0
    else:
        last_by_self = 0.0
        last_was_opening = 0.0
        last_price_change_ext = 0.0

    # --- Stalemate signal (1): consecutive turns standing price hasn't moved ---
    streak = 0
    for t in reversed(turn_history[:-1]):  # walk backwards from second-to-last
        if int(t["price"]) == int(turn_history[-1]["price"]):
            streak += 1
        else:
            break
    stalemate_streak = streak / msg_limit

    # --- Bid-ask gap from my perspective (2) ---
    # gap_ext positive = opp is firmer than me (gap to close).
    if self_offers and opp_offers:
        gap_ext = self_last_ext - opp_last_ext
    else:
        gap_ext = 0.0

    # --- Opp jitter signal (1): std of opp's offered extractions ---
    # High std = opp is hedging/flailing; low std = opp is committed.
    if len(opp_offers) >= 2:
        opp_exts_seq = [_ext(float(o["price"])) for o in opp_offers]
        m = sum(opp_exts_seq) / len(opp_exts_seq)
        opp_ext_std = (sum((x - m) ** 2 for x in opp_exts_seq) / (len(opp_exts_seq) - 1)) ** 0.5
    else:
        opp_ext_std = 0.0

    return [
        # Time / turn structure (4)
        turn_number_frac,
        history_count_frac,
        is_first_turn,
        is_my_last_turn,
        # Standing + target context (5)
        has_standing,
        standing_ext,
        accept_surplus,
        standing_beats_my_target,
        own_target_distance_to_standing,
        # My target (1)
        own_target_ext,
        # Self offer history (6)
        self_count_frac,
        self_last_ext,
        self_best_ext,
        self_total_concession_ext,
        self_last_concession_ext,
        self_concession_rate,
        # Opp offer history (6)
        opp_count_frac,
        opp_last_ext,
        opp_best_ext,
        opp_total_concession_ext,
        opp_last_concession_ext,
        opp_concession_rate,
        # Dynamics (4)
        self_minus_opp_concession_rate,
        self_concession_accel,
        opp_concession_accel,
        gap_ext,
        # Latest turn (3)
        last_by_self,
        last_was_opening,
        last_price_change_ext,
        # Stalemate + endgame (3)
        stalemate_streak,
        is_overall_last_turn,
        opp_ext_std,
    ][:32]


def extraction_to_price(extraction: float, own_value: float, role: str) -> int:
    """Apply log-symmetric extraction to produce a legal price.

    Buyer:  price = own_value · exp(-e)   (price ≤ own_value)
    Seller: price = own_value · exp(+e)   (price ≥ own_value)
    """
    import math as _math
    e = max(0.0, min(5.0, float(extraction)))  # log-space cap (e=5 ⇒ price ratio ~150×)
    if role == "buyer":
        price = int(round(own_value * _math.exp(-e)))
        return max(1, min(int(own_value), price))
    price = int(round(own_value * _math.exp(e)))
    return max(int(own_value), price)


def negotiation_mimic_action(
    alias: str,
    *,
    role: str,
    own_private_value: float,
    own_target_price: float | None,
    turn_history: list[dict],
    standing_price: float | None,
    turn_index: int,
    message_limit: int,
) -> dict:
    """Run the mimic and return {action, proposed_price, message_text}.

    action ∈ {"continue", "accept"}.
    proposed_price is meaningful when action == "continue".
    """
    model = load_mimic(alias)
    if model is None:
        # Fallback: honest accept-or-counter at standing
        if standing_price is None:
            price = extraction_to_price(0.10, float(own_private_value), role)
            return {"action": "continue", "proposed_price": price, "message_text": ""}
        acceptable = ((role == "buyer" and standing_price <= own_private_value) or
                      (role == "seller" and standing_price >= own_private_value))
        if acceptable:
            return {"action": "accept", "proposed_price": int(standing_price), "message_text": ""}
        return {"action": "continue", "proposed_price": int(standing_price), "message_text": ""}

    x = build_feature_vector(
        role=role,
        own_private_value=own_private_value,
        own_target_price=own_target_price,
        turn_history=turn_history,
        standing_price=standing_price,
        turn_index=turn_index,
        message_limit=message_limit,
    )
    xt = torch.tensor([x], dtype=torch.float32)
    with torch.no_grad():
        counter_ext_t, reservation_ext_t = model(xt)
    counter_ext = float(counter_ext_t.squeeze(0))
    reservation_ext = float(reservation_ext_t.squeeze(0))

    # Deterministic, reproducible RNG seeded from (alias, role, turn, values) —
    # same pattern as the original mimic, so a given state always samples the
    # same way (needed for paired/reproducible evaluation).
    h = hashlib.sha256()
    for part in (alias or "", role or "", str(int(turn_index)),
                 str(round(float(own_private_value), 4)),
                 str(round(float(standing_price or 0.0), 4))):
        h.update(part.encode("utf-8"))
    rng = _random.Random(int.from_bytes(h.digest()[:8], "big"))

    T = MIMIC_TEMPERATURE
    # --- Counter price: regressor output + Gaussian noise (auction recipe) ---
    noisy_counter = counter_ext + (rng.gauss(0.0, PRICE_NOISE * T) if T > 0 else 0.0)
    proposed_price = extraction_to_price(noisy_counter, float(own_private_value), role)

    # --- Accept decision: temperature-Bernoulli on the threshold classifier ---
    # The reservation head defines P(accept) = sigmoid(temp·(standing_ext −
    # reservation_ext)), exactly as trained by BCE. Stochastic inference samples
    # from it (auction's Bernoulli recipe); T=0 recovers the deterministic
    # threshold. A small margin shifts the operating point; deadline forces a
    # legal close on the final turn.
    action_name = "continue"
    if standing_price is not None:
        legal = ((role == "buyer" and standing_price <= own_private_value) or
                 (role == "seller" and standing_price >= own_private_value))
        if legal:
            standing_ext = _price_to_extraction(float(standing_price), float(own_private_value), role)
            turns_remaining = max(0, int(message_limit) - len(turn_history))
            margin_adj = standing_ext - reservation_ext - RESERVATION_MARGIN
            p_accept = 1.0 / (1.0 + math.exp(-ACCEPT_TEMP * margin_adj))
            if T > 0:
                # Temperature-sharpen the Bernoulli (T<1 sharper, T>1 softer).
                sa = p_accept ** (1.0 / T)
                sb = (1.0 - p_accept) ** (1.0 / T)
                p_accept = sa / max(1e-9, sa + sb)
                accept = rng.random() < p_accept
            else:
                accept = p_accept >= 0.5
            if accept or turns_remaining <= 1:
                action_name = "accept"

    return {"action": action_name, "proposed_price": proposed_price, "message_text": ""}
