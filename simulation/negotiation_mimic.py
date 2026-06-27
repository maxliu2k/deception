"""Negotiation mimic NN inference (fully symmetric, role-agnostic).

The mimic processes both roles identically:
  * 31-dim input vector built from role-agnostic features (no role flag)
  * 3-class action head (continue / accept / reject)
  * scalar extraction-ratio head applied symmetrically downstream

Symmetric extraction mapping:
    buyer  price = round(own_value * (1.0 - e))     clipped to [1, own_value]
    seller price = round(own_value * (1.0 + e))     clipped to [own_value, inf)

All numeric features are expressed in "extraction relative to my own_value"
units so the same value distribution applies regardless of role.
"""
from __future__ import annotations

import hashlib
import random as _random
from pathlib import Path

import torch
import torch.nn as nn


INPUT_DIM = 32
ACTION_DIM = 3
HIDDEN = 64
MODELS_DIR = Path(__file__).parent / "models" / "negotiation_v5_sym"
_NET_CACHE: dict[str, "_NegotiationMimic"] = {}


class _NegotiationMimic(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, hidden: int = HIDDEN, action_dim: int = ACTION_DIM):
        super().__init__()
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.action_head = nn.Linear(hidden, action_dim)
        self.ext_head = nn.Linear(hidden, 1)  # extraction ratio (>= 0)

    def forward(self, x: torch.Tensor):
        x = (x - self.input_mean) / (self.input_std + 1e-6)
        h = self.trunk(x)
        return self.action_head(h), self.ext_head(h).squeeze(-1)


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def load_mimic(alias: str) -> _NegotiationMimic | None:
    bare = alias[len("Mimic-"):] if alias.startswith("Mimic-") else alias
    if bare in _NET_CACHE:
        return _NET_CACHE[bare]
    path = MODELS_DIR / f"mimic_{_safe_filename(bare)}.pt"
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sd = payload.get("state_dict") or payload.get("policy")
    if sd is None:
        return None
    model = _NegotiationMimic(
        input_dim=int(payload.get("input_dim", INPUT_DIM)),
        hidden=int(payload.get("hidden", HIDDEN)),
        action_dim=int(payload.get("action_dim", ACTION_DIM)),
    )
    model.load_state_dict(sd)
    model.eval()
    _NET_CACHE[bare] = model
    return model


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

    action ∈ {"continue", "accept", "reject"}.
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
        action_logits, ext_pred = model(xt)
    probs = torch.softmax(action_logits.squeeze(0), dim=-1).tolist()
    raw_ext = float(ext_pred.squeeze(0))

    h = hashlib.sha256()
    h.update((alias or "").encode("utf-8"))
    h.update((role or "").encode("utf-8"))
    h.update(str(int(turn_index)).encode("utf-8"))
    h.update(str(round(float(own_private_value), 4)).encode("utf-8"))
    h.update(str(round(float(standing_price or 0.0), 4)).encode("utf-8"))
    rng = _random.Random(int.from_bytes(h.digest()[:8], "big"))
    r = rng.random()
    cum = 0.0
    action_idx = 0
    for i, p in enumerate(probs):
        cum += p
        if r < cum:
            action_idx = i
            break
    action_name = ["continue", "accept", "reject"][action_idx]

    proposed_price = extraction_to_price(raw_ext, float(own_private_value), role)

    if action_name == "accept" and standing_price is None:
        action_name = "continue"
    if action_name == "accept" and standing_price is not None:
        legal = ((role == "buyer" and standing_price <= own_private_value) or
                 (role == "seller" and standing_price >= own_private_value))
        if not legal:
            action_name = "continue"

    return {"action": action_name, "proposed_price": proposed_price, "message_text": ""}
