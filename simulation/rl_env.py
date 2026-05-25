"""Standalone in-process auction simulator for reinforcement learning.

Mirrors the open-painting-auction logic from simulation/env.py without any
FastAPI/server dependencies, so episodes step ~1000x faster than going
through HTTP. The learner agent plays one seat; the other four seats are
filled by mimic agents from simulation/mimic_agent.py (validated as a
statistically-equivalent surrogate for real LLMs at V=0.041).

Gym-style interface:
    env = RLAuctionEnv(num_paintings=12, start_budget=10000)
    obs, info = env.reset(learner_seat=0, opponent_aliases=["Mimic-Grok", ...])
    while not done:
        action_idx = policy(obs)            # 0 = PASS, 1 = RAISE-min
        obs, reward, done, info = env.step(action_idx)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from .mimic_agent import build_feature_vector, mimic_bid
from .state import OpenAuctionAction


MIMIC_ALIASES = ["Mimic-Grok", "Mimic-Opus", "Mimic-GPT-5.4", "Mimic-Pro", "Mimic-Llama"]

# Action space (same shape as T1-T4 — no jump bids, only pass-or-min-raise).
ACTION_PASS = 0
ACTION_RAISE = 1
N_ACTIONS = 2


@dataclass
class _BidderState:
    bidder_id: str
    alias: str
    remaining_budget: int
    paintings_won: int = 0
    is_learner: bool = False


@dataclass
class _RoundState:
    painting_id: str
    current_bid: int = 0
    current_leader: str | None = None
    active: list[str] = field(default_factory=list)
    passed: list[str] = field(default_factory=list)
    turn_order: list[str] = field(default_factory=list)
    turn_index: int = 0
    bid_history: list[dict] = field(default_factory=list)


def _min_raise(current_bid: int) -> int:
    if current_bid < 1000:
        return 50
    if current_bid < 3000:
        return 100
    return 250


class RLAuctionEnv:
    """Step-based auction simulator. One seat is the learner; the rest play via mimic_bid()."""

    OPENING_BID = 100
    NUM_BIDDERS = 5

    def __init__(self, num_paintings: int = 12, start_budget: int = 10000):
        self.num_paintings = int(num_paintings)
        self.start_budget = int(start_budget)
        self.bidders: dict[str, _BidderState] = {}
        self.round: _RoundState | None = None
        self.done = False
        self.learner_id: str | None = None
        self.painting_idx = 0
        self.results: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        learner_seat: int = 0,
        opponent_aliases: list[str] | None = None,
        rng_seed: int | None = None,
    ) -> tuple[list[float], dict]:
        """Start a new auction.

        learner_seat: 0..NUM_BIDDERS-1 — which bidder index is the learner.
        opponent_aliases: list of (NUM_BIDDERS - 1) mimic aliases filling the
            other seats in order. Defaults to a deterministic rotation.
        """
        if rng_seed is not None:
            random.seed(int(rng_seed))
        if opponent_aliases is None:
            opponent_aliases = [a for a in MIMIC_ALIASES if a != MIMIC_ALIASES[learner_seat % len(MIMIC_ALIASES)]]
        if len(opponent_aliases) != self.NUM_BIDDERS - 1:
            raise ValueError(
                f"opponent_aliases must have exactly {self.NUM_BIDDERS - 1} entries; got {len(opponent_aliases)}"
            )
        self.bidders = {}
        opp_iter = iter(opponent_aliases)
        for seat in range(self.NUM_BIDDERS):
            bid_id = f"bidder_{seat + 1}"
            if seat == learner_seat:
                alias = "Learner"
                self.learner_id = bid_id
            else:
                alias = next(opp_iter)
            self.bidders[bid_id] = _BidderState(
                bidder_id=bid_id,
                alias=alias,
                remaining_budget=self.start_budget,
                is_learner=(alias == "Learner"),
            )
        self.done = False
        self.painting_idx = 0
        self.results = []
        self._start_painting()
        self._advance_to_learner_or_end()
        if self.done:
            return [0.0] * 32, {"final_wins": 0}
        return self._observe_learner(), {"painting_index": self.painting_idx}

    def step(self, action_idx: int) -> tuple[list[float], float, bool, dict]:
        """Learner takes ``action_idx`` (0=PASS, 1=RAISE-min).

        The env then auto-runs opponents until either the next learner turn
        or the auction terminates. Reward is the number of paintings the
        learner won during this transition (always 0 or 1 in practice).
        """
        if self.done:
            return [0.0] * 32, 0.0, True, {"final_wins": self.bidders[self.learner_id].paintings_won if self.learner_id else 0}
        learner = self.bidders[self.learner_id]
        wins_before = learner.paintings_won
        if action_idx == ACTION_RAISE:
            min_req = self._min_next_bid()
            if min_req <= learner.remaining_budget:
                self._apply_action(
                    self.learner_id,
                    OpenAuctionAction(
                        action_type="raise",
                        bid_amount=int(min_req),
                        message_text=f"BID {min_req}",
                    ),
                )
            else:
                # Forced PASS — can't afford the minimum legal raise.
                self._apply_action(
                    self.learner_id,
                    OpenAuctionAction(action_type="pass", bid_amount=None, message_text="PASS"),
                )
        else:
            self._apply_action(
                self.learner_id,
                OpenAuctionAction(action_type="pass", bid_amount=None, message_text="PASS"),
            )
        self._advance_turn()
        self._advance_to_learner_or_end()
        wins_after = self.bidders[self.learner_id].paintings_won
        reward = float(wins_after - wins_before)
        info = {"painting_index": self.painting_idx, "wins": wins_after}
        if self.done:
            return [0.0] * 32, reward, True, {**info, "final_wins": wins_after}
        return self._observe_learner(), reward, False, info

    def final_summary(self) -> dict:
        return {
            "results": list(self.results),
            "final": {
                bid: {
                    "alias": b.alias,
                    "won": b.paintings_won,
                    "spent": self.start_budget - b.remaining_budget,
                }
                for bid, b in self.bidders.items()
            },
        }

    # ------------------------------------------------------------------
    # Internal step machinery
    # ------------------------------------------------------------------

    def _start_painting(self) -> None:
        bid_ids = list(self.bidders.keys())
        self.round = _RoundState(
            painting_id=f"painting_{self.painting_idx + 1}",
            current_bid=0,
            current_leader=None,
            active=list(bid_ids),
            passed=[],
            turn_order=list(bid_ids),
            turn_index=0,
            bid_history=[],
        )

    def _advance_turn(self) -> None:
        if not self.round or not self.round.active:
            return
        total = len(self.round.turn_order)
        next_idx = self.round.turn_index
        for _ in range(total):
            next_idx = (next_idx + 1) % total
            if self.round.turn_order[next_idx] in self.round.active:
                self.round.turn_index = next_idx
                return

    def _min_next_bid(self) -> int:
        assert self.round is not None
        if self.round.current_leader is None:
            return self.OPENING_BID
        return int(self.round.current_bid) + _min_raise(int(self.round.current_bid))

    def _current_bidder_id(self) -> str | None:
        if self.round is None or not self.round.active:
            return None
        return self.round.turn_order[self.round.turn_index]

    def _resolve_painting(self) -> bool:
        """Try to close out the current painting. Returns True if it advanced to the next one."""
        if self.round is None:
            return False
        # Single remaining active bidder wins at current price.
        if self.round.current_leader is not None and len(self.round.active) == 1:
            winner = self.round.current_leader
            wb = self.bidders[winner]
            price = int(self.round.current_bid)
            wb.remaining_budget = max(0, wb.remaining_budget - price)
            wb.paintings_won += 1
            self.results.append(
                {
                    "painting": self.round.painting_id,
                    "winner_id": winner,
                    "winner_alias": wb.alias,
                    "price": price,
                }
            )
            self.painting_idx += 1
            return True
        # All bidders passed with no leader -> unsold.
        if self.round.current_leader is None and not self.round.active:
            self.results.append(
                {
                    "painting": self.round.painting_id,
                    "winner_id": None,
                    "winner_alias": None,
                    "price": 0,
                }
            )
            self.painting_idx += 1
            return True
        return False

    def _apply_action(self, bidder_id: str, action: OpenAuctionAction) -> None:
        assert self.round is not None
        bs = self.bidders[bidder_id]
        if action.action_type == "pass":
            if bidder_id in self.round.active:
                self.round.active.remove(bidder_id)
            if bidder_id not in self.round.passed:
                self.round.passed.append(bidder_id)
            self.round.bid_history.append(
                {
                    "bidder_id": bidder_id,
                    "action_type": "pass",
                    "bid_before": int(self.round.current_bid),
                }
            )
            return
        # action is "raise"
        bid_amount = int(action.bid_amount or 0)
        min_req = self._min_next_bid()
        if bid_amount < min_req or bid_amount > bs.remaining_budget:
            # Treat invalid raise as forced PASS (consistent with env.py's
            # rejection-then-bidder-drops behavior, minus the invalid counter).
            if bidder_id in self.round.active:
                self.round.active.remove(bidder_id)
            if bidder_id not in self.round.passed:
                self.round.passed.append(bidder_id)
            self.round.bid_history.append(
                {
                    "bidder_id": bidder_id,
                    "action_type": "pass",
                    "bid_before": int(self.round.current_bid),
                    "invalidated": True,
                }
            )
            return
        self.round.bid_history.append(
            {
                "bidder_id": bidder_id,
                "action_type": "raise",
                "bid_amount": bid_amount,
                "bid_before": int(self.round.current_bid),
            }
        )
        self.round.current_bid = bid_amount
        self.round.current_leader = bidder_id

    def _public_bid_table(self) -> dict[str, dict]:
        current_bids: dict[str, int | None] = {}
        if self.round is not None:
            for h in self.round.bid_history:
                if h.get("action_type") == "raise":
                    current_bids[h["bidder_id"]] = int(h["bid_amount"])
        return {
            bid: {
                "current_bid_this_painting": current_bids.get(bid),
                "remaining_budget": int(b.remaining_budget),
                "paintings_won": int(b.paintings_won),
            }
            for bid, b in self.bidders.items()
        }

    def _mimic_act(self, bidder_id: str) -> None:
        assert self.round is not None
        bs = self.bidders[bidder_id]
        all_budgets = {bid: int(b.remaining_budget) for bid, b in self.bidders.items()}
        all_counts = {bid: int(b.paintings_won) for bid, b in self.bidders.items()}
        action = mimic_bid(
            alias=bs.alias,
            bidder_id=bidder_id,
            your_budget=int(bs.remaining_budget),
            your_count=int(bs.paintings_won),
            current_bid=int(self.round.current_bid),
            current_leader=self.round.current_leader,
            active_bidders=list(self.round.active),
            bid_history=list(self.round.bid_history),
            all_budgets=all_budgets,
            all_counts=all_counts,
            public_bid_table=self._public_bid_table(),
            painting_number=self.painting_idx + 1,
            total_paintings=self.num_paintings,
            paintings_remaining=self.num_paintings - self.painting_idx,
            is_last_painting=(self.painting_idx + 1 >= self.num_paintings),
            min_next_bid=int(self._min_next_bid()),
            start_budget=int(self.start_budget),
        )
        self._apply_action(bidder_id, action)

    def _observe_learner(self) -> list[float]:
        """Build the 32-dim feature vector the learner sees — identical shape to mimics."""
        assert self.round is not None and self.learner_id is not None
        bs = self.bidders[self.learner_id]
        all_budgets = {bid: int(b.remaining_budget) for bid, b in self.bidders.items()}
        all_counts = {bid: int(b.paintings_won) for bid, b in self.bidders.items()}
        return build_feature_vector(
            bidder_id=self.learner_id,
            your_budget=int(bs.remaining_budget),
            your_count=int(bs.paintings_won),
            current_bid=int(self.round.current_bid),
            current_leader=self.round.current_leader,
            active_bidders=list(self.round.active),
            bid_history=list(self.round.bid_history),
            all_budgets=all_budgets,
            all_counts=all_counts,
            public_bid_table=self._public_bid_table(),
            painting_number=self.painting_idx + 1,
            total_paintings=self.num_paintings,
            paintings_remaining=self.num_paintings - self.painting_idx,
            is_last_painting=(self.painting_idx + 1 >= self.num_paintings),
            min_next_bid=int(self._min_next_bid()),
            start_budget=int(self.start_budget),
        )

    def _advance_to_learner_or_end(self) -> None:
        """Run opponent turns / resolve paintings until learner's turn or auction terminates."""
        guard = 0
        while not self.done:
            guard += 1
            if guard > 2000:
                # Defensive: prevents an infinite loop if logic ever desyncs.
                self.done = True
                return
            if self.round is None:
                if self.painting_idx >= self.num_paintings:
                    self.done = True
                    return
                self._start_painting()
                continue
            if self._resolve_painting():
                if self.painting_idx >= self.num_paintings:
                    self.done = True
                    return
                self._start_painting()
                continue
            current = self._current_bidder_id()
            if current is None:
                # Defensive: no active bidder and not resolvable -> end painting.
                self.painting_idx += 1
                if self.painting_idx >= self.num_paintings:
                    self.done = True
                    return
                self._start_painting()
                continue
            if self.bidders[current].is_learner:
                return
            self._mimic_act(current)
            self._advance_turn()
