"""Standalone in-process Deception Competition simulator for reinforcement learning.

Continuous action space: the learner emits a 5-vector claim ∈ [0,1]^5 each round.
Opponents are filled by deception mimics (Mimic-{Grok,Opus,GPT-5.4,Pro,Llama}).
Mirrors env.py's deception_competition step logic without FastAPI dependencies,
so episodes step ~1000× faster than going through HTTP.

Gym-style interface:
    env = RLDeceptionEnv(num_rounds=12)
    obs, info = env.reset(learner_seat=0, opponent_aliases=["Mimic-Grok", ...])
    while not done:
        claim = policy(obs)                    # 5-vector in [0,1]
        obs, reward, done, info = env.step(claim)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from .mimic_agent import deception_mimic_claim
from .reward import (
    deception_competition_discrepancy_per_attr,
    deception_competition_reward,
    deception_competition_score,
    deception_competition_trust_from_history_per_attr,
)
from .state import (
    DECEPTION_INITIAL_TRUST,
    DECEPTION_NUM_AGENTS,
    DECEPTION_NUM_ATTRS,
    DECEPTION_TRUTH_PRECISION,
)


DEFAULT_MIMIC_ALIASES = ["Mimic-Grok", "Mimic-Opus", "Mimic-GPT-5.4", "Mimic-Pro", "Mimic-Llama"]

# Special opponent alias: a frozen RL policy from a checkpoint, used for self-play.
T5D_FROZEN_PREFIX = "T5D-frozen:"


@dataclass
class _Agent:
    agent_id: str
    alias: str
    is_learner: bool = False
    trust: list[float] = field(default_factory=lambda: [DECEPTION_INITIAL_TRUST] * DECEPTION_NUM_ATTRS)
    win_count: int = 0
    rounds_played: int = 0
    sum_discrepancy_when_won: list[float] = field(default_factory=lambda: [0.0] * DECEPTION_NUM_ATTRS)
    total_reward: float = 0.0


@dataclass
class _Round:
    round_idx: int
    truth: list[float]
    claims: dict[str, list[float]] = field(default_factory=dict)


class RLDeceptionEnv:
    """Step-based deception competition simulator. One seat is the learner."""

    def __init__(self, num_rounds: int = 12):
        self.num_rounds = int(num_rounds)
        self.agents: dict[str, _Agent] = {}
        self.agent_order: list[str] = []
        self.learner_id: str | None = None
        self.truth_schedule: list[list[float]] = []
        self.round_idx: int = 0
        self.rounds: list[_Round] = []
        self.done = False
        self._rng = random.Random()

    # ── Setup ─────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        learner_seat: int = 0,
        opponent_aliases: list[str] | None = None,
        seed: int | None = None,
    ) -> tuple[list[float], dict[str, Any]]:
        if seed is not None:
            self._rng.seed(int(seed))
        opponents = list(opponent_aliases or DEFAULT_MIMIC_ALIASES[:DECEPTION_NUM_AGENTS - 1])
        if len(opponents) < DECEPTION_NUM_AGENTS - 1:
            raise ValueError(f"need at least {DECEPTION_NUM_AGENTS - 1} opponent aliases")
        opponents = opponents[: DECEPTION_NUM_AGENTS - 1]

        # Build 5 agent slots.
        self.agents = {}
        self.agent_order = [f"agent_{i + 1}" for i in range(DECEPTION_NUM_AGENTS)]
        learner_seat = max(0, min(DECEPTION_NUM_AGENTS - 1, int(learner_seat)))
        opp_idx = 0
        for i, aid in enumerate(self.agent_order):
            if i == learner_seat:
                self.agents[aid] = _Agent(agent_id=aid, alias="Learner", is_learner=True)
            else:
                self.agents[aid] = _Agent(agent_id=aid, alias=opponents[opp_idx])
                opp_idx += 1
        self.learner_id = self.agent_order[learner_seat]

        # Pre-sample 12 truth vectors.
        seed_val = self._rng.randint(0, 2**31 - 1)
        self.truth_schedule = []
        for round_idx in range(self.num_rounds):
            local = random.Random(seed_val * 10_000 + round_idx)
            self.truth_schedule.append(
                [round(local.uniform(0.0, 1.0), DECEPTION_TRUTH_PRECISION) for _ in range(DECEPTION_NUM_ATTRS)]
            )
        self.round_idx = 0
        self.rounds = []
        self.done = False
        return self._learner_obs(), self._info()

    # ── Step ──────────────────────────────────────────────────────────────

    def step(self, claim: list[float]) -> tuple[list[float], float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode already complete; call reset().")
        if len(claim) != DECEPTION_NUM_ATTRS:
            raise ValueError(f"claim must have {DECEPTION_NUM_ATTRS} entries")
        learner_claim = [round(max(0.0, min(1.0, float(v))), DECEPTION_TRUTH_PRECISION) for v in claim]

        truth = list(self.truth_schedule[self.round_idx])

        # Gather all 5 claims.
        all_claims: dict[str, list[float]] = {}
        for i, aid in enumerate(self.agent_order):
            agent = self.agents[aid]
            if agent.is_learner:
                all_claims[aid] = learner_claim
                continue
            own_trust = list(agent.trust)
            opp_trusts = [list(self.agents[oid].trust) for j, oid in enumerate(self.agent_order) if j != i]
            opp_wins = [int(self.agents[oid].win_count) for j, oid in enumerate(self.agent_order) if j != i]
            mimic_c = deception_mimic_claim(
                agent.alias,
                list(truth),
                own_trust,
                opp_trusts,
                round_index=self.round_idx,
                total_rounds=self.num_rounds,
                own_wins_count=int(agent.win_count),
                opponents_wins_count=opp_wins,
            )
            all_claims[aid] = [round(max(0.0, min(1.0, float(v))), DECEPTION_TRUTH_PRECISION) for v in mimic_c]

        # Score + per-attribute discrepancy + winners.
        score_by: dict[str, float] = {}
        disc_by: dict[str, list[float]] = {}
        for aid in self.agent_order:
            agent = self.agents[aid]
            score_by[aid] = deception_competition_score(
                claim=all_claims[aid], trust=agent.trust
            )
            disc_by[aid] = deception_competition_discrepancy_per_attr(
                claim=all_claims[aid], truth=truth
            )

        max_score = max(score_by.values()) if score_by else 0.0
        winners = sorted([aid for aid, s in score_by.items() if s == max_score]) if score_by else []

        # Distribute rewards + update per-attribute trust history of winners.
        round_rewards: dict[str, float] = {}
        for aid in self.agent_order:
            agent = self.agents[aid]
            is_winner = aid in winners
            r = deception_competition_reward(
                is_winner=is_winner,
                num_winners=len(winners),
            )
            round_rewards[aid] = r
            agent.total_reward += r
            if is_winner:
                d_vec = disc_by[aid]
                agent.sum_discrepancy_when_won = [
                    s + d for s, d in zip(agent.sum_discrepancy_when_won, d_vec)
                ]
                agent.win_count += 1
            agent.rounds_played += 1
            agent.trust = deception_competition_trust_from_history_per_attr(
                sum_discrepancy_when_won=agent.sum_discrepancy_when_won,
                rounds_played=agent.rounds_played,
                wins_count=agent.win_count,
            )

        self.rounds.append(_Round(round_idx=self.round_idx, truth=list(truth), claims=dict(all_claims)))
        self.round_idx += 1
        learner_reward = round_rewards[self.learner_id] if self.learner_id else 0.0

        if self.round_idx >= self.num_rounds:
            self.done = True

        return self._learner_obs(), float(learner_reward), bool(self.done), self._info()

    # ── Observation ───────────────────────────────────────────────────────

    def _learner_obs(self) -> list[float]:
        """Return the learner's observation as a 1-D vector.

        Layout (19-dim):
            [t0..t4]                  ← truth this round (or zeros if done)
            [own_trust_0..4]          ← learner's per-attribute trust (5 floats)
            [opp_max_trust_0..4]      ← max opponent trust per attribute (5 floats)
            [round_idx / num_rounds]  ← episode progress in [0, 1]
            [rounds_remaining / num_rounds]
            [own_wins / num_rounds]   ← own wins so far (track-record denom)
            [opp_max_wins / num_rounds] ← strongest opp's wins so far
        """
        if self.done or self.round_idx >= self.num_rounds:
            truth = [0.0] * DECEPTION_NUM_ATTRS
        else:
            truth = self.truth_schedule[self.round_idx]
        learner = self.agents[self.learner_id] if self.learner_id else None
        learner_trust = list(learner.trust) if learner else [1.0] * DECEPTION_NUM_ATTRS
        opp_max_per_attr = [1.0] * DECEPTION_NUM_ATTRS
        opp_max_wins = 0
        if self.learner_id:
            learner_idx = self.agent_order.index(self.learner_id)
            opp_states = [
                self.agents[self.agent_order[(learner_idx + k) % DECEPTION_NUM_AGENTS]]
                for k in range(1, DECEPTION_NUM_AGENTS)
            ]
            for a in range(DECEPTION_NUM_ATTRS):
                opp_max_per_attr[a] = max(o.trust[a] for o in opp_states)
            opp_max_wins = max((o.win_count for o in opp_states), default=0)
        progress = self.round_idx / max(1, self.num_rounds)
        N = max(1, self.num_rounds)
        own_wins_norm = (learner.win_count if learner else 0) / N
        opp_max_wins_norm = opp_max_wins / N
        return (
            list(truth)
            + list(learner_trust)
            + list(opp_max_per_attr)
            + [progress, 1.0 - progress, own_wins_norm, opp_max_wins_norm]
        )

    def _info(self) -> dict[str, Any]:
        return {
            "round_idx": self.round_idx,
            "num_rounds": self.num_rounds,
            "learner_id": self.learner_id,
            "learner_total_reward": self.agents[self.learner_id].total_reward if self.learner_id else 0.0,
            "learner_trust": list(self.agents[self.learner_id].trust) if self.learner_id else [1.0] * DECEPTION_NUM_ATTRS,
            "learner_wins": self.agents[self.learner_id].win_count if self.learner_id else 0,
        }


# ── T4 imitation collector (for BC warm-start) ─────────────────────────────

def collect_t4_imitation_episode(
    env: RLDeceptionEnv,
    opponent_aliases: list[str],
    seed: int | None = None,
) -> list[tuple[list[float], list[float]]]:
    """Run one episode with the T4 heuristic in the learner seat.

    Returns (obs, t4_claim) pairs for supervised pretraining of the policy.
    """
    from .policies import DECEPTION_TIER_POLICIES
    import os as _os
    # BC source tier — defaults to T4 for back-compat; in track-record mode
    # T4 is a reactive failure-mode and BC-warmstarting from it traps PPO in
    # a bad basin. T2 (Uniform-Backoff) is the empirically dominant math tier
    # under track-record and is a much better starting policy. Tunable via env
    # var DECEPTION_BC_SOURCE so we can compare side-by-side.
    bc_alias = _os.environ.get("DECEPTION_BC_SOURCE", "Math-T4").strip()
    if bc_alias not in DECEPTION_TIER_POLICIES:
        bc_alias = "Math-T4"
    t4 = DECEPTION_TIER_POLICIES[bc_alias]
    obs, _ = env.reset(learner_seat=0, opponent_aliases=opponent_aliases, seed=seed)
    samples: list[tuple[list[float], list[float]]] = []
    while not env.done:
        # Reconstruct policy kwargs from the 19-dim obs.
        # Layout: [truth_5, own_trust_5, opp_max_trust_5, progress_1,
        #          remaining_1, own_wins_norm_1, opp_max_wins_norm_1]
        truth = obs[0:DECEPTION_NUM_ATTRS]
        own_trust = obs[DECEPTION_NUM_ATTRS : DECEPTION_NUM_ATTRS * 2]
        opp_max_trust = obs[DECEPTION_NUM_ATTRS * 2 : DECEPTION_NUM_ATTRS * 3]
        # Recover wins counts from the normalized obs slots.
        own_wins = int(round(obs[17] * env.num_rounds))
        opp_max_wins = int(round(obs[18] * env.num_rounds))
        claim = t4(
            list(truth),
            own_trust=list(own_trust),
            opponents_trust=[list(opp_max_trust)] * (DECEPTION_NUM_AGENTS - 1),
            round_index=env.round_idx,
            total_rounds=env.num_rounds,
            own_wins_count=own_wins,
            opponents_wins_count=[opp_max_wins] * (DECEPTION_NUM_AGENTS - 1),
        )
        samples.append((list(obs), [round(float(v), 2) for v in claim]))
        obs, _, done, _ = env.step(claim)
        if done:
            break
    return samples
