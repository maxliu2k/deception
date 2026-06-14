"""Build a richer deception-competition dataset for mimic training.

The exporter walks save_slots/*/deception_episode/episode_log.json and writes
one JSONL row per (round, agent). Each row keeps a flat `x` vector for model
training plus structured features, targets, outcomes, and labels for analysis.

The training feature vector `x` is the **leakage-safe** schema: it contains
only quantities the agent can observe at decision time (the same information
set the LLM sees in its prompt). It deliberately EXCLUDES the round's
population mean mu (unknowable before all claims are submitted in this
simultaneous-move game) and the agent's seat index (the game is
position-symmetric and the seat is not in the prompt).

Output defaults:
  simulation/datasets/deception_dataset_v3.jsonl
  simulation/datasets/deception_dataset_v3_meta.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Iterable, Sequence


REPO = Path(__file__).resolve().parents[1]
SIM = REPO / "simulation"
RUNTIME_SAVE_SLOTS = SIM / ".runtime" / "save_slots"
DEFAULT_OUT = SIM / "datasets" / "deception_dataset_v3.jsonl"
DEFAULT_META = SIM / "datasets" / "deception_dataset_v3_meta.json"

ATTR_NAMES = ["beach", "food", "pool", "room", "service"]
# Leakage-safe 23-dim feature vector: observed truth + visibility mask +
# preferences + trust (own + 4 opponents) + round progress + buyer rule
# parameters. No population mean (leak) and no seat index (not observable).
FEATURE_NAMES = (
    [f"observed_{a}" for a in ATTR_NAMES]
    + [f"visible_{a}" for a in ATTR_NAMES]
    + [f"pref_{a}" for a in ATTR_NAMES]
    + ["own_trust", "opp1_trust", "opp2_trust", "opp3_trust", "opp4_trust"]
    + ["round_fraction", "threshold", "penalty"]
)


def _iter_episode_logs(roots: Iterable[Path]) -> Iterable[tuple[Path, dict]]:
    for root in roots:
        if not root.exists():
            continue
        for log_path in sorted(root.rglob("episode_log.json")):
            try:
                payload = json.loads(log_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"  [warn] skipped {log_path}: {exc}", file=sys.stderr)
                continue
            yield log_path, payload


def _parse_since(value: str) -> float | None:
    """Parse a --since value into epoch seconds. Accepts an epoch number or an
    ISO date/datetime (local time). Returns None if unparseable."""
    value = value.strip()
    try:
        return float(value)  # raw epoch seconds
    except ValueError:
        pass
    from datetime import datetime
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).timestamp()
        except ValueError:
            continue
    return None


def _manifest_slot_ids(manifest_path: Path) -> set[str] | None:
    """Slot ids of the OK jobs in a run_deception_sweep manifest, or None on error."""
    try:
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(man, list):
        return None
    return {str(r.get("slot_id")) for r in man if r.get("ok") and r.get("slot_id")}


def _as_float_list(values: object, n: int, default: float = 0.0) -> list[float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return [default] * n
    out: list[float] = []
    for value in list(values)[:n]:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            out.append(default)
    while len(out) < n:
        out.append(default)
    return out


def _float_or_none(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_vec(values: Sequence[float], digits: int = 4) -> list[float]:
    return [round(float(v), digits) for v in values]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _behavior_label(
    *,
    claim_error: float,
    weighted_exaggeration: float,
    weighted_underclaim: float,
    caught: bool,
) -> str:
    if claim_error <= 0.05:
        return "truthful"
    has_exaggeration = weighted_exaggeration > 0.005
    has_underclaim = weighted_underclaim > 0.005
    if caught and has_exaggeration:
        return "caught_exaggeration"
    if has_exaggeration and has_underclaim:
        return "mixed_distortion"
    if has_exaggeration:
        return "exaggerating"
    if has_underclaim:
        return "underclaiming"
    return "distorted"


def _fallback_claim_error(claim: Sequence[float], truth: Sequence[float]) -> float:
    return sum(abs(float(c) - float(t)) for c, t in zip(claim, truth))


def _fallback_weighted_exaggeration(
    claim: Sequence[float],
    truth: Sequence[float],
    preferences: Sequence[float],
) -> float:
    return sum(float(w) * max(0.0, float(c) - float(t)) for c, t, w in zip(claim, truth, preferences))


def _fallback_weighted_underclaim(
    claim: Sequence[float],
    truth: Sequence[float],
    preferences: Sequence[float],
) -> float:
    return sum(float(w) * max(0.0, float(t) - float(c)) for c, t, w in zip(claim, truth, preferences))


def _verified_attr_name(value: object) -> str | None:
    try:
        idx = int(value)
    except (TypeError, ValueError):
        return None
    if 0 <= idx < len(ATTR_NAMES):
        return ATTR_NAMES[idx]
    return None


def _score_margin_from_scores(score_by_agent: dict) -> float | None:
    scores = [
        float(score)
        for score in score_by_agent.values()
        if score is not None
    ]
    if not scores:
        return None
    scores.sort(reverse=True)
    if len(scores) == 1:
        return scores[0]
    return scores[0] - scores[1]


def _win_type_for_round(r: dict, winners: set[str], score_margin: float | None) -> str:
    existing = r.get("win_type")
    if isinstance(existing, str) and existing:
        return existing
    if not winners:
        return "no_win"
    if len(winners) > 1:
        return "ambiguous_win"
    if score_margin is not None and score_margin < 0.05:
        return "ambiguous_win"
    return "apparent_clear_win"


def _build_rows(log_path: Path, payload: dict, episode_index: int) -> list[dict]:
    """Convert one episode_log.json into per (round, agent) training rows."""
    rounds = payload.get("rounds") or []
    agents = payload.get("agents") or []
    if not agents:
        return []

    agent_id_to_index = {a["agent_id"]: int(a["agent_index"]) for a in agents}
    agent_id_to_alias = {a["agent_id"]: str(a.get("alias", a["agent_id"])) for a in agents}
    ordered_ids = [aid for aid, _ in sorted(agent_id_to_index.items(), key=lambda kv: kv[1])]

    preferences = _as_float_list(payload.get("preferences"), 5, default=0.0)
    information_mode = str(payload.get("information_mode") or "full")
    partial_known_count = int(payload.get("partial_known_count") or 3)
    threshold = float(payload.get("threshold", 0.0) or 0.0)
    penalty = float(payload.get("penalty", 0.0) or 0.0)
    num_rounds = int(payload.get("num_rounds") or max(len(rounds), 1))
    max_agent_index = max(len(ordered_ids) - 1, 1)

    rows: list[dict] = []
    for r in rounds:
        truth = _as_float_list(r.get("truth"), 5, default=0.0)
        if len(truth) != 5:
            continue
        population_mean = _as_float_list(r.get("population_mean"), 5, default=0.0)
        round_idx = int(r.get("round_idx", 0) or 0)
        round_fraction = round_idx / max(num_rounds - 1, 1)

        trust_before = dict(r.get("trust_before") or {})
        trust_after = dict(r.get("trust_after") or {})
        claims = dict(r.get("claims_by_agent") or {})
        reasoning_map = dict(r.get("reasoning_by_agent") or {})
        verified_attrs = dict(r.get("verified_attr_by_agent") or {})
        caught_by_agent = dict(r.get("caught_by_agent") or {})
        score_by_agent = dict(r.get("score_by_agent") or {})
        true_score_by_agent = dict(r.get("true_score_by_agent") or {})
        claim_error_by_agent = dict(r.get("claim_error_by_agent") or {})
        weighted_ex_by_agent = dict(r.get("weighted_exaggeration_by_agent") or {})
        weighted_under_by_agent = dict(r.get("weighted_underclaim_by_agent") or {})
        rewards_by_agent = dict(r.get("rewards_by_agent") or {})
        observed_truth_by_agent = dict(r.get("observed_truth_by_agent") or {})
        visible_attrs_by_agent = dict(r.get("visible_attrs_by_agent") or {})
        winners = set(r.get("winners") or [])
        ideal_truth_winners = set(r.get("ideal_truth_winners") or r.get("true_winners") or [])
        truthful_apparent_winners = set(r.get("truthful_apparent_winners") or [])
        score_margin = _float_or_none(r.get("score_margin"))
        if score_margin is None:
            score_margin = _score_margin_from_scores(score_by_agent)
        win_type = _win_type_for_round(r, winners, score_margin)

        for aid in ordered_ids:
            if aid not in claims or aid not in trust_before:
                continue
            own_idx = agent_id_to_index[aid]
            claim = _as_float_list(claims.get(aid), 5, default=0.0)
            observed_raw = observed_truth_by_agent.get(aid)
            if isinstance(observed_raw, Sequence) and not isinstance(observed_raw, (str, bytes)):
                observed_truth = []
                for v in list(observed_raw)[:5]:
                    try:
                        observed_truth.append(None if v is None else float(v))
                    except (TypeError, ValueError):
                        observed_truth.append(None)
                while len(observed_truth) < 5:
                    observed_truth.append(None)
            else:
                observed_truth = list(truth)
            observed_for_x = [0.5 if v is None else float(v) for v in observed_truth]
            visible_attrs = visible_attrs_by_agent.get(aid)
            if not isinstance(visible_attrs, list):
                visible_attrs = list(range(5))
            visible_set = {int(a) for a in visible_attrs}
            visible_mask = [1 if a in visible_set else 0 for a in range(5)]

            own_trust = float(trust_before.get(aid, 0.0) or 0.0)
            # Opponents in ascending agent-index order, skipping self. MUST match
            # the live runner's `[t for j, t in enumerate(trusts) if j != i]`
            # ordering in server._build_actions_live_deception_competition so the
            # opp-trust features line up between training and inference.
            opp_ids = [ordered_ids[j] for j in range(len(ordered_ids)) if j != own_idx]
            opp_trusts = [float(trust_before.get(oid, 0.0) or 0.0) for oid in opp_ids]
            while len(opp_trusts) < 4:
                opp_trusts.append(0.0)

            claim_error = _float_or_none(claim_error_by_agent.get(aid))
            if claim_error is None:
                claim_error = _fallback_claim_error(claim, truth)
            weighted_exaggeration = _float_or_none(weighted_ex_by_agent.get(aid))
            if weighted_exaggeration is None:
                weighted_exaggeration = _fallback_weighted_exaggeration(claim, truth, preferences)
            weighted_underclaim = _float_or_none(weighted_under_by_agent.get(aid))
            if weighted_underclaim is None:
                weighted_underclaim = _fallback_weighted_underclaim(claim, truth, preferences)
            caught = bool(caught_by_agent.get(aid, False))

            # Leakage-safe 23-dim feature vector (decision-time observable only).
            x = (
                observed_for_x
                + [float(m) for m in visible_mask]
                + preferences
                + [own_trust]
                + opp_trusts[:4]
                + [round_fraction, threshold, penalty]
            )
            # y_lied is the supervision label only on VISIBLE attributes (where a
            # mismatch is a deliberate lie). On hidden attributes a mismatch is a
            # guess, not a lie, so the trainer masks the lie head there.
            target_lied = [int(round(claim[a], 2) != round(truth[a], 2)) for a in range(5)]
            target_claim = _round_vec(claim, 4)

            behavior_label = _behavior_label(
                claim_error=claim_error,
                weighted_exaggeration=weighted_exaggeration,
                weighted_underclaim=weighted_underclaim,
                caught=caught,
            )

            rows.append({
                "version": "v3",
                "x": _round_vec(x, 4),
                # Canonical training targets (names consumed by train_deception_nn).
                "y_lied": target_lied,
                "y_claim": target_claim,
                "visible_mask": list(visible_mask),
                "features": {
                    "truth": _round_vec(truth, 4),
                    "observed_truth": [None if v is None else round(float(v), 4) for v in observed_truth],
                    "visible_attrs": list(visible_attrs),
                    "information_mode": information_mode,
                    "partial_known_count": partial_known_count,
                    "population_mean": _round_vec(population_mean, 4),
                    "preferences": _round_vec(preferences, 4),
                    "threshold": round(threshold, 4),
                    "penalty": round(penalty, 4),
                    "round_index": round_idx,
                    "num_rounds": num_rounds,
                    "round_fraction": round(round_fraction, 4),
                    "own_trust": round(own_trust, 4),
                    "opponent_trusts": _round_vec(opp_trusts[:4], 4),
                    "opponent_ids": opp_ids,
                    "opponent_trust_mean": round(_mean(opp_trusts[:4]), 4),
                    "opponent_trust_min": round(min(opp_trusts[:4]) if opp_trusts else 0.0, 4),
                    "opponent_trust_max": round(max(opp_trusts[:4]) if opp_trusts else 0.0, 4),
                    "agent_index": own_idx,
                    "agent_index_norm": round(own_idx / max_agent_index, 4),
                },
                "target_claim": target_claim,
                "target_lied": target_lied,
                "behavior_label": behavior_label,
                "outcomes": {
                    "caught": caught,
                    "verified_attr_index": verified_attrs.get(aid),
                    "verified_attr": _verified_attr_name(verified_attrs.get(aid)),
                    "buyer_score": _float_or_none(score_by_agent.get(aid)),
                    "true_score": _float_or_none(true_score_by_agent.get(aid)),
                    "claim_error": round(claim_error, 4),
                    "weighted_exaggeration": round(weighted_exaggeration, 4),
                    "weighted_underclaim": round(weighted_underclaim, 4),
                    "reward": _float_or_none(rewards_by_agent.get(aid)),
                    "trust_before": round(own_trust, 4),
                    "trust_after": _float_or_none(trust_after.get(aid)),
                    "apparent_winner": aid in winners,
                    "ideal_truth_winner": aid in ideal_truth_winners,
                    "truthful_apparent_winner": aid in truthful_apparent_winners,
                    "win_type": win_type,
                    "score_margin": score_margin,
                },
                "metadata": {
                    "alias": agent_id_to_alias[aid],
                    "agent_id": aid,
                    "agent_index": own_idx,
                    "episode_index": episode_index,
                    "round_index": round_idx,
                    "source_log": str(log_path),
                    "truth_seed": payload.get("truth_seed"),
                    "session_id": payload.get("session_id"),
                    "complete_episode": bool(payload.get("complete")),
                    "raw_response": (reasoning_map.get(aid) or {}).get("raw_response"),
                    "reasoning": (reasoning_map.get(aid) or {}).get("reasoning"),
                    "parse_success": True,
                },
            })
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save-slots-root", default=str(RUNTIME_SAVE_SLOTS),
                   help="Walk this directory recursively for episode_log.json files.")
    p.add_argument("--folder", default=None,
                   help="Optional folder-name filter (matches catalog folder name).")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--meta", default=str(DEFAULT_META))
    p.add_argument("--include-incomplete", action="store_true",
                   help="Include partial episodes. Default: only fully complete episodes.")
    p.add_argument("--manifest", default=None,
                   help="Path to a run_deception_sweep manifest JSON. Restrict extraction to ONLY "
                        "the slots that sweep created (slot_id of ok jobs). Use this to build a "
                        "single-config dataset from one sweep and avoid pooling old/stale episodes.")
    p.add_argument("--since", default=None,
                   help="Only include episode logs modified at/after this time (epoch seconds or "
                        "ISO date/datetime, local). Another way to isolate a fresh run from old logs.")
    args = p.parse_args()

    allowed_sids: set[str] | None = None
    if args.manifest:
        allowed_sids = _manifest_slot_ids(Path(args.manifest))
        if not allowed_sids:
            print(f"ERROR: no ok slot_ids found in manifest {args.manifest!r}", file=sys.stderr)
            return 1
        print(f"Manifest filter: restricting to {len(allowed_sids)} slot(s) from {args.manifest}")
    since_ts: float | None = None
    if args.since:
        since_ts = _parse_since(args.since)
        if since_ts is None:
            print(f"ERROR: could not parse --since {args.since!r} "
                  f"(use epoch seconds or YYYY-MM-DD[THH:MM])", file=sys.stderr)
            return 1
        print(f"Since filter: only logs modified at/after "
              f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(since_ts))}")

    save_slots_root = Path(args.save_slots_root)
    if args.folder:
        catalog_path = SIM / ".runtime" / "save_slots.json"
        try:
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print(f"ERROR: cannot read catalog at {catalog_path} (needed for --folder filter)", file=sys.stderr)
            return 1
        folder_id = None
        for f in catalog.get("folders") or []:
            if f.get("name") == args.folder:
                folder_id = f["folder_id"]
                break
        if folder_id is None:
            print(f"ERROR: folder {args.folder!r} not found in catalog", file=sys.stderr)
            return 1
        slot_ids = {s["slot_id"] for s in (catalog.get("slots") or []) if s.get("folder_id") == folder_id}
        roots = [save_slots_root / sid for sid in slot_ids]
        print(f"Filtering to {len(roots)} slots in folder {args.folder!r} (folder_id={folder_id})")
    else:
        roots = [save_slots_root]

    out_path = Path(args.out)
    meta_path = Path(args.meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    n_episodes = 0
    n_rows = 0
    n_skipped_incomplete = 0
    n_skipped_filtered = 0
    bidder_counts: Counter = Counter()
    behavior_counts: Counter = Counter()
    win_type_counts: Counter = Counter()
    bidder_vocab: OrderedDict[str, int] = OrderedDict()

    with out_path.open("w", encoding="utf-8") as fout:
        for log_path, payload in _iter_episode_logs(roots):
            # Run-isolation filters: keep the dataset to a single sweep/config.
            if allowed_sids is not None and str(payload.get("session_id")) not in allowed_sids:
                n_skipped_filtered += 1
                continue
            if since_ts is not None:
                try:
                    if log_path.stat().st_mtime < since_ts:
                        n_skipped_filtered += 1
                        continue
                except OSError:
                    n_skipped_filtered += 1
                    continue
            if not args.include_incomplete and not payload.get("complete"):
                n_skipped_incomplete += 1
                continue
            rows = _build_rows(log_path, payload, episode_index=n_episodes)
            for row in rows:
                alias = row["metadata"]["alias"]
                if alias not in bidder_vocab:
                    bidder_vocab[alias] = len(bidder_vocab)
                row["bidder_index"] = bidder_vocab[alias]
                # Promote to top level so the trainer can group by episode for
                # leave-one-episode-out CV (it reads r["episode_index"]).
                row["episode_index"] = row["metadata"]["episode_index"]
                row["round_index"] = row["metadata"]["round_index"]
                fout.write(json.dumps(row, separators=(",", ":")) + "\n")
                bidder_counts[alias] += 1
                behavior_counts[row["behavior_label"]] += 1
                win_type_counts[row["outcomes"].get("win_type", "no_win")] += 1
                n_rows += 1
            n_episodes += 1

    meta = {
        "version": "v3",
        "input_dim": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "attribute_names": ATTR_NAMES,
        "leakage_safe": True,
        "excluded_features": {
            "population_mean": "mu is the round's claim mean across all agents; unknowable before claims are submitted (simultaneous move).",
            "agent_index": "the game is position-symmetric and the seat is not in the agent's prompt.",
        },
        "target": {
            "y_claim": {"dim": 5, "description": "Agent's emitted claim vector (regressor target)."},
            "y_lied": {"dim": 5, "description": "Per-attribute claim differs from truth at 2 decimals (classifier target)."},
            "visible_mask": {"dim": 5, "description": "1 if the attribute's truth was observable to the agent; loss mask for the lie head."},
            "behavior_label": {"type": "categorical"},
        },
        "row_schema": {
            "x": "Flat numeric training features in feature_names order (23-dim, leakage-safe).",
            "y_lied / y_claim / visible_mask": "Canonical training targets/mask consumed by train_deception_nn.",
            "features": "Structured version of x plus opponent summaries.",
            "outcomes": "Observed consequences after buyer verification and reward assignment.",
            "metadata": "Episode/source/model identifiers. raw_response is null unless raw call logging is added.",
        },
        "bidder_vocab": dict(bidder_vocab),
        "row_count": n_rows,
        "episode_count": n_episodes,
        "skipped_incomplete_episodes": n_skipped_incomplete,
        "skipped_filtered_episodes": n_skipped_filtered,
        "manifest_filter": args.manifest,
        "since_filter": args.since,
        "per_alias_row_counts": dict(bidder_counts),
        "behavior_label_counts": dict(behavior_counts),
        "win_type_counts": dict(win_type_counts),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {n_rows} rows from {n_episodes} episodes to {out_path}")
    print(f"  Skipped {n_skipped_incomplete} incomplete episodes (use --include-incomplete to keep them)")
    if allowed_sids is not None or since_ts is not None:
        print(f"  Skipped {n_skipped_filtered} episodes outside the run filter (manifest/since)")
    print(f"  Bidder vocab: {dict(bidder_vocab)}")
    print(f"  Per-alias counts: {dict(bidder_counts.most_common())}")
    print(f"  Behavior labels: {dict(behavior_counts.most_common())}")
    print(f"  Win types: {dict(win_type_counts.most_common())}")
    print(f"  Meta -> {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
