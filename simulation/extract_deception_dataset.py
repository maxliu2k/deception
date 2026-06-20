"""Walk save_slots/*/deception_episode/episode_log.json and build a training dataset.

Output: simulation/datasets/deception_dataset_v1.jsonl with rows shaped for the
two-head mimic architecture (per-attribute trust schema, 30-dim input):

  {
    "x": [30-float vector — see simulation.mimic_agent.build_deception_mimic_input],
    "y_lied":  [0|1, 0|1, 0|1, 0|1, 0|1],         # per-attribute lie label
    "y_claim": [c0, c1, c2, c3, c4],              # per-attribute raw claim value
    "bidder_index": int,    # 0..(len(bidder_vocab)-1)
    "episode_index": int,
    "round_index": int,
    "source_log": str,
  }

Input vector is permutation-invariant in opponent slot order (per-attr
aggregates + sorted opp strengths), so opponent slot assignment in the source
loadout does not affect training.

Companion `*_meta.json` includes `bidder_vocab` (alias -> index) and the
attribute/feature breakdown.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Iterable

from simulation.mimic_agent import DECEPTION_MIMIC_INPUT_DIM, build_deception_mimic_input


REPO = Path(__file__).resolve().parents[1]
SIM = REPO / "simulation"
RUNTIME_SAVE_SLOTS = SIM / ".runtime" / "save_slots"
DEFAULT_OUT = SIM / "datasets" / "deception_dataset_v1.jsonl"
DEFAULT_META = SIM / "datasets" / "deception_dataset_v1_meta.json"


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


def _build_rows(log_path: Path, payload: dict, episode_index: int) -> list[dict]:
    """Convert one episode_log.json into per (round, agent) training rows."""
    rounds = payload.get("rounds") or []
    agents = payload.get("agents") or []
    if not agents:
        return []
    num_rounds = int(payload.get("num_rounds") or 12)
    agent_id_to_index = {a["agent_id"]: int(a["agent_index"]) for a in agents}
    agent_id_to_alias = {a["agent_id"]: str(a.get("alias", a["agent_id"])) for a in agents}
    # Order agents by agent_index so canonical positions are stable.
    ordered_ids = [aid for aid, _ in sorted(agent_id_to_index.items(), key=lambda kv: kv[1])]

    # Running wins_count tally per agent_id, snapshot BEFORE each round
    # (because the claim that gets emitted in round r is conditioned on the
    # wins state at the start of round r, not after r resolves).
    wins_before_round: dict[str, int] = {aid: 0 for aid in ordered_ids}

    rows: list[dict] = []
    # Walk rounds in chronological order so wins_before_round is correct.
    for r in sorted(rounds, key=lambda x: int(x.get("round_idx", 0))):
        truth = list(r.get("truth") or [])
        if len(truth) != 5:
            continue
        trust_before = dict(r.get("trust_before") or {})
        claims = dict(r.get("claims_by_agent") or {})
        round_idx = int(r.get("round_idx", 0))
        winners_this_round = r.get("winners") or []
        for aid in ordered_ids:
            if aid not in claims or aid not in trust_before:
                continue
            own_idx = agent_id_to_index[aid]
            claim = list(claims[aid])
            if len(claim) != 5:
                continue
            raw_own = trust_before[aid]
            if isinstance(raw_own, (int, float)):
                own_trust = [float(raw_own)] * 5
            else:
                own_trust = [float(v) for v in raw_own]
            opp_trust_vecs: list[list[float]] = []
            opp_wins: list[int] = []
            for other_id in ordered_ids:
                if other_id == aid:
                    continue
                raw_opp = trust_before.get(other_id)
                if raw_opp is None:
                    continue
                if isinstance(raw_opp, (int, float)):
                    opp_trust_vecs.append([float(raw_opp)] * 5)
                else:
                    opp_trust_vecs.append([float(v) for v in raw_opp])
                opp_wins.append(int(wins_before_round.get(other_id, 0)))
            x = build_deception_mimic_input(
                truth=truth,
                own_trust=own_trust,
                opponents_trust=opp_trust_vecs,
                round_index=round_idx,
                total_rounds=num_rounds,
                own_wins_count=wins_before_round.get(aid, 0),
                opponents_wins_count=opp_wins,
            )
            y_lied = [int(round(claim[a], 2) != round(truth[a], 2)) for a in range(5)]
            y_claim = [round(float(claim[a]), 2) for a in range(5)]
            rows.append({
                "x": [round(v, 4) for v in x],
                "y_lied": y_lied,
                "y_claim": y_claim,
                "alias": agent_id_to_alias[aid],
                "agent_index": own_idx,
                "episode_index": episode_index,
                "round_index": round_idx,
                "source_log": str(log_path),
            })
        # Update running wins AFTER all rows for this round are emitted.
        for w_aid in winners_this_round:
            if w_aid in wins_before_round:
                wins_before_round[w_aid] += 1
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
                   help="Include partial episodes (D11). Default: only fully complete episodes.")
    args = p.parse_args()

    save_slots_root = Path(args.save_slots_root)
    if args.folder:
        # Filter by folder via the catalog.
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

    n_episodes = 0
    n_rows = 0
    n_skipped_incomplete = 0
    bidder_counts: Counter = Counter()
    bidder_vocab: OrderedDict[str, int] = OrderedDict()

    with out_path.open("w", encoding="utf-8") as fout:
        for log_path, payload in _iter_episode_logs(roots):
            if not args.include_incomplete and not payload.get("complete"):
                n_skipped_incomplete += 1
                continue
            rows = _build_rows(log_path, payload, episode_index=n_episodes)
            for r in rows:
                alias = r["alias"]
                if alias not in bidder_vocab:
                    bidder_vocab[alias] = len(bidder_vocab)
                r_out = {
                    "x": r["x"],
                    "y_lied": r["y_lied"],
                    "y_claim": r["y_claim"],
                    "bidder_index": bidder_vocab[alias],
                    "episode_index": r["episode_index"],
                    "round_index": r["round_index"],
                    "source_log": r["source_log"],
                }
                fout.write(json.dumps(r_out, separators=(",", ":")) + "\n")
                bidder_counts[alias] += 1
                n_rows += 1
            n_episodes += 1

    meta = {
        "version": "v2_per_attr_trust",
        "input_dim": DECEPTION_MIMIC_INPUT_DIM,
        "feature_names": [
            "truth_beach", "truth_food", "truth_pool", "truth_room", "truth_service",
            "own_trust_beach", "own_trust_food", "own_trust_pool", "own_trust_room", "own_trust_service",
            "opp_max_beach", "opp_max_food", "opp_max_pool", "opp_max_room", "opp_max_service",
            "opp_mean_beach", "opp_mean_food", "opp_mean_pool", "opp_mean_room", "opp_mean_service",
            "opp_min_beach", "opp_min_food", "opp_min_pool", "opp_min_room", "opp_min_service",
            "opp_strength_1", "opp_strength_2", "opp_strength_3", "opp_strength_4",
            "round_progress",
        ],
        "output_heads": {
            "lie_classifier": {"dim": 5, "loss": "BCE per attribute"},
            "claim_regressor": {"dim": 5, "loss": "masked MSE per attribute (mask = lie_classifier label)"},
        },
        "attribute_names": ["beach", "food", "pool", "room", "service"],
        "bidder_vocab": dict(bidder_vocab),
        "row_count": n_rows,
        "episode_count": n_episodes,
        "skipped_incomplete_episodes": n_skipped_incomplete,
        "per_alias_row_counts": dict(bidder_counts),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {n_rows} rows from {n_episodes} episodes to {out_path}")
    print(f"  Skipped {n_skipped_incomplete} incomplete episodes (use --include-incomplete to keep them)")
    print(f"  Bidder vocab: {dict(bidder_vocab)}")
    print(f"  Per-alias counts: {dict(bidder_counts.most_common())}")
    print(f"  Meta -> {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
