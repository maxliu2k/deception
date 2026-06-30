"""Extract per-turn training rows from persisted negotiation episodes.

Reads runtime.pkl files from save_slots/<sid>/ and produces a JSONL dataset
in the same schema as the existing negotiation_nn_dataset_nn.jsonl:
    x: 32-float feature vector (built by build_feature_vector)
    y_action: 0=continue, 1=accept   (binary — there is no reject in this game)
    y_accept, y_continue: one-hot
    y_price: raw $
    y_extraction: log-symmetric extraction label
    y_price_delta, y_price_delta_ratio
    y_offer_mask: 1 when action involves a new price
    model_index, role_index, turn_number, episode_index, source_log_index, seed

Usage:
    python -m simulation.extract_negotiation_dataset \\
        --folder "Negotiation Real LLM v1" \\
        --out simulation/datasets/negotiation_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

from simulation.negotiation_mimic import build_feature_vector


SIM = Path(__file__).parent
RUNTIME = SIM / ".runtime" / "save_slots"
CATALOG = SIM / ".runtime" / "save_slots.json"

DEFAULT_MODEL_VOCAB = {"GPT-5.4": 0, "Grok": 1, "Llama": 2, "Opus": 3, "Pro": 4}


def build_rows_from_runtime(sid: str, pkl_path: Path, episode_index: int,
                            model_vocab: dict[str, int]) -> list[dict]:
    """Walk one negotiation episode and emit per-turn training rows."""
    try:
        with pkl_path.open("rb") as f:
            runtime = pickle.load(f)
    except Exception:
        return []
    env = runtime.get("env")
    if env is None or env.result is None:
        return []
    if str(env.config.get("mode") or "") != "buyer_seller_negotiation":
        return []
    selected = env.config.get("selected_models") or []
    if len(selected) < 2:
        return []
    buyer_alias, seller_alias = selected[0], selected[1]
    if buyer_alias not in model_vocab or seller_alias not in model_vocab:
        return []

    buyer = env.world.get("buyer_true")
    seller = env.world.get("seller_true")
    if buyer is None or seller is None:
        return []
    msg_limit = int(env.result.derived.get("message_limit") or 8)
    seed = int(env.config.get("seed") or 0)

    # Pull the actual turns + agreed_price
    turns = env.world.get("negotiation_turns") or []
    agreed_price = env.result.derived.get("agreed_price")

    rows: list[dict] = []
    standing_price = None
    for i, turn in enumerate(turns):
        speaker = str(turn.speaker).lower()
        role = speaker
        is_seller = (role == "seller")
        own_value = float(seller.baseline_value) if is_seller else float(buyer.budget)
        # Symmetric target: seller.target_price (new field) mirrors buyer.target_price
        own_target = (float(getattr(seller, "target_price", 0) or seller.asking_price)
                      if is_seller else float(buyer.target_price))
        alias = seller_alias if is_seller else buyer_alias

        # State BEFORE this turn → features (31-dim symmetric)
        history_before = [
            {"speaker": str(t.speaker).lower(), "price": int(t.proposed_price)}
            for t in turns[:i]
        ]
        x = build_feature_vector(
            role=role,
            own_private_value=own_value,
            own_target_price=own_target,
            turn_history=history_before,
            standing_price=standing_price,
            turn_index=i,
            message_limit=msg_limit,
        )
        # Determine label
        price = int(turn.proposed_price)
        is_accept = (i == len(turns) - 1) and (agreed_price is not None) and (price == int(agreed_price))
        if is_accept:
            y_action = 1; offer_mask = 0.0
        else:
            y_action = 0; offer_mask = 1.0
        prev_standing = standing_price if standing_price is not None else price
        y_price_delta = float(price - prev_standing) if standing_price is not None else 0.0
        # Log-symmetric extraction label (truly multiplicative-symmetric):
        #   buyer:  e = log(budget / price)
        #   seller: e = log(price / baseline)
        # Both positive when the price gives surplus to this role.
        import math as _math
        if own_value <= 0 or price <= 0:
            y_extraction = 0.0
        elif is_seller:
            y_extraction = _math.log(float(price) / own_value)
        else:
            y_extraction = _math.log(own_value / float(price))

        rows.append({
            "x": [round(float(v), 6) for v in x],
            "y_action": y_action,
            "y_accept": 1.0 if y_action == 1 else 0.0,
            "y_continue": 1.0 if y_action == 0 else 0.0,
            "y_price": float(price),
            "y_extraction": float(y_extraction),
            "y_price_delta": y_price_delta,
            "y_price_delta_ratio": y_price_delta / own_value if own_value else 0.0,
            "y_offer_mask": offer_mask,
            "episode_index": int(episode_index),
            "source_log_index": int(episode_index),
            "turn_number": int(i + 1),
            "model_index": int(model_vocab[alias]),
            "role_index": 1 if is_seller else 0,
            "seed": int(seed),
            "_source_slot": str(sid),
            "_alias": alias,
        })
        standing_price = float(price)
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder", required=True, help="Save-slots catalog folder name.")
    p.add_argument("--out", default="simulation/datasets/negotiation_v2.jsonl")
    p.add_argument("--meta", default="simulation/datasets/negotiation_v2_meta.json")
    args = p.parse_args()

    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    folder_id = None
    for f in catalog.get("folders") or []:
        if f.get("name") == args.folder:
            folder_id = f["folder_id"]; break
    if folder_id is None:
        print(f"ERROR: folder {args.folder!r} not found", file=sys.stderr); return 1
    slot_ids = [s["slot_id"] for s in catalog.get("slots") or [] if s.get("folder_id") == folder_id]
    print(f"Folder {args.folder!r} has {len(slot_ids)} slots", flush=True)

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.meta)

    all_rows = []
    n_eps = 0
    for ep_i, sid in enumerate(slot_ids):
        pkl = RUNTIME / sid / "runtime.pkl"
        if not pkl.exists():
            continue
        rows = build_rows_from_runtime(sid, pkl, ep_i, DEFAULT_MODEL_VOCAB)
        if rows:
            n_eps += 1
            all_rows.extend(rows)

    with out_path.open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(all_rows)} rows from {n_eps} episodes -> {out_path}", flush=True)

    # Stats by alias
    from collections import Counter
    by_alias = Counter(r["_alias"] for r in all_rows)
    print(f"Per-alias row counts: {dict(by_alias)}", flush=True)
    by_role = Counter(r["role_index"] for r in all_rows)
    print(f"Per-role row counts: {dict(by_role)} (0=buyer, 1=seller)", flush=True)

    meta = {
        "feature_names": [
            "turn_number_frac","history_count_frac","is_first_turn","is_my_last_turn",
            "has_standing","standing_ext","accept_surplus","standing_beats_my_target",
            "own_target_distance_to_standing","own_target_ext",
            "self_count_frac","self_last_ext","self_best_ext",
            "self_total_concession_ext","self_last_concession_ext","self_concession_rate",
            "opp_count_frac","opp_last_ext","opp_best_ext",
            "opp_total_concession_ext","opp_last_concession_ext","opp_concession_rate",
            "self_minus_opp_concession_rate","self_concession_accel","opp_concession_accel",
            "gap_ext",
            "last_by_self","last_was_opening","last_price_change_ext",
            "stalemate_streak","is_overall_last_turn","opp_ext_std",
        ],
        "input_dim": 32,
        "model_vocab": DEFAULT_MODEL_VOCAB,
        "action_definition": {"continue": 0, "accept": 1},
        "role_definition": {"buyer": 0, "seller": 1},
        "n_episodes": n_eps,
        "n_rows": len(all_rows),
        "per_alias_rows": dict(by_alias),
        "source_folder": args.folder,
        "schema_version": "v4-symmetric",
        "extraction_label": "y_extraction = log-symmetric extraction; buyer=log(own_value/price), seller=log(price/own_value); positive favors me",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote meta -> {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
