import argparse
import json
import random
import re
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


HOUSING_DIR = Path(__file__).resolve().parents[1] / "dialogue_generation" / "housing"
if str(HOUSING_DIR) not in sys.path:
    sys.path.insert(0, str(HOUSING_DIR))

from jaxseq_list import jaxseq_list  # noqa: E402


def write_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def _seller_turn_count(conversation: str) -> int:
    if not isinstance(conversation, str):
        return 0
    return len(re.findall(r"\bSeller:", conversation))


def _normalize_convo_for_jaxseq(convo):
    # jaxseq_list in dialogue_generation/housing expects newer metric fields even when
    # it later overwrites the score logic. Older runs can be patched with safe defaults.
    c = dict(convo)

    c.setdefault("belief_misalignment", 0.0)
    c.setdefault("listener_alignment", 0.0)
    c.setdefault("agree", False)

    for key in (
        "big_truth",
        "garage_truth",
        "quiet_truth",
        "basement_truth",
        "backyard_truth",
        "big_pref",
        "garage_pref",
        "quiet_pref",
        "basement_pref",
        "backyard_pref",
    ):
        c.setdefault(key, False)

    need_beliefs = max(_seller_turn_count(c.get("conversation", "")), 1)
    belief_bool = c.get("belief_bool")
    if not isinstance(belief_bool, list):
        belief_bool = []
    if len(belief_bool) < need_beliefs:
        belief_bool = belief_bool + ([[]] * (need_beliefs - len(belief_bool)))
    c["belief_bool"] = belief_bool
    return c


def _split_conversations(convos, train_frac=0.8, validation_frac=0.1, seed=0):
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be between 0 and 1.")
    if not 0.0 <= validation_frac < 1.0:
        raise ValueError("validation_frac must be between 0 and 1.")
    if train_frac + validation_frac >= 1.0:
        raise ValueError("train_frac + validation_frac must be less than 1.")

    shuffled = list(convos)
    random.Random(seed).shuffle(shuffled)
    train_len = int(train_frac * len(shuffled))
    validation_len = int(validation_frac * len(shuffled))
    return {
        "train": shuffled[:train_len],
        "validation": shuffled[train_len:train_len + validation_len],
        "test": shuffled[train_len + validation_len:],
    }


def _convert_conversations(convos, split_name, metadata_dict):
    jaxseq_rows = []
    skipped = 0
    patched = 0

    for convo in tqdm(convos, desc=f"Converting {split_name} conversations"):
        normalized = _normalize_convo_for_jaxseq(convo)
        if normalized != convo:
            patched += 1
        try:
            lines = jaxseq_list(normalized)
        except Exception as exc:
            skipped += 1
            if skipped <= 10:
                idx = convo.get("index", "unknown")
                print(f"Skipping {split_name} convo index={idx}: {type(exc).__name__}: {exc}")
            continue
        for line in lines:
            metadata_dict[line["in_text"]] = [
                line["preference_distribution"],
                line["beliefs"],
                line["listener_alignment"],
            ]
            line.pop("preference_distribution", None)
            line.pop("beliefs", None)
            line.pop("listener_alignment", None)
        jaxseq_rows.extend(lines)

    return jaxseq_rows, patched, skipped


def build_datasets(input_json, train_frac=0.8, validation_frac=0.1, seed=0):
    metadata_dict = {}

    with open(input_json, "r", encoding="utf-8") as f:
        convos = json.load(f)

    convo_splits = _split_conversations(
        convos,
        train_frac=train_frac,
        validation_frac=validation_frac,
        seed=seed,
    )
    row_splits = {}
    split_stats = {}
    for offset, (split_name, split_convos) in enumerate(convo_splits.items()):
        rows, patched, skipped = _convert_conversations(split_convos, split_name, metadata_dict)
        random.Random(seed + offset).shuffle(rows)
        row_splits[split_name] = rows
        split_stats[split_name] = {
            "num_convos": len(split_convos),
            "patched_convos": patched,
            "skipped_convos": skipped,
            "num_rows": len(rows),
        }

    stats = {
        "num_convos": len(convos),
        "patched_convos": sum(item["patched_convos"] for item in split_stats.values()),
        "skipped_convos": sum(item["skipped_convos"] for item in split_stats.values()),
        "num_rows": sum(item["num_rows"] for item in split_stats.values()),
        "seed": seed,
        "train_frac": train_frac,
        "validation_frac": validation_frac,
        "test_frac": 1.0 - train_frac - validation_frac,
        "split_unit": "conversation",
        "splits": split_stats,
    }
    return row_splits["train"], row_splits["validation"], row_splits["test"], metadata_dict, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert housing conglomerated JSON into grouped train/validation/test JSONL.")
    parser.add_argument("--input_file", default="conglomerated_data.json")
    parser.add_argument("--train_out", default="train.jsonl")
    parser.add_argument("--validation_out", default="validation.jsonl")
    parser.add_argument("--test_out", default="test.jsonl")
    parser.add_argument("--metadata_out", default="metadata.json")
    parser.add_argument("--split_manifest_out", default="split_manifest.json")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--validation_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_data, validation_data, test_data, metadata_dict, stats = build_datasets(
        args.input_file,
        train_frac=args.train_frac,
        validation_frac=args.validation_frac,
        seed=args.seed,
    )

    write_jsonl(args.train_out, train_data)
    write_jsonl(args.validation_out, validation_data)
    write_jsonl(args.test_out, test_data)
    with open(args.metadata_out, "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=4)
    with open(args.split_manifest_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    print(
        f"Wrote {len(train_data)} train rows, {len(validation_data)} validation rows, {len(test_data)} test rows, "
        f"and {len(metadata_dict)} metadata entries."
    )
    print(
        f"Processed {stats['num_convos']} convos "
        f"(patched {stats['patched_convos']}, skipped {stats['skipped_convos']})."
    )
