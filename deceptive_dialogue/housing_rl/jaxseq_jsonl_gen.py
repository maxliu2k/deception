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


def build_datasets(input_json, train_frac=0.8, seed=0):
    random.seed(seed)

    jaxseq_rows = []
    metadata_dict = {}
    skipped = 0
    patched = 0

    with open(input_json, "r", encoding="utf-8") as f:
        convos = json.load(f)

    for convo in tqdm(convos, desc="Converting conversations"):
        normalized = _normalize_convo_for_jaxseq(convo)
        if normalized != convo:
            patched += 1
        try:
            lines = jaxseq_list(normalized)
        except Exception as exc:
            skipped += 1
            if skipped <= 10:
                idx = convo.get("index", "unknown")
                print(f"Skipping convo index={idx}: {type(exc).__name__}: {exc}")
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

    random.shuffle(jaxseq_rows)

    train_len = int(train_frac * len(jaxseq_rows))
    train_data = jaxseq_rows[:train_len]
    eval_data = jaxseq_rows[train_len:]
    stats = {
        "num_convos": len(convos),
        "patched_convos": patched,
        "skipped_convos": skipped,
        "num_rows": len(jaxseq_rows),
    }
    return train_data, eval_data, metadata_dict, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert housing conglomerated JSON into train/test JSONL.")
    parser.add_argument("--input_file", default="conglomerated_data.json")
    parser.add_argument("--train_out", default="train.jsonl")
    parser.add_argument("--test_out", default="test.jsonl")
    parser.add_argument("--metadata_out", default="metadata.json")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_data, eval_data, metadata_dict, stats = build_datasets(
        args.input_file,
        train_frac=args.train_frac,
        seed=args.seed,
    )

    write_jsonl(args.train_out, train_data)
    write_jsonl(args.test_out, eval_data)
    with open(args.metadata_out, "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=4)

    print(
        f"Wrote {len(train_data)} train rows, {len(eval_data)} test rows, "
        f"and {len(metadata_dict)} metadata entries."
    )
    print(
        f"Processed {stats['num_convos']} convos "
        f"(patched {stats['patched_convos']}, skipped {stats['skipped_convos']})."
    )
