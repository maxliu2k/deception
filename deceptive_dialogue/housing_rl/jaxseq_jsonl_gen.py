import argparse
import json
import random
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


def build_datasets(input_json, train_frac=0.8, seed=0):
    random.seed(seed)

    jaxseq_rows = []
    metadata_dict = {}

    with open(input_json, "r", encoding="utf-8") as f:
        convos = json.load(f)

    for convo in tqdm(convos, desc="Converting conversations"):
        lines = jaxseq_list(convo)
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
    return train_data, eval_data, metadata_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert housing conglomerated JSON into train/test JSONL.")
    parser.add_argument("--input_file", default="conglomerated_data.json")
    parser.add_argument("--train_out", default="train.jsonl")
    parser.add_argument("--test_out", default="test.jsonl")
    parser.add_argument("--metadata_out", default="metadata.json")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_data, eval_data, metadata_dict = build_datasets(
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

