import argparse
import json
import os
from pathlib import Path


def infer_deception_type(filename: str) -> str:
    name = filename.lower()
    if "deception" in name:
        return "deceptive"
    if "allcosts" in name:
        return "allcosts"
    if "honest" in name:
        return "honest"
    if "truthful" in name:
        return "truthful"
    return "nondeceptive"


def conglomerate_json_files(exp_dir, selected_models=None, selected_seeds=None, selected_deceptive=None):
    """
    Collect housing conversation runs into a single JSON file.

    Expected source layout:
        exp/<model>-<seed>/*.json
    """
    conglomerated_data = []
    exp_dir = str(exp_dir)

    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"exp_dir not found: {exp_dir}")

    for model_seed_dir in os.listdir(exp_dir):
        model_seed_path = os.path.join(exp_dir, model_seed_dir)
        if not os.path.isdir(model_seed_path):
            continue

        try:
            model_name, seed = model_seed_dir.rsplit("-", 1)
            seed = int(seed)
        except ValueError:
            print(f"Skipping directory: {model_seed_dir} (does not match '<model>-<seed>')")
            continue

        if selected_models and model_name not in selected_models:
            continue
        if selected_seeds and seed not in selected_seeds:
            continue

        for json_file in os.listdir(model_seed_path):
            if not json_file.endswith(".json"):
                continue

            deception_type = infer_deception_type(json_file)
            if selected_deceptive and deception_type not in selected_deceptive:
                continue

            json_file_path = os.path.join(model_seed_path, json_file)
            with open(json_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for run in data:
                if "listener_alignment" not in run:
                    print("WARNING: run not updated", model_seed_path)
                if "belief_differential_end" not in run or "listener_alignment_binary" not in run:
                    print(
                        "WARNING: run not updated",
                        model_seed_path,
                        "/",
                        json_file,
                        "index",
                        run.get("index", "unknown"),
                    )
                if "llama_metrics" in run:
                    del run["llama_metrics"]
                conglomerated_data.append(run)

    return conglomerated_data


def save_conglomerated_json(output_file, conglomerated_data):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(conglomerated_data, f, indent=4)


def parse_csv_list(value):
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_csv_ints(value):
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    default_exp_dir = repo_root / "dialogue_generation" / "housing" / "exp"

    parser = argparse.ArgumentParser(description="Consolidate housing dialogue JSON files for RL dataset generation.")
    parser.add_argument(
        "--exp_dir",
        default=str(default_exp_dir),
        help="Path to housing exp directory (default: deceptive_dialogue/dialogue_generation/housing/exp).",
    )
    parser.add_argument("--output_file", default="conglomerated_data.json", help="Output JSON filename.")
    parser.add_argument("--models", default="", help="Comma-separated models. Empty means all.")
    parser.add_argument("--seeds", default="", help="Comma-separated integer seeds. Empty means all.")
    parser.add_argument(
        "--deception_types",
        default="",
        help="Comma-separated types (deceptive, nondeceptive, honest, truthful, allcosts). Empty means all.",
    )
    args = parser.parse_args()

    selected_models = parse_csv_list(args.models)
    selected_seeds = parse_csv_ints(args.seeds)
    selected_deceptive = parse_csv_list(args.deception_types)

    conglomerated_data = conglomerate_json_files(
        args.exp_dir,
        selected_models=selected_models,
        selected_seeds=selected_seeds,
        selected_deceptive=selected_deceptive,
    )
    save_conglomerated_json(args.output_file, conglomerated_data)
    print(f"Conglomerated {len(conglomerated_data)} runs to {args.output_file}")

