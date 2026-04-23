from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one auction policy checkpoint per bidder model by invoking "
            "simulation/train_auction_policy_nn.py with --target-model."
        )
    )
    parser.add_argument(
        "--dataset",
        default="simulation/datasets/auction_nn_dataset_v5.jsonl",
        help="Path to the (non-nn) JSONL dataset used by train_auction_policy_nn.py",
    )
    parser.add_argument(
        "--output-dir",
        default="simulation/models/per_model",
        help="Directory for per-model checkpoints and metadata.",
    )
    parser.add_argument(
        "--base-name",
        default="auction_policy_nn",
        help="Base filename prefix for generated artifacts.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--train-logs", default="")
    parser.add_argument("--val-logs", default="")
    parser.add_argument("--test-logs", default="")
    parser.add_argument(
        "--models",
        default="",
        help="Optional comma-separated explicit model list. If omitted, inferred from dataset bidder_id.",
    )
    return parser.parse_args()


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text or "").strip()) or "model"


def _dataset_rows(dataset_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _infer_models(rows: list[dict[str, Any]]) -> list[str]:
    models: set[str] = set()
    for row in rows:
        inp = row.get("llm_input") or {}
        bidder = str(row.get("bidder_id") or inp.get("your_bidder_id") or "").strip()
        if bidder:
            models.add(bidder)
    return sorted(models)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = repo_root / dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _dataset_rows(dataset_path)
    if not rows:
        raise RuntimeError("Dataset has no rows.")

    if str(args.models or "").strip():
        models = [m.strip() for m in str(args.models).split(",") if m.strip()]
    else:
        models = _infer_models(rows)
    if not models:
        raise RuntimeError("No bidder models found.")

    trainer = Path(__file__).with_name("train_auction_policy_nn.py")
    if not trainer.exists():
        raise FileNotFoundError(f"Trainer script not found: {trainer}")

    print(f"Dataset: {dataset_path}")
    print(f"Models to train: {models}")
    print(f"Output dir: {output_dir}")

    failures: list[dict[str, str]] = []
    for model_name in models:
        model_slug = _slug(model_name)
        out_pt = output_dir / f"{args.base_name}_{model_slug}.pt"
        out_meta = output_dir / f"{args.base_name}_{model_slug}_meta.json"
        out_plot = output_dir / f"{args.base_name}_{model_slug}_loss.png"

        cmd = [
            sys.executable,
            str(trainer),
            "--dataset",
            str(dataset_path),
            "--target-model",
            model_name,
            "--output-pt",
            str(out_pt),
            "--output-meta",
            str(out_meta),
            "--loss-plot",
            str(out_plot),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--hidden",
            str(args.hidden),
            "--depth",
            str(args.depth),
            "--dropout",
            str(args.dropout),
            "--weight-decay",
            str(args.weight_decay),
            "--seed",
            str(args.seed),
            "--early-stop-patience",
            str(args.early_stop_patience),
            "--early-stop-min-delta",
            str(args.early_stop_min_delta),
        ]
        if str(args.train_logs or "").strip():
            cmd.extend(["--train-logs", str(args.train_logs)])
        if str(args.val_logs or "").strip():
            cmd.extend(["--val-logs", str(args.val_logs)])
        if str(args.test_logs or "").strip():
            cmd.extend(["--test-logs", str(args.test_logs)])

        print(f"\n=== Training {model_name} ===")
        result = subprocess.run(cmd, cwd=str(repo_root))
        if result.returncode != 0:
            failures.append({"model": model_name, "code": str(result.returncode)})
            print(f"FAILED {model_name} (exit={result.returncode})")
        else:
            print(f"OK {model_name} -> {out_pt.name}")

    if failures:
        print("\nCompleted with failures:")
        for failure in failures:
            print(f"  model={failure['model']} exit={failure['code']}")
        raise SystemExit(1)

    print("\nAll model checkpoints saved successfully.")


if __name__ == "__main__":
    main()
