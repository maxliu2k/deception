from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "PyTorch is required for this script. Install it with: pip install torch"
    ) from exc


@dataclass
class Example:
    features: list[float]
    action_label: float  # 0.0 pass, 1.0 raise
    raise_amount_label: float  # bid_amount - current_bid (0 for pass)
    min_raise: float
    bidder_id: str
    source_log_path: str


class AuctionPolicyNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        depth: int = 3,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(max(1, depth)):
            layers.append(nn.Linear(d, hidden))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        self.backbone = nn.Sequential(*layers)
        self.action_head = nn.Linear(d, 1)  # logit for raise vs pass
        self.raise_head = nn.Linear(d, 1)   # predicted raise amount above current bid

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        action_logits = self.action_head(h).squeeze(-1)
        raise_pred = self.raise_head(h).squeeze(-1)
        return action_logits, raise_pred


def apply_initialization(model: nn.Module, init_name: str) -> None:
    name = str(init_name or "default").strip().lower()
    if name in {"", "default"}:
        return
    if name in {"he", "kaiming"}:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        return
    raise ValueError(f"Unknown init scheme: {init_name}. Supported: default, he")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a neural net policy on auction imitation dataset JSONL."
    )
    parser.add_argument(
        "--dataset",
        default="simulation/datasets/auction_nn_dataset_v5.jsonl",
        help="Path to auction_nn_dataset jsonl",
    )
    parser.add_argument(
        "--output-pt",
        default="simulation/models/auction_policy_nn.pt",
        help="Path to write trained PyTorch checkpoint (.pt)",
    )
    parser.add_argument(
        "--output-meta",
        default="simulation/models/auction_policy_nn_meta.json",
        help="Path to write feature metadata json",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--batch-norm",
        type=int,
        default=1,
        help="Use BatchNorm1d in hidden layers (1=yes, 0=no).",
    )
    parser.add_argument(
        "--init",
        default="default",
        help="Weight initialization scheme: default or he",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.15,
        help="Fraction held out for test in random_split mode (separate from val).",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--target-model",
        default="",
        help="If set, train only on rows where bidder_id matches this model (e.g. GPT-5.4).",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help="Stop if validation loss does not improve for this many epochs.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="Minimum val loss decrease to count as an improvement.",
    )
    parser.add_argument(
        "--train-logs",
        default="",
        help="Comma-separated 1-based auction log indices for training (e.g. 1,2,3).",
    )
    parser.add_argument(
        "--val-logs",
        default="",
        help="Comma-separated 1-based auction log indices for validation (e.g. 4).",
    )
    parser.add_argument(
        "--test-logs",
        default="",
        help="Comma-separated 1-based auction log indices for test (e.g. 5).",
    )
    parser.add_argument(
        "--delta-loss-weight",
        type=float,
        default=0.25,
        help="Weight for raise-amount regression loss relative to action BCE loss.",
    )
    parser.add_argument(
        "--loss-plot",
        default="",
        help="Optional output PNG path for train/val loss curve. Defaults next to output-meta.",
    )
    return parser.parse_args()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _compute_stats(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    n = float(len(values))
    mean = sum(values) / n
    vmin = min(values)
    vmax = max(values)
    var = sum((x - mean) * (x - mean) for x in values) / n
    std = math.sqrt(max(0.0, var))
    return mean, std, vmin, vmax


def _extract_example(
    row: dict[str, Any],
    bidder_vocab: dict[str, int],
    leader_vocab: dict[str, int],
) -> Example:
    inp = row.get("llm_input") or {}
    target = row.get("target") or {}
    bidder_id = str(row.get("bidder_id") or inp.get("your_bidder_id") or "")
    current_leader = str(inp.get("current_leader") or "")
    all_budgets = inp.get("all_budgets") or {}
    all_counts = inp.get("all_painting_counts") or {}
    public_bid = inp.get("public_bid_table") or {}
    active = list(inp.get("active_bidders") or [])
    passed = list(inp.get("passed_bidders") or [])
    bid_history = list(inp.get("bid_history") or [])
    completed = list(inp.get("completed_painting_summaries") or [])

    bidder_names = list(all_budgets.keys())
    if bidder_id and bidder_id not in bidder_names:
        bidder_names.append(bidder_id)

    cur_bid = _safe_float(inp.get("current_bid"))
    min_legal = _safe_float(inp.get("minimum_legal_bid"))
    rem_budget = _safe_float(inp.get("your_remaining_budget"))
    paints_rem = _safe_float(inp.get("paintings_remaining"))
    total_paints = _safe_float(inp.get("total_paintings"))
    is_last = 1.0 if bool(inp.get("is_last_painting")) else 0.0
    self_count = _safe_float((all_counts or {}).get(bidder_id, 0.0))
    own_current_bid = _safe_float((public_bid.get(bidder_id) or {}).get("current_bid_this_painting"))
    own_gap_to_min = max(0.0, min_legal - own_current_bid)

    budget_values = [_safe_float(all_budgets.get(b, 0.0)) for b in bidder_names]
    count_values = [_safe_float(all_counts.get(b, 0.0)) for b in bidder_names]
    current_bid_values = [
        _safe_float((public_bid.get(b) or {}).get("current_bid_this_painting", 0.0))
        for b in bidder_names
    ]
    b_mean, b_std, b_min, b_max = _compute_stats(budget_values)
    c_mean, c_std, c_min, c_max = _compute_stats(count_values)
    cb_mean, cb_std, cb_min, cb_max = _compute_stats(current_bid_values)

    opp_budgets = sorted(
        [_safe_float(all_budgets.get(b, 0.0)) for b in bidder_names if b != bidder_id],
        reverse=True,
    )
    opp_counts = sorted(
        [_safe_float(all_counts.get(b, 0.0)) for b in bidder_names if b != bidder_id],
        reverse=True,
    )

    def _topk(vals: list[float], k: int = 4) -> list[float]:
        out = vals[:k]
        if len(out) < k:
            out.extend([0.0] * (k - len(out)))
        return out

    top_opp_budgets = _topk(opp_budgets, 4)
    top_opp_counts = _topk(opp_counts, 4)

    raises = [h for h in bid_history if str(h.get("action_type") or "") == "raise"]
    passes = [h for h in bid_history if str(h.get("action_type") or "") == "pass"]
    last_hist = bid_history[-1] if bid_history else {}
    last_action_raise = 1.0 if str(last_hist.get("action_type") or "") == "raise" else 0.0
    last_action_pass = 1.0 if str(last_hist.get("action_type") or "") == "pass" else 0.0
    last_bid_amount = _safe_float(last_hist.get("bid_amount", 0.0))
    last_bidder_is_self = 1.0 if str(last_hist.get("bidder_id") or "") == bidder_id else 0.0

    winning_bids = [
        _safe_float(item.get("winning_bid"))
        for item in completed
        if item.get("winning_bid") is not None
    ]
    wb_mean, wb_std, wb_min, wb_max = _compute_stats(winning_bids)

    bidder_oh = [0.0] * len(bidder_vocab)
    if bidder_id in bidder_vocab:
        bidder_oh[bidder_vocab[bidder_id]] = 1.0
    leader_oh = [0.0] * len(leader_vocab)
    if current_leader in leader_vocab:
        leader_oh[leader_vocab[current_leader]] = 1.0

    features = [
        cur_bid,
        min_legal,
        rem_budget,
        paints_rem,
        total_paints,
        is_last,
        self_count,
        own_current_bid,
        own_gap_to_min,
        float(len(active)),
        float(len(passed)),
        float(len(bid_history)),
        float(len(raises)),
        float(len(passes)),
        last_action_raise,
        last_action_pass,
        last_bid_amount,
        last_bidder_is_self,
        float(len(completed)),
        b_mean,
        b_std,
        b_min,
        b_max,
        c_mean,
        c_std,
        c_min,
        c_max,
        cb_mean,
        cb_std,
        cb_min,
        cb_max,
        wb_mean,
        wb_std,
        wb_min,
        wb_max,
        *top_opp_budgets,
        *top_opp_counts,
        *bidder_oh,
        *leader_oh,
    ]

    action_type = str(target.get("action_type") or "").lower()
    action_label = 1.0 if action_type == "raise" else 0.0
    raise_amount_label = _safe_float(target.get("raise_size_from_prev_bid"), 0.0)
    if not math.isfinite(raise_amount_label) or raise_amount_label < 0:
        raise_amount_label = 0.0
    min_raise = max(0.0, min_legal - cur_bid)
    return Example(
        features=features,
        action_label=action_label,
        raise_amount_label=raise_amount_label,
        min_raise=min_raise,
        bidder_id=bidder_id,
        source_log_path=str(row.get("source_log_path") or ""),
    )


def load_rows(jsonl_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def filter_rows_by_target_model(rows: list[dict[str, Any]], target_model: str) -> list[dict[str, Any]]:
    target = str(target_model or "").strip()
    if not target:
        return rows
    filtered: list[dict[str, Any]] = []
    for row in rows:
        inp = row.get("llm_input") or {}
        bidder_id = str(row.get("bidder_id") or inp.get("your_bidder_id") or "").strip()
        if bidder_id == target:
            filtered.append(row)
    return filtered


def build_vocab_from_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    bidder_set: set[str] = set()
    leader_set: set[str] = {""}
    for row in rows:
        inp = row.get("llm_input") or {}
        bidder_id = str(row.get("bidder_id") or inp.get("your_bidder_id") or "")
        if bidder_id:
            bidder_set.add(bidder_id)
        leader_set.add(str(inp.get("current_leader") or ""))
        for name in (inp.get("all_budgets") or {}).keys():
            bidder_set.add(str(name))
            leader_set.add(str(name))
    bidder_vocab = {name: idx for idx, name in enumerate(sorted(bidder_set))}
    leader_vocab = {name: idx for idx, name in enumerate(sorted(leader_set))}
    return bidder_vocab, leader_vocab


def build_tensors(examples: list[Example]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.tensor([ex.features for ex in examples], dtype=torch.float32)
    y_action = torch.tensor([ex.action_label for ex in examples], dtype=torch.float32)
    y_raise_amount = torch.tensor([ex.raise_amount_label for ex in examples], dtype=torch.float32)
    min_raise = torch.tensor([ex.min_raise for ex in examples], dtype=torch.float32)
    return x, y_action, y_raise_amount, min_raise


def split_row_indices(n: int, val_frac: float, test_frac: float, seed: int) -> tuple[list[int], list[int], list[int]]:
    idx = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    n_val = max(1, int(round(n * val_frac))) if n > 1 else 0
    n_test = max(1, int(round(n * test_frac))) if n > 2 else 0
    if n_val + n_test >= n:
        n_val = max(1, n // 5)
        n_test = max(1, n // 5)
        if n_val + n_test >= n:
            n_test = max(0, n - n_val - 1)
    val_idx = idx[:n_val]
    test_idx = idx[n_val:n_val + n_test]
    train_idx = idx[n_val + n_test:]
    if not train_idx:
        train_idx = val_idx
    return train_idx, val_idx, test_idx


def _parse_indices_csv(raw: str) -> set[int]:
    out: set[int] = set()
    text = str(raw or "").strip()
    if not text:
        return out
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            idx = int(token)
        except Exception:
            continue
        if idx > 0:
            out.add(idx)
    return out


def split_row_indices_by_log(
    rows: list[dict[str, Any]],
    *,
    train_logs: set[int],
    val_logs: set[int],
    test_logs: set[int],
) -> tuple[list[int], list[int], list[int], list[str]]:
    ordered_logs: list[str] = []
    seen: set[str] = set()
    for row in rows:
        src = str(row.get("source_log_path") or "")
        if src and src not in seen:
            seen.add(src)
            ordered_logs.append(src)
    log_to_idx = {src: i + 1 for i, src in enumerate(ordered_logs)}
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for i, row in enumerate(rows):
        log_idx = log_to_idx.get(str(row.get("source_log_path") or ""))
        if log_idx in train_logs:
            train_idx.append(i)
        elif log_idx in val_logs:
            val_idx.append(i)
        elif log_idx in test_logs:
            test_idx.append(i)
    return train_idx, val_idx, test_idx, ordered_logs


def normalize_features(
    x_train: torch.Tensor, x_val: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0).clamp_min(1e-6)
    return (x_train - mean) / std, (x_val - mean) / std, mean, std


def postprocess_raise_amount(
    action_logits: torch.Tensor,
    predicted_raise_amount: torch.Tensor,
    min_raise: torch.Tensor,
) -> torch.Tensor:
    """
    Convert (action_logits, raw raise pred) into legal raise amounts:
    - If action head says pass (sigmoid < 0.5) -> 0
    - Else clip pred at >=0, quantize to nearest multiple of min_raise,
      and floor at min_raise so a "raise" decision never silently rounds to 0.
    """
    take_raise = torch.sigmoid(action_logits) >= 0.5
    pred = torch.relu(predicted_raise_amount)
    min_r = torch.clamp(min_raise, min=0.0)
    safe_min_r = torch.where(min_r > 0.0, min_r, torch.ones_like(min_r))
    quantized = torch.round(pred / safe_min_r) * safe_min_r
    quantized = torch.where(min_r > 0.0, quantized, torch.round(pred))
    floor = torch.where(min_r > 0.0, min_r, torch.ones_like(min_r))
    raise_amount = torch.where(quantized > 0.0, quantized, floor)
    out = torch.where(take_raise, raise_amount, torch.zeros_like(raise_amount))
    return torch.clamp(out, min=0.0)


def _action_pos_weight(y_action: torch.Tensor) -> torch.Tensor:
    pos = float((y_action == 1.0).float().sum().item())
    neg = float((y_action == 0.0).float().sum().item())
    if pos <= 0.0 or neg <= 0.0:
        return torch.tensor(1.0)
    return torch.tensor(neg / pos)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).resolve().parents[1] / dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    rows = load_rows(dataset_path)
    requested_target_model = str(args.target_model or "").strip()
    if requested_target_model:
        all_bidders = sorted(
            {
                str((row.get("bidder_id") or (row.get("llm_input") or {}).get("your_bidder_id") or "")).strip()
                for row in rows
                if str((row.get("bidder_id") or (row.get("llm_input") or {}).get("your_bidder_id") or "")).strip()
            }
        )
        rows = filter_rows_by_target_model(rows, requested_target_model)
        if not rows:
            raise RuntimeError(
                f"No rows found for --target-model '{requested_target_model}'. "
                f"Available bidder_ids in dataset: {all_bidders}"
            )
    if not rows:
        raise RuntimeError("No rows loaded from dataset.")

    requested_train_logs = _parse_indices_csv(args.train_logs)
    requested_val_logs = _parse_indices_csv(args.val_logs)
    requested_test_logs = _parse_indices_csv(args.test_logs)

    if requested_train_logs or requested_val_logs or requested_test_logs:
        train_row_idx, val_row_idx, test_row_idx, ordered_logs = split_row_indices_by_log(
            rows,
            train_logs=requested_train_logs,
            val_logs=requested_val_logs,
            test_logs=requested_test_logs,
        )
        if not train_row_idx:
            raise RuntimeError("Requested --train-logs produced no training rows.")
        if not val_row_idx:
            raise RuntimeError("Requested --val-logs produced no validation rows.")
        if not test_row_idx:
            print("Warning: --test-logs produced no test rows; test metrics will be empty.")
        split_mode = "explicit_log_split"
    else:
        train_row_idx, val_row_idx, test_row_idx = split_row_indices(
            len(rows), args.val_frac, args.test_frac, args.seed
        )
        ordered_logs = []
        split_mode = "random_split"

    train_rows = [rows[i] for i in train_row_idx]
    val_rows = [rows[i] for i in val_row_idx]
    test_rows = [rows[i] for i in test_row_idx]

    bidder_vocab, leader_vocab = build_vocab_from_rows(train_rows)

    train_examples = [_extract_example(r, bidder_vocab, leader_vocab) for r in train_rows]
    val_examples = [_extract_example(r, bidder_vocab, leader_vocab) for r in val_rows]
    test_examples = [_extract_example(r, bidder_vocab, leader_vocab) for r in test_rows]

    if not train_examples:
        raise RuntimeError("No training examples after split.")

    x_train, y_action_train, y_raise_train, min_raise_train = build_tensors(train_examples)
    x_val, y_action_val, y_raise_val, min_raise_val = (
        build_tensors(val_examples)
        if val_examples
        else (torch.empty(0, x_train.shape[1]), torch.empty(0), torch.empty(0), torch.empty(0))
    )
    x_test, y_action_test, y_raise_test, min_raise_test = (
        build_tensors(test_examples)
        if test_examples
        else (torch.empty(0, x_train.shape[1]), torch.empty(0), torch.empty(0), torch.empty(0))
    )

    x_train, x_val, feat_mean, feat_std = normalize_features(x_train, x_val)
    if x_test.shape[0] > 0:
        x_test = (x_test - feat_mean) / feat_std

    train_ds = TensorDataset(x_train, y_action_train, y_raise_train, min_raise_train)
    val_ds = TensorDataset(x_val, y_action_val, y_raise_val, min_raise_val)
    test_ds = TensorDataset(x_test, y_action_test, y_raise_test, min_raise_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = AuctionPolicyNet(
        in_dim=x_train.shape[1],
        hidden=args.hidden,
        depth=args.depth,
        dropout=args.dropout,
        use_batch_norm=bool(int(args.batch_norm)),
    )
    apply_initialization(model, str(args.init))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    pos_weight = _action_pos_weight(y_action_train)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    reg_loss_fn = nn.SmoothL1Loss(reduction="mean")
    delta_w = float(args.delta_loss_weight)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    val_acc_history: list[float] = []

    print(
        f"Training examples={len(train_examples)} val_examples={len(val_examples)} "
        f"test_examples={len(test_examples)} in_dim={x_train.shape[1]} "
        f"bidders={sorted(bidder_vocab.keys())} init={str(args.init)} "
        f"pos_weight={float(pos_weight.item()):.3f}"
    )
    if requested_target_model:
        print(f"Target model filter active: bidder_id == '{requested_target_model}'")
    if ordered_logs:
        print("Log order (1-based):")
        for i, path in enumerate(ordered_logs, start=1):
            print(f"  {i}: {path}")
        print(
            f"Using explicit split train={sorted(requested_train_logs)} "
            f"val={sorted(requested_val_logs)} test={sorted(requested_test_logs)}"
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_accum = 0.0
        train_count = 0
        for xb, ya, yr, _min_r in train_loader:
            optimizer.zero_grad(set_to_none=True)
            action_logits, raise_pred = model(xb)
            bce = bce_loss_fn(action_logits, ya)
            mask = ya > 0.5
            if mask.any():
                reg = reg_loss_fn(torch.relu(raise_pred[mask]), yr[mask])
            else:
                reg = torch.zeros((), device=xb.device)
            loss = bce + delta_w * reg
            loss.backward()
            optimizer.step()
            train_loss_accum += float(loss.item()) * xb.shape[0]
            train_count += int(xb.shape[0])

        model.eval()
        val_loss_accum = 0.0
        val_count = 0
        action_logits_all: list[torch.Tensor] = []
        pred_raise_all: list[torch.Tensor] = []
        target_raise_all: list[torch.Tensor] = []
        min_raise_val_all: list[torch.Tensor] = []
        y_action_all_val: list[torch.Tensor] = []
        with torch.no_grad():
            for xb, ya, yr, min_r in val_loader:
                action_logits, raise_pred = model(xb)
                bce = bce_loss_fn(action_logits, ya)
                mask = ya > 0.5
                if mask.any():
                    reg = reg_loss_fn(torch.relu(raise_pred[mask]), yr[mask])
                else:
                    reg = torch.zeros((), device=xb.device)
                loss = bce + delta_w * reg
                val_loss_accum += float(loss.item()) * xb.shape[0]
                val_count += int(xb.shape[0])
                action_logits_all.append(action_logits)
                pred_raise_all.append(raise_pred)
                target_raise_all.append(yr)
                min_raise_val_all.append(min_r)
                y_action_all_val.append(ya)

        train_loss = train_loss_accum / max(1, train_count)
        val_loss = val_loss_accum / max(1, val_count)
        if val_count > 0:
            action_logits_cat = torch.cat(action_logits_all, dim=0)
            pred_raise_cat = torch.cat(pred_raise_all, dim=0)
            target_raise_cat = torch.cat(target_raise_all, dim=0)
            min_raise_cat = torch.cat(min_raise_val_all, dim=0)
            y_action_cat = torch.cat(y_action_all_val, dim=0)
            pred_raise_post = postprocess_raise_amount(action_logits_cat, pred_raise_cat, min_raise_cat)
            mae = float(torch.mean(torch.abs(pred_raise_post - target_raise_cat)).item())
            rmse = float(torch.sqrt(torch.mean((pred_raise_post - target_raise_cat) ** 2)).item())
            pred_action = (torch.sigmoid(action_logits_cat) >= 0.5).float()
            action_acc = float((pred_action == y_action_cat).float().mean().item())
        else:
            mae = 0.0
            rmse = 0.0
            action_acc = 0.0
        train_loss_history.append(float(train_loss))
        val_loss_history.append(float(val_loss))
        val_acc_history.append(float(action_acc))
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_action_acc={action_acc:.3f} val_raise_mae={mae:.2f} val_raise_rmse={rmse:.2f}"
        )

        improved = (best_val_loss - val_loss) > float(args.early_stop_min_delta)
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if int(args.early_stop_patience) > 0 and epochs_without_improvement >= int(args.early_stop_patience):
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(best epoch {best_epoch}, best_val_loss={best_val_loss:.4f})."
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test metrics on held-out test set.
    model.eval()
    test_loss_accum = 0.0
    test_count = 0
    test_action_logits_all: list[torch.Tensor] = []
    test_pred_raise_all: list[torch.Tensor] = []
    test_target_raise_all: list[torch.Tensor] = []
    test_min_raise_all: list[torch.Tensor] = []
    test_action_all: list[torch.Tensor] = []
    with torch.no_grad():
        for xb, ya, yr, min_r in test_loader:
            action_logits, raise_pred = model(xb)
            bce = bce_loss_fn(action_logits, ya)
            mask = ya > 0.5
            if mask.any():
                reg = reg_loss_fn(torch.relu(raise_pred[mask]), yr[mask])
            else:
                reg = torch.zeros((), device=xb.device)
            loss = bce + delta_w * reg
            test_loss_accum += float(loss.item()) * xb.shape[0]
            test_count += int(xb.shape[0])
            test_action_logits_all.append(action_logits)
            test_pred_raise_all.append(raise_pred)
            test_target_raise_all.append(yr)
            test_min_raise_all.append(min_r)
            test_action_all.append(ya)

    if test_count > 0:
        test_loss = test_loss_accum / max(1, test_count)
        test_action_logits_cat = torch.cat(test_action_logits_all, dim=0)
        test_pred_raise_cat = torch.cat(test_pred_raise_all, dim=0)
        test_target_raise_cat = torch.cat(test_target_raise_all, dim=0)
        test_min_raise_cat = torch.cat(test_min_raise_all, dim=0)
        test_action_cat = torch.cat(test_action_all, dim=0)
        test_pred_raise_post = postprocess_raise_amount(
            test_action_logits_cat, test_pred_raise_cat, test_min_raise_cat
        )
        test_mae = float(torch.mean(torch.abs(test_pred_raise_post - test_target_raise_cat)).item())
        test_rmse = float(torch.sqrt(torch.mean((test_pred_raise_post - test_target_raise_cat) ** 2)).item())
        test_action_pred = (torch.sigmoid(test_action_logits_cat) >= 0.5).float()
        test_action_acc = float((test_action_pred == test_action_cat).float().mean().item())
    else:
        test_loss = 0.0
        test_mae = 0.0
        test_rmse = 0.0
        test_action_acc = 0.0
        print("Warning: no test examples; test metrics are placeholders.")

    test_metrics = {
        "action_accuracy": test_action_acc,
        "raise_mae": test_mae,
        "raise_rmse": test_rmse,
    }
    print(
        f"test_loss={test_loss:.4f} test_action_acc={test_action_acc:.3f} "
        f"test_raise_mae={test_mae:.2f} test_raise_rmse={test_rmse:.2f}"
    )

    out_pt = Path(args.output_pt)
    out_meta = Path(args.output_meta)
    repo_root = Path(__file__).resolve().parents[1]
    if not out_pt.is_absolute():
        out_pt = repo_root / out_pt
    if not out_meta.is_absolute():
        out_meta = repo_root / out_meta
    if args.loss_plot:
        out_plot = Path(args.loss_plot)
        if not out_plot.is_absolute():
            out_plot = repo_root / out_plot
    else:
        out_plot = out_meta.with_name(out_meta.stem + "_loss.png")
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    plot_written = False
    plot_error: str | None = None
    try:
        import matplotlib.pyplot as plt  # type: ignore

        epochs_axis = list(range(1, len(train_loss_history) + 1))
        plt.figure(figsize=(7, 4.5))
        plt.plot(epochs_axis, train_loss_history, label="train_loss")
        plt.plot(epochs_axis, val_loss_history, label="val_loss")
        if best_epoch > 0:
            plt.axvline(best_epoch, color="gray", linestyle="--", linewidth=1.0, label=f"best_epoch={best_epoch}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Auction Policy Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_plot, dpi=150)
        plt.close()
        plot_written = True
    except Exception as exc:
        plot_error = str(exc)
        print(f"Loss plot skipped: {plot_error}")

    ckpt = {
        "model_state_dict": model.state_dict(),
        "in_dim": int(x_train.shape[1]),
        "hidden": int(args.hidden),
        "depth": int(args.depth),
        "dropout": float(args.dropout),
        "batch_norm": bool(int(args.batch_norm)),
        "init": str(args.init),
        "feature_mean": feat_mean,
        "feature_std": feat_std,
        "bidder_vocab": bidder_vocab,
        "leader_vocab": leader_vocab,
        "dataset_path": str(dataset_path),
        "target_model": requested_target_model or None,
        "objective": "bce_action_plus_weighted_smoothl1_raise",
        "delta_loss_weight": float(delta_w),
        "action_pos_weight": float(pos_weight.item()),
        "output_postprocess_rule": (
            "action head sigmoid>=0.5 selects raise; raise amount = quantize(relu(pred), min_raise) "
            "with floor at min_raise so a raise decision never silently rounds to 0; pass -> 0"
        ),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(len(train_loss_history)),
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "val_acc_history": val_acc_history,
        "test_loss": float(test_loss),
        "test_metrics": test_metrics,
        "seed": int(args.seed),
    }
    torch.save(ckpt, out_pt)

    meta = {
        "checkpoint_path": str(out_pt),
        "dataset_path": str(dataset_path),
        "target_model": requested_target_model or None,
        "objective": "bce_action_plus_weighted_smoothl1_raise",
        "output_postprocess_rule": (
            "action head sigmoid>=0.5 selects raise; raise amount = quantize(relu(pred), min_raise) "
            "with floor at min_raise so a raise decision never silently rounds to 0; pass -> 0"
        ),
        "examples_total": len(train_examples) + len(val_examples) + len(test_examples),
        "examples_train": len(train_examples),
        "examples_val": len(val_examples),
        "examples_test": len(test_examples),
        "in_dim": int(x_train.shape[1]),
        "hidden": int(args.hidden),
        "depth": int(args.depth),
        "dropout": float(args.dropout),
        "batch_norm": bool(int(args.batch_norm)),
        "init": str(args.init),
        "epochs": int(args.epochs),
        "epochs_ran": int(len(train_loss_history)),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "delta_loss_weight": float(delta_w),
        "action_pos_weight": float(pos_weight.item()),
        "early_stop_patience": int(args.early_stop_patience),
        "early_stop_min_delta": float(args.early_stop_min_delta),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "test_loss": float(test_loss),
        "test_metrics": test_metrics,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "val_acc_history": val_acc_history,
        "loss_plot_path": str(out_plot),
        "loss_plot_written": bool(plot_written),
        "loss_plot_error": plot_error,
        "split_mode": split_mode,
        "val_frac": float(args.val_frac),
        "test_frac": float(args.test_frac),
        "train_logs": sorted(requested_train_logs),
        "val_logs": sorted(requested_val_logs),
        "test_logs": sorted(requested_test_logs),
        "ordered_logs": ordered_logs,
        "bidder_vocab": bidder_vocab,
        "leader_vocab": leader_vocab,
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved checkpoint: {out_pt}")
    print(f"Saved metadata:   {out_meta}")
    if plot_written:
        print(f"Saved loss plot:  {out_plot}")


if __name__ == "__main__":
    main()
