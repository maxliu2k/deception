"""Train per-LLM deception mimic networks.

Architecture (D9):
  Input:  10-dim [t0..t4, own_trust, opp1..opp4_trust]
  Output: 5-dim lie classifier (sigmoid)  +  5-dim claim regressor (per-attribute)

Loss:
  BCE per attribute on the classifier head
  Masked MSE on the regressor head (mask = y_lied label)

One model per LLM alias. Each model is saved as a single `.pt` file containing
both heads' state dicts plus the input z-score buffers.

Usage:
    python -m simulation.train_deception_nn \
        --data simulation/datasets/deception_dataset_v1.jsonl \
        --meta simulation/datasets/deception_dataset_v1_meta.json \
        --out-dir simulation/models/deception_v1
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ── Models ──────────────────────────────────────────────────────────────────

class DeceptionMimic(nn.Module):
    """Two-head deception mimic.

    Shared trunk processes the 10-dim input. Two independent linear heads emit:
      - lie_logits: 5 floats (sigmoid → P(lie on attr a))
      - claim_regr: 5 floats (sigmoid → claim value in [0,1] when lying)
    """

    def __init__(self, input_dim: int = 10, hidden: int = 32, dropout: float = 0.2, num_attrs: int = 5):
        super().__init__()
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lie_head = nn.Linear(hidden, num_attrs)
        self.claim_head = nn.Linear(hidden, num_attrs)

    def set_scaler(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.input_mean.copy_(mean)
        self.input_std.copy_(std)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = (x - self.input_mean) / self.input_std
        h = self.trunk(x)
        lie_logits = self.lie_head(h)
        claim_raw = torch.sigmoid(self.claim_head(h))   # constrain to [0, 1]
        return lie_logits, claim_raw


def fit_scaler(X: np.ndarray, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.from_numpy(X.mean(axis=0).astype(np.float32))
    std = torch.from_numpy(X.std(axis=0).astype(np.float32))
    std = torch.clamp(std, min=eps)
    return mean, std


# ── Dataset loading ─────────────────────────────────────────────────────────

def load_dataset(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def rows_to_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array([r["x"] for r in rows], dtype=np.float32)
    y_lied = np.array([r["y_lied"] for r in rows], dtype=np.float32)
    y_claim = np.array([r["y_claim"] for r in rows], dtype=np.float32)
    return X, y_lied, y_claim


# ── Training ────────────────────────────────────────────────────────────────

def train_one(
    bidder_rows: list[dict],
    *,
    args: argparse.Namespace,
    device: torch.device,
    bidder_name: str,
    out_path: Path,
) -> dict:
    rng = random.Random(args.seed)
    rng.shuffle(bidder_rows)
    n = len(bidder_rows)
    n_val = max(1, n // 10)
    val_rows = bidder_rows[:n_val]
    train_rows = bidder_rows[n_val:]

    X_tr, y_lied_tr, y_claim_tr = rows_to_arrays(train_rows)
    X_val, y_lied_val, y_claim_val = rows_to_arrays(val_rows)

    mean, std = fit_scaler(X_tr)
    model = DeceptionMimic(input_dim=X_tr.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    model.set_scaler(mean.to(device), std.to(device))

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss(reduction="mean")
    mse = nn.MSELoss(reduction="none")

    tr_ds = TensorDataset(
        torch.from_numpy(X_tr),
        torch.from_numpy(y_lied_tr),
        torch.from_numpy(y_claim_tr),
    )
    loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    best_val = math.inf
    best_state = None
    patience = args.patience
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        for xb, yl, yc in loader:
            xb = xb.to(device)
            yl = yl.to(device)
            yc = yc.to(device)
            lie_logits, claim_pred = model(xb)
            loss_lie = bce(lie_logits, yl)
            # Masked MSE: only train regressor on attributes that were lied on.
            mask = yl.float()
            per_elem = mse(claim_pred, yc) * mask
            mask_sum = mask.sum()
            loss_claim = per_elem.sum() / mask_sum if float(mask_sum) > 0 else per_elem.sum() * 0.0
            loss = loss_lie + args.claim_loss_weight * loss_claim
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        # Validation
        model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X_val).to(device)
            yl = torch.from_numpy(y_lied_val).to(device)
            yc = torch.from_numpy(y_claim_val).to(device)
            lie_logits, claim_pred = model(xb)
            val_lie = bce(lie_logits, yl).item()
            mask = yl.float()
            per_elem = mse(claim_pred, yc) * mask
            mask_sum = mask.sum()
            val_claim = (per_elem.sum() / mask_sum).item() if float(mask_sum) > 0 else 0.0
            val_total = val_lie + args.claim_loss_weight * val_claim

        if val_total + 1e-5 < best_val:
            best_val = val_total
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            print(f"  [{bidder_name}] epoch {epoch + 1:3d}/{args.epochs}  "
                  f"val_lie={val_lie:.4f}  val_claim_mse={val_claim:.4f}  "
                  f"val_total={val_total:.4f}  best={best_val:.4f}  patience={epochs_no_improve}/{patience}")
        if epochs_no_improve >= patience:
            print(f"  [{bidder_name}] early stop at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "alias": bidder_name,
        "input_dim": int(X_tr.shape[1]),
        "hidden": args.hidden,
        "num_attrs": 5,
        "state_dict": model.state_dict(),
    }, out_path)
    print(f"  [{bidder_name}] saved -> {out_path}")
    return {
        "alias": bidder_name,
        "n_train": int(len(train_rows)),
        "n_val": int(len(val_rows)),
        "best_val_loss": float(best_val),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="simulation/datasets/deception_dataset_v1.jsonl")
    p.add_argument("--meta", default="simulation/datasets/deception_dataset_v1_meta.json")
    p.add_argument("--out-dir", default="simulation/models/deception_v1")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--claim-loss-weight", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only-alias", default="",
                   help="If set, train only this alias (e.g. 'GPT-5.4').")
    args = p.parse_args()

    data_path = Path(args.data)
    meta_path = Path(args.meta)
    if not data_path.exists():
        print(f"ERROR: dataset {data_path} not found")
        return 1
    rows = load_dataset(data_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    bidder_vocab = meta.get("bidder_vocab") or {}
    # Reverse: index -> alias
    idx_to_alias = {v: k for k, v in bidder_vocab.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    by_idx: dict[int, list[dict]] = {}
    for r in rows:
        by_idx.setdefault(int(r["bidder_index"]), []).append(r)

    print(f"Loaded {len(rows)} rows; training {len(by_idx)} bidders on device={device}")
    out_dir = Path(args.out_dir)
    summaries: list[dict] = []
    for idx, bidder_rows in sorted(by_idx.items()):
        alias = idx_to_alias.get(idx, f"bidder_{idx}")
        if args.only_alias and alias != args.only_alias:
            continue
        print(f"\n=== Training mimic for '{alias}'  ({len(bidder_rows)} rows) ===")
        if len(bidder_rows) < 20:
            print(f"  [warn] only {len(bidder_rows)} rows; results will be noisy.")
        out_path = out_dir / f"mimic_{alias}.pt"
        try:
            summary = train_one(bidder_rows, args=args, device=device, bidder_name=alias, out_path=out_path)
            summaries.append(summary)
        except Exception as exc:
            print(f"  [{alias}] FAILED: {exc!r}")

    summary_path = out_dir / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nWrote training summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
