"""Train per-LLM negotiation mimics (reservation-threshold, single trunk + two heads).

Architecture:
    Input (32-dim) -> Linear(H) + Tanh -> Linear(H) + Tanh
      -> counter head:     Linear(1)   # counter extraction (log-symmetric)
      -> reservation head: Linear(1)   # accept threshold

Decision (binary — the action space is continue or accept; a negotiation ends
only by acceptance or by hitting the message limit, so there is no reject):
    accept iff standing_ext >= reservation_ext, else counter at counter_ext.

Training:
- Counter: MSE on y_extraction, masked by y_offer_mask (offer rows only).
- Reservation: BCE on "accept iff standing_ext >= reservation" via
  P(accept) = sigmoid(temp * (standing_ext - reservation)), over rows that
  faced a standing offer.

Outputs:
    <out-dir>/mimic_<alias>.pt          — state dict
    <out-dir>/training_summary.json     — per-alias val losses
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_DATA = "simulation/datasets/negotiation_v9.jsonl"
DEFAULT_META = "simulation/datasets/negotiation_v9_meta.json"
DEFAULT_OUT  = "simulation/models/negotiation_v6_resv"
INPUT_DIM    = 32
HIDDEN       = 64
STANDING_EXT_IDX = 5   # index of standing_ext in the 32-dim feature vector
HAS_STANDING_IDX = 4   # index of has_standing
ACCEPT_TEMP = 10.0     # sigmoid temperature for the threshold BCE


class NegotiationMimic(nn.Module):
    """Reservation-threshold mimic. Outputs (counter_ext, reservation_ext).
    Accept iff standing_ext >= reservation_ext; else counter at counter_ext.
    The action space is binary (continue or accept) — a negotiation ends only
    by acceptance or by reaching the message limit, so there is no reject."""

    def __init__(self, input_dim: int = INPUT_DIM, hidden: int = HIDDEN):
        super().__init__()
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std",  torch.ones(input_dim))
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),    nn.Tanh(),
        )
        self.ext_head         = nn.Linear(hidden, 1)  # counter extraction
        self.reservation_head = nn.Linear(hidden, 1)  # accept threshold

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = (x - self.input_mean) / (self.input_std + 1e-6)
        h = self.trunk(x)
        return self.ext_head(h).squeeze(-1), self.reservation_head(h).squeeze(-1)


def load_dataset(data_path: str) -> list[dict]:
    rows = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def train_one(alias: str, alias_rows: list[dict], out_dir: Path, args) -> dict:
    rng = random.Random(args.seed)
    rng.shuffle(alias_rows)
    n_val = max(1, len(alias_rows) // 10)
    val = alias_rows[:n_val]
    train = alias_rows[n_val:]
    if not train:
        print(f"  [{alias}] WARN: empty train set; skip.")
        return {"alias": alias, "skipped": True}

    def tensors(rows):
        X = torch.tensor([r["x"] for r in rows], dtype=torch.float32)
        Yacc = torch.tensor([1.0 if int(r["y_action"]) == 1 else 0.0 for r in rows], dtype=torch.float32)
        Ypr = torch.tensor([r["y_extraction"] for r in rows], dtype=torch.float32)
        Mpr = torch.tensor([r["y_offer_mask"] for r in rows], dtype=torch.float32)
        e_standing = X[:, STANDING_EXT_IDX]
        has_standing = X[:, HAS_STANDING_IDX]
        return X, Yacc, Ypr, Mpr, e_standing, has_standing

    X_tr, Yacc_tr, Y_pr, M_pr, Estand_tr, Hstand_tr = tensors(train)
    X_val, Yacc_val, Y_p_val, M_p_val, Estand_val, Hstand_val = tensors(val)
    class_counts = Counter(int(r["y_action"]) for r in train)  # kept for the summary

    # Standardize input features (compute on train).
    mean = X_tr.mean(dim=0)
    std  = X_tr.std(dim=0)
    std  = torch.where(std < 1e-6, torch.ones_like(std), std)

    model = NegotiationMimic(hidden=getattr(args, "hidden", HIDDEN))
    model.input_mean.copy_(mean)
    model.input_std.copy_(std)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ds = TensorDataset(X_tr, Yacc_tr, Y_pr, M_pr, Estand_tr, Hstand_tr)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    def losses(xb, yacc, ypb, mpb, estand, hstand):
        counter_ext, reservation = model(xb)
        # Counter price: MSE on extraction over rows that made an offer.
        price_err = (counter_ext - ypb) ** 2 * mpb
        price_loss = price_err.sum() / mpb.sum().clamp(min=1.0)
        # Accept threshold: BCE on "accept iff standing_ext >= reservation".
        # P(accept) = sigmoid(temp * (standing_ext - reservation)). Only rows
        # that faced a standing offer carry an accept/continue decision.
        logit = ACCEPT_TEMP * (estand - reservation)
        bce = F.binary_cross_entropy_with_logits(logit, yacc, reduction="none")
        acc_loss = (bce * hstand).sum() / hstand.sum().clamp(min=1.0)
        return price_loss, acc_loss

    ACC_LAMBDA = 1.0
    best_val_loss = math.inf
    best_state = None
    patience_left = args.patience
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yacc, ypb, mpb, estand, hstand in loader:
            price_loss, acc_loss = losses(xb, yacc, ypb, mpb, estand, hstand)
            loss = args.price_loss_weight * price_loss + ACC_LAMBDA * acc_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_price, val_acc = losses(X_val, Yacc_val, Y_p_val, M_p_val, Estand_val, Hstand_val)
            val_price = val_price.item(); val_acc = val_acc.item()
            val_total = args.price_loss_weight * val_price + ACC_LAMBDA * val_acc

        if val_total < best_val_loss - 1e-5:
            best_val_loss = val_total
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1

        if epoch % args.log_every == 0 or epoch == 1:
            print(f"  [{alias}] epoch {epoch:3d}/{args.epochs}  "
                  f"train={loss.item():.4f}  val_accept_bce={val_acc:.4f}  "
                  f"val_price={val_price:.4f}  best={best_val_loss:.4f}  pat={patience_left}/{args.patience}",
                  flush=True)

        if patience_left <= 0:
            print(f"  [{alias}] early stop at epoch {epoch}", flush=True)
            break

    if best_state:
        model.load_state_dict(best_state)
    out_path = out_dir / f"mimic_{alias}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": INPUT_DIM,
        "hidden": getattr(args, "hidden", HIDDEN),
        "alias": alias,
        "arch": "reservation_threshold",
        "best_val_loss": best_val_loss,
        "n_train": len(train),
        "n_val": len(val),
        "accept_count": class_counts.get(1, 0),
        "continue_count": class_counts.get(0, 0),
    }, out_path)
    print(f"  [{alias}] saved -> {out_path}", flush=True)
    return {
        "alias": alias,
        "best_val_loss": float(best_val_loss),
        "n_train": len(train),
        "n_val": len(val),
        "accept_rate": class_counts.get(1, 0) / max(1, len(train)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--meta", default=DEFAULT_META)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=HIDDEN)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--price-loss-weight", type=float, default=2.0,
                   help="Multiplier on the masked MSE price loss (counter-balance vs CE).")
    args = p.parse_args()

    data_path = Path(args.data)
    meta_path = Path(args.meta)
    if not data_path.exists():
        print(f"ERROR: dataset {data_path} not found", file=sys.stderr); return 1

    rows = load_dataset(str(data_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    model_vocab = meta.get("model_vocab") or {}
    idx_to_alias = {v: k for k, v in model_vocab.items()}

    by_idx: dict[int, list[dict]] = {}
    for r in rows:
        by_idx.setdefault(int(r["model_index"]), []).append(r)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    print(f"Loaded {len(rows)} rows, training {len(by_idx)} bidders on {device}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for idx in sorted(by_idx):
        alias = idx_to_alias.get(idx, f"idx_{idx}")
        print(f"\n=== Training {alias!r} ({len(by_idx[idx])} rows) ===", flush=True)
        s = train_one(alias, list(by_idx[idx]), out_dir, args)
        if not s.get("skipped"):
            summaries.append(s)

    (out_dir / "training_summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nWrote training summary -> {out_dir/'training_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
