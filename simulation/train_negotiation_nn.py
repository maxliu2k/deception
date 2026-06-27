"""Train per-LLM negotiation mimics on the multi-head dataset.

Architecture (single shared trunk + two heads):
    Input (32-dim)
      -> Linear(64) + Tanh -> Linear(64) + Tanh
      -> action head: Linear(3)     # logits over {continue, accept, reject}
      -> price head: Linear(1)      # y_price_ratio (price / own_private_value)

Training:
- Per-LLM model (5 .pt checkpoints under models/negotiation_v1/).
- Action: cross-entropy on y_action (3-way).
- Price: MSE on y_price_ratio, MASKED by y_offer_mask (1 when offer, 0 otherwise).
  Reject/accept rows contribute zero to the price loss.
- Action over-represents `continue` (0); we class-weight CE inversely to class
  frequency so accept/reject don't get ignored.

Outputs:
    models/negotiation_v1/mimic_<alias>.pt        — state dict
    models/negotiation_v1/training_summary.json   — per-alias val MSE/CE
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


DEFAULT_DATA = "simulation/datasets/negotiation_v5_sym.jsonl"
DEFAULT_META = "simulation/datasets/negotiation_v5_sym_meta.json"
DEFAULT_OUT  = "simulation/models/negotiation_v5_sym"
INPUT_DIM    = 32
ACTION_DIM   = 3
HIDDEN       = 64


class NegotiationMimic(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, hidden: int = HIDDEN, action_dim: int = ACTION_DIM):
        super().__init__()
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std",  torch.ones(input_dim))
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),    nn.Tanh(),
        )
        self.action_head = nn.Linear(hidden, action_dim)
        self.ext_head    = nn.Linear(hidden, 1)  # extraction ratio (>= 0 at inference)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = (x - self.input_mean) / (self.input_std + 1e-6)
        h = self.trunk(x)
        return self.action_head(h), self.ext_head(h).squeeze(-1)


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

    X_tr  = torch.tensor([r["x"] for r in train], dtype=torch.float32)
    Y_act = torch.tensor([r["y_action"] for r in train], dtype=torch.long)
    Y_pr  = torch.tensor([r["y_extraction"] for r in train], dtype=torch.float32)
    M_pr  = torch.tensor([r["y_offer_mask"] for r in train], dtype=torch.float32)

    X_val   = torch.tensor([r["x"] for r in val], dtype=torch.float32)
    Y_a_val = torch.tensor([r["y_action"] for r in val], dtype=torch.long)
    Y_p_val = torch.tensor([r["y_extraction"] for r in val], dtype=torch.float32)
    M_p_val = torch.tensor([r["y_offer_mask"] for r in val], dtype=torch.float32)

    # Action class weights via effective-number-of-samples (Cui et al. 2019).
    # w_k = (1 - β) / (1 - β^n_k), with β=0.999. Smoother than inverse-frequency
    # — avoids the over-correction that made the v4/v5 mimics over-predict rare
    # classes (mimic accept rate was 2× LLM accept rate under raw 1/freq weights).
    class_counts = Counter(int(c) for c in Y_act.tolist())
    beta = 0.999
    weights = torch.tensor(
        [(1.0 - beta) / max(1e-12, 1.0 - beta ** max(1, class_counts.get(i, 0)))
         for i in range(ACTION_DIM)],
        dtype=torch.float32,
    )
    weights = weights / weights.mean()
    # Empirical target action distribution for the KL distribution-matching
    # auxiliary loss (forces mimic's predicted action distribution to match
    # the LLM's empirical one; directly minimises the chi² gap we measure).
    n_total = sum(class_counts.values())
    target_action_dist = torch.tensor(
        [class_counts.get(i, 0) / max(1, n_total) for i in range(ACTION_DIM)],
        dtype=torch.float32,
    ).clamp(min=1e-6)
    target_action_dist = target_action_dist / target_action_dist.sum()

    # Standardize input features (compute on train).
    mean = X_tr.mean(dim=0)
    std  = X_tr.std(dim=0)
    std  = torch.where(std < 1e-6, torch.ones_like(std), std)

    model = NegotiationMimic()
    model.input_mean.copy_(mean)
    model.input_std.copy_(std)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ds = TensorDataset(X_tr, Y_act, Y_pr, M_pr)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    def focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                   class_w: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
        """Focal loss with class weighting. Down-weights easy examples
        (high predicted probability for true class) so training focuses
        on the hard/ambiguous boundary cases that drive miscalibration."""
        log_p = F.log_softmax(logits, dim=-1)
        log_p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = log_p_t.exp()
        w_t = class_w[targets]
        return -(w_t * (1.0 - p_t).pow(gamma) * log_p_t).mean()

    def kl_dist_match(logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
        """KL(target_dist || mean predicted dist). Forces the mimic's batch-mean
        action distribution to match the empirical LLM action distribution."""
        mean_pred = F.softmax(logits, dim=-1).mean(dim=0).clamp(min=1e-6)
        mean_pred = mean_pred / mean_pred.sum()
        return torch.sum(target_dist * (torch.log(target_dist) - torch.log(mean_pred)))

    KL_LAMBDA = 0.5  # auxiliary weight on distribution-matching term
    FOCAL_GAMMA = 2.0

    best_val_loss = math.inf
    best_state = None
    patience_left = args.patience
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yab, ypb, mpb in loader:
            logits, price_pred = model(xb)
            act_loss = focal_loss(logits, yab, weights, gamma=FOCAL_GAMMA)
            kl_loss = kl_dist_match(logits, target_action_dist)
            price_err = (price_pred - ypb) ** 2 * mpb
            denom = mpb.sum().clamp(min=1.0)
            price_loss = price_err.sum() / denom
            loss = act_loss + KL_LAMBDA * kl_loss + args.price_loss_weight * price_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            logits_v, price_v = model(X_val)
            val_act = focal_loss(logits_v, Y_a_val, weights, gamma=FOCAL_GAMMA).item()
            val_kl = kl_dist_match(logits_v, target_action_dist).item()
            denom_v = M_p_val.sum().clamp(min=1.0).item()
            val_price = (((price_v - Y_p_val) ** 2 * M_p_val).sum() / denom_v).item()
            val_total = val_act + KL_LAMBDA * val_kl + args.price_loss_weight * val_price

        if val_total < best_val_loss - 1e-5:
            best_val_loss = val_total
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1

        if epoch % args.log_every == 0 or epoch == 1:
            print(f"  [{alias}] epoch {epoch:3d}/{args.epochs}  "
                  f"train={loss.item():.4f}  val_focal={val_act:.4f}  val_kl={val_kl:.4f}  "
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
        "hidden": HIDDEN,
        "action_dim": ACTION_DIM,
        "alias": alias,
        "class_weights": weights.tolist(),
        "best_val_loss": best_val_loss,
        "n_train": len(train),
        "n_val": len(val),
        "action_class_counts": dict(class_counts),
    }, out_path)
    print(f"  [{alias}] saved -> {out_path}", flush=True)
    return {
        "alias": alias,
        "best_val_loss": float(best_val_loss),
        "n_train": len(train),
        "n_val": len(val),
        "action_class_counts": dict(class_counts),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--meta", default=DEFAULT_META)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
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
