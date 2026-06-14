"""Train per-LLM deception mimic networks.

Architecture (leakage-safe v3):
  Input:  23-dim [observed_truth(5), visible_mask(5), preferences(5),
                  own_trust, opp1..opp4_trust, round_fraction, threshold, penalty]
  Output: 5-dim lie classifier (sigmoid)  +  5-dim claim regressor (per-attribute)

Loss:
  BCE per attribute on the lie head, masked to VISIBLE attributes only (on a
    hidden attribute a claim != truth is a guess, not a deliberate lie, so the
    label is ill-defined and excluded).
  Masked MSE on the regressor head, on the attributes that are actually emitted
    by the regressor at inference time: (visible AND lied) OR hidden. On a
    visible attribute that was not lied on, inference snaps to the known truth,
    so the regressor is not trained there.

Validation:
  Leave-one-episode-out cross-validation gives the honest decision-level
  accuracy (lie-head accuracy + regressor MAE). The deployed model is then
  trained on all episodes except one held-out test episode.

One model per LLM alias. Each model is saved as a single `.pt` file containing
both heads' state dicts plus the input z-score buffers.

Usage:
    python -m simulation.train_deception_nn \
        --data simulation/datasets/deception_dataset_v3.jsonl \
        --meta simulation/datasets/deception_dataset_v3_meta.json \
        --out-dir simulation/models/deception_v2
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
    """Two-head deception mimic (small feed-forward MLP).

    A shared 2-layer trunk processes the 23-dim leakage-safe input; two
    independent linear heads emit:
      - lie_logits: 5 floats (sigmoid → P(lie on attr a)); trained with BCE
        masked to VISIBLE attributes.
      - claim_raw:  5 floats (sigmoid → claim value in [0, 1]); trained with MSE
        masked to the emitted attributes ((visible AND lied) OR hidden).
    """

    def __init__(self, input_dim: int = 23, hidden: int = 32, dropout: float = 0.2, num_attrs: int = 5):
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


def rows_to_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.array([r["x"] for r in rows], dtype=np.float32)
    y_lied = np.array([r["y_lied"] for r in rows], dtype=np.float32)
    y_claim = np.array([r["y_claim"] for r in rows], dtype=np.float32)
    # visible_mask defaults to all-visible for legacy rows that predate the field.
    vis = np.array(
        [r.get("visible_mask", [1, 1, 1, 1, 1]) for r in rows], dtype=np.float32
    )
    return X, y_lied, y_claim, vis


def _regressor_mask(y_lied: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """Attributes the regressor is responsible for at inference time.

    (visible AND lied) OR hidden. On a visible, non-lied attribute the mimic
    snaps to the known truth, so the regressor is not supervised there.
    """
    visible_lied = vis * y_lied
    hidden = 1.0 - vis
    return np.clip(visible_lied + hidden, 0.0, 1.0)


def _episode_indices(rows: list[dict]) -> list[int]:
    return sorted({int(r.get("episode_index", 0)) for r in rows})


# ── Training ────────────────────────────────────────────────────────────────

def _masked_losses(
    lie_logits: torch.Tensor,
    claim_pred: torch.Tensor,
    yl: torch.Tensor,
    yc: torch.Tensor,
    vis: torch.Tensor,
    bce_none: nn.Module,
    mse_none: nn.Module,
    claim_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (total, lie, claim). Lie head masked to visible attrs; regressor
    masked to (visible AND lied) OR hidden."""
    lie_mask = vis
    lie_per = bce_none(lie_logits, yl) * lie_mask
    lie_den = lie_mask.sum()
    loss_lie = lie_per.sum() / lie_den if float(lie_den) > 0 else lie_per.sum() * 0.0

    reg_mask = torch.clamp(vis * yl + (1.0 - vis), 0.0, 1.0)
    reg_per = mse_none(claim_pred, yc) * reg_mask
    reg_den = reg_mask.sum()
    loss_claim = reg_per.sum() / reg_den if float(reg_den) > 0 else reg_per.sum() * 0.0

    total = loss_lie + claim_loss_weight * loss_claim
    return total, loss_lie, loss_claim


def _fit(
    train_rows: list[dict],
    val_rows: list[dict],
    *,
    args: argparse.Namespace,
    device: torch.device,
    bidder_name: str,
    verbose: bool = True,
) -> tuple[DeceptionMimic, dict]:
    """Train a model on train_rows, early-stopping on val_rows. Returns the best
    model and a metrics dict."""
    X_tr, y_lied_tr, y_claim_tr, vis_tr = rows_to_arrays(train_rows)
    X_val, y_lied_val, y_claim_val, vis_val = rows_to_arrays(val_rows)

    mean, std = fit_scaler(X_tr)
    model = DeceptionMimic(input_dim=X_tr.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    model.set_scaler(mean.to(device), std.to(device))

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce_none = nn.BCEWithLogitsLoss(reduction="none")
    mse_none = nn.MSELoss(reduction="none")

    tr_ds = TensorDataset(
        torch.from_numpy(X_tr),
        torch.from_numpy(y_lied_tr),
        torch.from_numpy(y_claim_tr),
        torch.from_numpy(vis_tr),
    )
    loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    val_pack = (
        torch.from_numpy(X_val).to(device),
        torch.from_numpy(y_lied_val).to(device),
        torch.from_numpy(y_claim_val).to(device),
        torch.from_numpy(vis_val).to(device),
    )

    best_val = math.inf
    best_state = None
    epochs_no_improve = 0
    for epoch in range(args.epochs):
        model.train()
        for xb, yl, yc, vb in loader:
            xb, yl, yc, vb = xb.to(device), yl.to(device), yc.to(device), vb.to(device)
            lie_logits, claim_pred = model(xb)
            loss, _, _ = _masked_losses(lie_logits, claim_pred, yl, yc, vb, bce_none, mse_none, args.claim_loss_weight)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        model.eval()
        with torch.no_grad():
            xb, yl, yc, vb = val_pack
            lie_logits, claim_pred = model(xb)
            val_total, val_lie, val_claim = _masked_losses(
                lie_logits, claim_pred, yl, yc, vb, bce_none, mse_none, args.claim_loss_weight
            )
            val_total = float(val_total)

        if val_total + 1e-5 < best_val:
            best_val = val_total
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if verbose and ((epoch + 1) % args.log_every == 0 or epoch == 0):
            print(f"  [{bidder_name}] epoch {epoch + 1:3d}/{args.epochs}  "
                  f"val_lie={float(val_lie):.4f}  val_claim_mse={float(val_claim):.4f}  "
                  f"val_total={val_total:.4f}  best={best_val:.4f}  patience={epochs_no_improve}/{args.patience}")
        if epochs_no_improve >= args.patience:
            if verbose:
                print(f"  [{bidder_name}] early stop at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_loss": float(best_val), "n_train": len(train_rows), "n_val": len(val_rows)}


def _evaluate(model: DeceptionMimic, rows: list[dict], device: torch.device) -> dict:
    """Decision-level metrics on held-out rows: lie-head accuracy (visible attrs)
    and regressor MAE (on attributes the regressor emits at inference)."""
    if not rows:
        return {"lie_acc": float("nan"), "claim_mae": float("nan"), "n_rows": 0}
    X, y_lied, y_claim, vis = rows_to_arrays(rows)
    reg_mask = _regressor_mask(y_lied, vis)
    model.eval()
    with torch.no_grad():
        lie_logits, claim_pred = model(torch.from_numpy(X).to(device))
        p_lie = torch.sigmoid(lie_logits).cpu().numpy()
        claim_pred = claim_pred.cpu().numpy()
    pred_lie = (p_lie > 0.5).astype(np.float32)
    vis_sum = float(vis.sum())
    lie_acc = float(((pred_lie == y_lied) * vis).sum() / vis_sum) if vis_sum > 0 else float("nan")
    reg_sum = float(reg_mask.sum())
    claim_mae = float((np.abs(claim_pred - y_claim) * reg_mask).sum() / reg_sum) if reg_sum > 0 else float("nan")
    return {"lie_acc": lie_acc, "claim_mae": claim_mae, "n_rows": len(rows)}


def _claim_residual_std(model: DeceptionMimic, rows: list[dict], device: torch.device) -> list[float]:
    """Per-attribute std of the regressor residual (pred − y_claim), over the
    attributes the regressor is responsible for at inference ((visible AND lied)
    OR hidden).

    The regressor outputs a conditional point estimate (≈ the mean claim), so on
    its own it under-disperses: it nails central tendency (low MAE) but produces a
    claim distribution far narrower than the real LLM's. This residual std is the
    spread it leaves unexplained; the mimic samples Gaussian noise with this std at
    inference so the synthetic claim distribution matches the real one's width."""
    if not rows:
        return [0.0] * 5
    X, y_lied, y_claim, vis = rows_to_arrays(rows)
    reg_mask = _regressor_mask(y_lied, vis)
    model.eval()
    with torch.no_grad():
        _, claim_pred = model(torch.from_numpy(X).to(device))
        claim_pred = claim_pred.cpu().numpy()
    resid = claim_pred - y_claim  # (n, 5)
    stds: list[float] = []
    for a in range(claim_pred.shape[1]):
        m = reg_mask[:, a] > 0.5
        stds.append(float(np.std(resid[m, a])) if int(m.sum()) >= 2 else 0.0)
    return stds


def loo_episode_cv(bidder_rows: list[dict], *, args: argparse.Namespace, device: torch.device, bidder_name: str) -> dict:
    """Leave-one-episode-out CV: for each episode, train on the rest and evaluate
    on the held-out episode. Returns averaged decision-level metrics."""
    episodes = _episode_indices(bidder_rows)
    if len(episodes) < 2:
        return {"folds": 0, "lie_acc": float("nan"), "claim_mae": float("nan"),
                "note": "fewer than 2 episodes; CV skipped"}
    cv_folds = getattr(args, "cv_folds", -1)
    if cv_folds == 0:
        return {"folds": 0, "lie_acc": float("nan"), "claim_mae": float("nan"),
                "note": "CV skipped (--cv-folds 0)"}
    if cv_folds > 0 and len(episodes) > cv_folds:
        # Deterministic stride subsample so the folds still span the full range.
        step = len(episodes) / float(cv_folds)
        episodes = [episodes[int(i * step)] for i in range(cv_folds)]
    fold_metrics: list[dict] = []
    for held in episodes:
        test_rows = [r for r in bidder_rows if int(r.get("episode_index", 0)) == held]
        train_rows = [r for r in bidder_rows if int(r.get("episode_index", 0)) != held]
        if not train_rows or not test_rows:
            continue
        rng = random.Random(args.seed + held)
        rng.shuffle(train_rows)
        n_val = max(1, len(train_rows) // 10)
        model, _ = _fit(train_rows[n_val:], train_rows[:n_val], args=args, device=device,
                        bidder_name=f"{bidder_name}|loo{held}", verbose=False)
        fold_metrics.append(_evaluate(model, test_rows, device))
    if not fold_metrics:
        return {"folds": 0, "lie_acc": float("nan"), "claim_mae": float("nan")}
    accs = [m["lie_acc"] for m in fold_metrics if not math.isnan(m["lie_acc"])]
    maes = [m["claim_mae"] for m in fold_metrics if not math.isnan(m["claim_mae"])]
    return {
        "folds": len(fold_metrics),
        "lie_acc": float(sum(accs) / len(accs)) if accs else float("nan"),
        "claim_mae": float(sum(maes) / len(maes)) if maes else float("nan"),
        "per_fold": fold_metrics,
    }


def train_one(
    bidder_rows: list[dict],
    *,
    args: argparse.Namespace,
    device: torch.device,
    bidder_name: str,
    out_path: Path,
) -> dict:
    # 1. Honest accuracy estimate via leave-one-episode-out CV.
    cv = loo_episode_cv(bidder_rows, args=args, device=device, bidder_name=bidder_name)
    if cv.get("folds"):
        print(f"  [{bidder_name}] LOO-CV ({cv['folds']} folds): "
              f"lie_acc={cv['lie_acc']:.4f}  claim_mae={cv['claim_mae']:.4f}")
    else:
        print(f"  [{bidder_name}] LOO-CV skipped ({cv.get('note', 'n/a')})")

    # 2. Final deployed model: hold out one test episode, train on the rest.
    episodes = _episode_indices(bidder_rows)
    rng = random.Random(args.seed)
    if len(episodes) >= 2:
        test_ep = episodes[-1]
        test_rows = [r for r in bidder_rows if int(r.get("episode_index", 0)) == test_ep]
        pool = [r for r in bidder_rows if int(r.get("episode_index", 0)) != test_ep]
    else:
        test_ep = None
        test_rows = []
        pool = list(bidder_rows)
    rng.shuffle(pool)
    n_val = max(1, len(pool) // 10)
    val_rows = pool[:n_val]
    train_rows = pool[n_val:]

    model, fit_info = _fit(train_rows, val_rows, args=args, device=device, bidder_name=bidder_name)
    test_metrics = _evaluate(model, test_rows, device) if test_rows else {}

    # Calibrated output spread: std of the regressor residual on the deployed
    # model, used to add Gaussian noise at inference so the synthetic claim
    # distribution matches the real LLM's width (the point estimate under-disperses).
    claim_resid_std = _claim_residual_std(model, bidder_rows, device)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    X_probe = np.array([bidder_rows[0]["x"]], dtype=np.float32)
    torch.save({
        "alias": bidder_name,
        "input_dim": int(X_probe.shape[1]),
        "hidden": args.hidden,
        "num_attrs": 5,
        "state_dict": model.state_dict(),
        "claim_resid_std": [round(s, 4) for s in claim_resid_std],
    }, out_path)
    print(f"  [{bidder_name}] saved -> {out_path}  "
          f"resid_std={[round(s, 3) for s in claim_resid_std]}")
    summary = {
        "alias": bidder_name,
        "n_rows": len(bidder_rows),
        "n_episodes": len(episodes),
        "n_train": fit_info["n_train"],
        "n_val": fit_info["n_val"],
        "best_val_loss": fit_info["best_val_loss"],
        "cv": {k: v for k, v in cv.items() if k != "per_fold"},
        "test_episode": test_ep,
        "test_metrics": test_metrics,
        "claim_resid_std": [round(s, 4) for s in claim_resid_std],
    }
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="simulation/datasets/deception_dataset_v3.jsonl")
    p.add_argument("--meta", default="simulation/datasets/deception_dataset_v3_meta.json")
    p.add_argument("--out-dir", default="simulation/models/deception_v2")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--claim-loss-weight", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--cv-folds", type=int, default=-1,
                   help="LOO-CV episodes: -1 = all (full leave-one-episode-out), "
                        "0 = skip CV entirely (fast; CV is diagnostic-only), "
                        "N>0 = subsample N episodes for a faster diagnostic.")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only-alias", default="",
                   help="If set, train only this alias (e.g. 'GPT-5.4').")
    p.add_argument("--threads", type=int, default=1,
                   help="torch CPU thread cap. For this ~2k-param net, 1 is fastest "
                        "(multithreading overhead dominates the math); 0 = leave default.")
    args = p.parse_args()

    if args.threads and args.threads > 0:
        torch.set_num_threads(args.threads)

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
