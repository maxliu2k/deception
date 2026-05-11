"""
Train two independent neural networks per bidder (LLM):

  - AuctionClassifier  — predicts raise vs pass         (BCE loss)
  - AuctionRegressor   — predicts overbid above min legal (Huber, raise-only)

Each network has its own trunk, optimizer, and early-stopping curve.
Leave-one-episode-out CV reports honest accuracy / MAE before final training.
Final models are trained on all episodes except a held-out test episode and
saved as .pt checkpoints.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, log_loss


# ---------------------------------------------------------------------------
# Models — fully separate networks
# ---------------------------------------------------------------------------

class AuctionClassifier(nn.Module):
    def __init__(self, input_dim: int = 32, hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        # Non-trainable z-score buffers; populated by .set_scaler before training.
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def set_scaler(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.input_mean.copy_(mean)
        self.input_std.copy_(std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.input_mean) / self.input_std
        return self.net(x).squeeze(-1)


class AuctionRegressor(nn.Module):
    def __init__(self, input_dim: int = 32, hidden: int = 32, dropout: float = 0.2):
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
            nn.Linear(hidden, 1),
        )
        self.softplus = nn.Softplus()

    def set_scaler(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.input_mean.copy_(mean)
        self.input_std.copy_(std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.input_mean) / self.input_std
        return self.softplus(self.trunk(x)).squeeze(-1)


def fit_scaler(X: np.ndarray, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.from_numpy(X.mean(axis=0).astype(np.float32))
    std = torch.from_numpy(X.std(axis=0).astype(np.float32))
    std = torch.clamp(std, min=eps)  # guard against zero-variance features
    return mean, std


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def fit_temperature(
    model: "AuctionClassifier",
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    *,
    min_T: float = 0.05,
    max_iter: int = 200,
) -> float:
    """Fit a single scalar T (Platt-style temperature) so sigmoid(logit/T)
    minimises validation BCE. Standard post-hoc calibration trick — preserves
    argmax accuracy but tightens log_loss and sampling fidelity."""
    if X_val is None or len(X_val) == 0:
        return 1.0
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_val).to(device)).detach()
    targets = torch.from_numpy(y_val).to(device)
    T = torch.nn.Parameter(torch.tensor(1.0, device=device))
    optimizer = torch.optim.LBFGS([T], lr=0.1, max_iter=max_iter)
    loss_fn = nn.BCEWithLogitsLoss()

    def _closure():
        optimizer.zero_grad()
        scaled = logits / T.clamp(min=min_T)
        loss = loss_fn(scaled, targets)
        loss.backward()
        return loss

    try:
        optimizer.step(_closure)
    except Exception:
        return 1.0
    return float(T.detach().clamp(min=min_T).cpu().item())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_meta(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rows_to_arrays(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.array([r["x"] for r in rows], dtype=np.float32)
    y_action = np.array([r["y_action"] for r in rows], dtype=np.float32)
    y_delta_steps = np.array([r.get("y_delta_steps", 0.0) for r in rows], dtype=np.float32)
    y_mask = np.array([r["y_delta_mask"] for r in rows], dtype=np.float32)
    step_dollars = np.array([r.get("step_dollars", 50.0) for r in rows], dtype=np.float32)
    return X, y_action, y_delta_steps, y_mask, step_dollars


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_classifier(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[AuctionClassifier, dict[str, float]]:
    model = AuctionClassifier(input_dim=X_tr.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    mean, std = fit_scaler(X_tr)
    model.set_scaler(mean.to(device), std.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = _make_loader(X_tr, y_tr, args.batch_size, shuffle=True)

    best_val = float("inf")
    best_state: dict | None = None
    no_improve = 0

    for _ in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            logit = model(xb)
            loss = loss_fn(logit, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if X_val is not None and len(X_val):
            model.eval()
            with torch.no_grad():
                logit = model(torch.from_numpy(X_val).to(device))
                val_loss = loss_fn(logit, torch.from_numpy(y_val).to(device)).item()
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_loss": best_val}


def train_regressor(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[AuctionRegressor, dict[str, float]]:
    model = AuctionRegressor(input_dim=X_tr.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    mean, std = fit_scaler(X_tr)
    model.set_scaler(mean.to(device), std.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.HuberLoss(delta=args.huber_delta)

    train_loader = _make_loader(X_tr, y_tr, args.batch_size, shuffle=True)

    best_val = float("inf")
    best_state: dict | None = None
    no_improve = 0

    for _ in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if X_val is not None and len(X_val):
            model.eval()
            with torch.no_grad():
                pred = model(torch.from_numpy(X_val).to(device))
                val_loss = loss_fn(pred, torch.from_numpy(y_val).to(device)).item()
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_loss": best_val}


# ---------------------------------------------------------------------------
# Leave-one-episode-out CV
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_clf(model: AuctionClassifier, X: np.ndarray, y: np.ndarray, device: torch.device, *, temperature: float = 1.0) -> tuple[float, float]:
    model.eval()
    logit = model(torch.from_numpy(X).to(device)).cpu().numpy()
    if temperature != 1.0 and temperature > 0:
        logit = logit / float(temperature)
    proba = 1.0 / (1.0 + np.exp(-logit))
    proba = np.clip(proba, 1e-7, 1 - 1e-7)
    preds = (proba >= 0.5).astype(np.float32)
    # `labels=[0,1]` so log_loss tolerates folds where the held-out test
    # episode contains only one class (e.g. all-PASS).
    return float(accuracy_score(y, preds)), float(log_loss(y, proba, labels=[0.0, 1.0]))


@torch.no_grad()
def _eval_reg(
    model: AuctionRegressor,
    X: np.ndarray,
    y_steps: np.ndarray,
    device: torch.device,
) -> float:
    """MAE in legal-bid-step units (the regressor's actual training target)."""
    model.eval()
    pred_steps = np.round(model(torch.from_numpy(X).to(device)).cpu().numpy()).clip(min=0.0)
    return float(np.abs(pred_steps - y_steps).mean())


def loo_cv(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    episodes = sorted({r["episode_index"] for r in rows})
    if len(episodes) < 3:
        return {}

    rng = random.Random(args.seed)

    accs, losses, maes_steps = [], [], []

    for held_out in episodes:
        train_rows = [r for r in rows if r["episode_index"] != held_out]
        test_rows = [r for r in rows if r["episode_index"] == held_out]
        if not train_rows or not test_rows:
            continue

        # carve a small val split from train for early stopping
        rng.shuffle(train_rows)
        n_val = max(1, len(train_rows) // 10)
        val_rows = train_rows[:n_val]
        sub_train = train_rows[n_val:]

        X_tr, y_act_tr, y_steps_tr, y_mask_tr, _ = rows_to_arrays(sub_train)
        X_val, y_act_val, y_steps_val, y_mask_val, _ = rows_to_arrays(val_rows)
        X_te, y_act_te, y_steps_te, y_mask_te, _ = rows_to_arrays(test_rows)

        clf, _ = train_classifier(X_tr, y_act_tr, X_val, y_act_val, args, device)
        acc, ll = _eval_clf(clf, X_te, y_act_te, device)
        accs.append(acc); losses.append(ll)

        m_tr = y_mask_tr.astype(bool)
        m_val = y_mask_val.astype(bool)
        m_te = y_mask_te.astype(bool)
        if m_tr.sum() >= 5 and m_te.sum() >= 1:
            y_d_tr = y_steps_tr[m_tr]
            y_d_val = y_steps_val[m_val] if m_val.sum() >= 1 else None
            X_d_val = X_val[m_val] if m_val.sum() >= 1 else None
            reg, _ = train_regressor(X_tr[m_tr], y_d_tr, X_d_val, y_d_val, args, device)
            maes_steps.append(_eval_reg(reg, X_te[m_te], y_steps_te[m_te], device))

    return {
        "cv_acc_mean": float(np.mean(accs)),
        "cv_acc_std": float(np.std(accs)),
        "cv_log_loss": float(np.mean(losses)),
        "cv_mae_steps": float(np.mean(maes_steps)) if maes_steps else float("nan"),
        "cv_folds": len(accs),
    }


# ---------------------------------------------------------------------------
# Per-bidder pipeline
# ---------------------------------------------------------------------------

def _safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-.]", "_", name)


def train_one(
    bidder_name: str,
    bidder_rows: list[dict[str, Any]],
    test_ep: int,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    print(f"\n[{bidder_name}]  total={len(bidder_rows)}")

    cv = loo_cv(bidder_rows, args, device)
    if cv:
        print(
            f"  LOO-CV ({cv['cv_folds']} folds)  "
            f"acc={cv['cv_acc_mean']:.3f} +/- {cv['cv_acc_std']:.3f}  "
            f"log_loss={cv['cv_log_loss']:.4f}  "
            f"mae[steps]={cv['cv_mae_steps']:.3f}"
        )

    # Final model: train on everything except test episode
    train_rows = [r for r in bidder_rows if r["episode_index"] != test_ep]
    test_rows = [r for r in bidder_rows if r["episode_index"] == test_ep]
    if not train_rows:
        print(f"  Skipping {bidder_name}: no training rows.")
        return {}

    rng = random.Random(args.seed)
    pool = list(train_rows)
    rng.shuffle(pool)
    n_val = max(1, len(pool) // 10)
    val_rows = pool[:n_val]
    sub_train = pool[n_val:]

    X_tr, y_act_tr, y_steps_tr, y_mask_tr, _ = rows_to_arrays(sub_train)
    X_val, y_act_val, y_steps_val, y_mask_val, _ = rows_to_arrays(val_rows)

    clf, _ = train_classifier(X_tr, y_act_tr, X_val, y_act_val, args, device)
    # Post-hoc temperature scaling on val. Optional because tiny val sets
    # (typical here: ~30 rows for Pro) produce unreliable T fits — the chi²
    # test showed calibration *hurt* mimic-vs-LLM distributional fidelity at
    # this data scale. Off by default; enable with --calibrate.
    if getattr(args, "calibrate", False):
        calibration_T = fit_temperature(clf, X_val, y_act_val, device)
    else:
        calibration_T = 1.0

    reg = None
    m_tr = y_mask_tr.astype(bool)
    m_val = y_mask_val.astype(bool)
    if m_tr.sum() >= 5:
        y_d_tr = y_steps_tr[m_tr]
        if m_val.sum() >= 1:
            y_d_val = y_steps_val[m_val]
            X_d_val = X_val[m_val]
        else:
            y_d_val, X_d_val = None, None
        reg, _ = train_regressor(X_tr[m_tr], y_d_tr, X_d_val, y_d_val, args, device)

    test_metrics: dict[str, float] = {}
    if test_rows:
        X_te, y_act_te, y_steps_te, y_mask_te, _ = rows_to_arrays(test_rows)
        acc, ll_raw = _eval_clf(clf, X_te, y_act_te, device, temperature=1.0)
        _,   ll_cal = _eval_clf(clf, X_te, y_act_te, device, temperature=calibration_T)
        test_metrics["test_acc"] = acc
        test_metrics["test_log_loss"] = ll_raw
        test_metrics["test_log_loss_calibrated"] = ll_cal
        m_te = y_mask_te.astype(bool)
        if reg is not None and m_te.sum() >= 1:
            test_metrics["test_mae_steps"] = _eval_reg(reg, X_te[m_te], y_steps_te[m_te], device)
        msg = (
            f"  Test (ep {test_ep})  acc={acc:.3f}  log_loss={ll_raw:.4f}->{ll_cal:.4f}  "
            f"T={calibration_T:.3f}"
        )
        if "test_mae_steps" in test_metrics:
            msg += f"  mae[steps]={test_metrics['test_mae_steps']:.3f}"
        print(msg)

    safe = _safe_filename(bidder_name)
    clf_path = out_dir / f"auction_clf_v6_{safe}.pt"
    torch.save({
        "kind": "classifier",
        "model_state": clf.state_dict(),
        "input_dim": X_tr.shape[1],
        "hidden": args.hidden,
        "dropout": args.dropout,
        "bidder_name": bidder_name,
        "calibration_temperature": calibration_T,
    }, clf_path)
    print(f"  Saved clf -> {clf_path.name}  (calibration T={calibration_T:.3f})")

    if reg is not None:
        reg_path = out_dir / f"auction_reg_v6_{safe}.pt"
        torch.save({
            "kind": "regressor",
            "target": "y_delta_steps",
            "model_state": reg.state_dict(),
            "input_dim": X_tr.shape[1],
            "hidden": args.hidden,
            "dropout": args.dropout,
            "bidder_name": bidder_name,
        }, reg_path)
        print(f"  Saved reg -> {reg_path.name}")

    return {**cv, **test_metrics}


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    rows = load_dataset(Path(args.data))
    meta = load_meta(Path(args.meta))
    print(f"Loaded {len(rows)} rows")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean stale .joblib files so the mimic agent can't accidentally load them
    for stale in out_dir.glob("auction_*_v6_*.joblib"):
        stale.unlink()

    index_to_name: dict[int, str] = {v: k for k, v in (meta.get("bidder_vocab") or {}).items()}

    # Drop any rows from a Mimic-* bidder — those are model-replays of LLMs we
    # already train against, and including them creates a self-feedback loop.
    rows = [r for r in rows if not str(index_to_name.get(r["bidder_index"], "")).startswith("Mimic-")]
    print(f"After filtering Mimic-* bidders: {len(rows)} rows")

    by_bidder: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        name = index_to_name.get(row["bidder_index"], f"bidder_{row['bidder_index']}")
        by_bidder.setdefault(name, []).append(row)

    # Test episode must be one where every kept bidder participated.
    rng = random.Random(args.seed)
    eps_per_bidder = [set(r["episode_index"] for r in rs) for rs in by_bidder.values()]
    candidate_eps = sorted(set.intersection(*eps_per_bidder)) if eps_per_bidder else []
    if not candidate_eps:
        candidate_eps = sorted({r["episode_index"] for r in rows})
    test_ep = rng.choice(candidate_eps)
    print(f"Test episode: {test_ep}  |  Candidates: {candidate_eps}")

    print(f"Bidders: {sorted(by_bidder)}")

    results: dict[str, dict] = {}
    for name in sorted(by_bidder):
        results[name] = train_one(name, by_bidder[name], test_ep, args, device, out_dir)

    print("\n=== Summary ===")
    for name, m in sorted(results.items()):
        if not m:
            continue
        cv_acc = m.get("cv_acc_mean", float("nan"))
        cv_std = m.get("cv_acc_std", float("nan"))
        te_acc = m.get("test_acc", float("nan"))
        print(f"  {name:20s}  CV acc={cv_acc:.3f}+/-{cv_std:.3f}  test_acc={te_acc:.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train classifier+regressor neural nets per bidder.")
    p.add_argument("--data", default="simulation/datasets/auction_nn_dataset_v6_nn.jsonl")
    p.add_argument("--meta", default="simulation/datasets/auction_nn_dataset_v6_nn_meta.json")
    p.add_argument("--output-dir", default="simulation/models/v6")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--calibrate", action="store_true",
                   help="Fit post-hoc Platt-style temperature scaling on val. Off by default.")
    p.add_argument("--huber-delta", type=float, default=1.0,
                   help="Huber transition; target unit is # of legal increments.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
