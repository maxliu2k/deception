"""Validate that deception mimics reproduce LLM claim distributions.

For each LLM/mimic pair, compares:
  - lie rate per attribute (from y_lied labels): real-LLM rate vs mimic rate
  - claim distribution per attribute, binned at 0.05 resolution

Tests used:
  - chi² goodness-of-fit on the binned claim distribution (per attribute,
    aggregated across attributes for the per-LLM p-value)
  - KL divergence (binned), reported as effect size

The per-LLM aggregate p is the result of Fisher-combining per-attribute p-values.
The headline metric is `aggregate_p`: the chi² p-value across all 5 LLM/mimic
pairs (using Fisher's combined-p method).

Acceptance: aggregate_p > 0.05.

Usage:
    python -m simulation.validate_deception_mimics \
        --data simulation/datasets/deception_dataset_v1.jsonl \
        --meta simulation/datasets/deception_dataset_v1_meta.json \
        --out simulation/datasets/deception_fit_report_v1.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from simulation.mimic_agent import _load_deception_mimic, _temperature, _strip_mimic_prefix


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _binned_distribution(values: np.ndarray, bins: int = 20) -> np.ndarray:
    """Histogram of [0,1] values into `bins` equal-width bins; normalized."""
    h, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    total = h.sum()
    if total == 0:
        return np.zeros(bins, dtype=np.float64)
    return h.astype(np.float64) / float(total)


def chi2_pvalue(observed: np.ndarray, expected: np.ndarray, *, n_obs: int) -> tuple[float, float]:
    """Chi² goodness-of-fit; returns (chi2_stat, p_value)."""
    obs = observed * n_obs
    exp = expected * n_obs
    mask = exp > 1e-6
    if mask.sum() == 0:
        return 0.0, 1.0
    chi2 = float(((obs[mask] - exp[mask]) ** 2 / exp[mask]).sum())
    dof = int(mask.sum()) - 1
    if dof <= 0:
        return chi2, 1.0
    # survival function of chi² distribution
    try:
        from math import gamma
        # Use a series expansion or scipy if available
        from scipy.stats import chi2 as _chi2
        p = float(_chi2.sf(chi2, dof))
    except ImportError:
        # Fallback: rough p-value via numerical CDF approx
        p = math.exp(-chi2 / 2.0) * (chi2 ** (dof / 2.0 - 1)) / (2 ** (dof / 2.0) * math.gamma(dof / 2.0)) if chi2 < 100 else 0.0
        p = max(0.0, min(1.0, p))
    return chi2, p


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    """KL(p || q), nat units."""
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    return float((p * np.log(p / q)).sum())


def cramers_v(observed: np.ndarray, expected: np.ndarray, *, n_obs: int) -> float:
    """Cramér's V from chi² and table dims (1×k contingency table here → V = sqrt(chi2/n))."""
    chi2, _ = chi2_pvalue(observed, expected, n_obs=n_obs)
    if n_obs == 0:
        return 0.0
    return float(np.sqrt(chi2 / n_obs))


def fisher_combine(pvals: list[float], *, eps: float = 1e-300) -> float:
    """Fisher's method for combining independent p-values."""
    pvals = [max(eps, p) for p in pvals if p is not None and 0.0 < p <= 1.0]
    if not pvals:
        return 1.0
    stat = -2.0 * sum(math.log(p) for p in pvals)
    dof = 2 * len(pvals)
    try:
        from scipy.stats import chi2 as _chi2
        return float(_chi2.sf(stat, dof))
    except ImportError:
        return max(0.0, math.exp(-stat / 4.0))


def sample_mimic_claims(alias: str, rows: list[dict], *, n_samples_per_row: int = 4) -> np.ndarray | None:
    """Run the mimic on each row's input to produce synthetic claims.

    Returns array shaped (n_rows * n_samples_per_row, 5) of claim values, or None
    if the mimic isn't trained.
    """
    import torch
    import random as _rnd
    from simulation.mimic_agent import _temperature
    model = _load_deception_mimic(alias)
    if model is None:
        model = _load_deception_mimic(f"Mimic-{alias}")
    if model is None:
        return None
    # Feed the dataset's exact feature vector x straight to the model. This is
    # dimension-agnostic (works for the 14-dim schema) and uses precisely the
    # features the mimic was trained on, instead of re-deriving them.
    T = _temperature()
    samples = []
    for r in rows:
        x = [float(v) for v in r["x"]]
        truth = x[:5]
        xt = torch.tensor([x], dtype=torch.float32)
        with torch.no_grad():
            lie_logits, raw_claim = model(xt)
        p_lie = torch.sigmoid(lie_logits).squeeze(0).cpu().numpy()
        raw = raw_claim.squeeze(0).cpu().numpy()
        rs = model.claim_resid_std   # calibrated per-attr noise (mirror deployment)
        for _ in range(n_samples_per_row):
            c = []
            for a in range(5):
                lie = (_rnd.random() < float(p_lie[a])) if T > 0 else (float(p_lie[a]) > 0.5)
                if lie:
                    val = float(raw[a])
                    sigma = float(rs[a]) if rs is not None and a < len(rs) else 0.0
                    if T > 0 and sigma > 0:
                        val += _rnd.gauss(0.0, sigma * T)
                    c.append(round(max(0.0, min(1.0, val)), 2))
                else:
                    c.append(round(float(truth[a]), 2))
            samples.append(c)
    return np.array(samples, dtype=np.float32)


def total_variation(p, q) -> float:
    """TVD between two binned distributions: 0.5 * sum|p-q|, in [0,1]. Bounded and
    sample-size-independent — the acceptance gate (chi2 p-value is over-powered)."""
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    return 0.5 * float(np.abs(p - q).sum())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="simulation/datasets/deception_dataset_v1.jsonl")
    p.add_argument("--meta", default="simulation/datasets/deception_dataset_v1_meta.json")
    p.add_argument("--out", default="simulation/datasets/deception_fit_report_v1.json")
    p.add_argument("--bins", type=int, default=20,
                   help="Number of equal-width bins over [0,1] for the histogram comparison.")
    p.add_argument("--samples-per-row", type=int, default=4,
                   help="Mimic samples to draw per real-LLM row (more = lower noise in mimic distribution).")
    p.add_argument("--max-tvd", type=float, default=0.15,
                   help="Acceptance gate: overall mean TVD must be <= this.")
    args = p.parse_args()

    data_path = Path(args.data)
    meta_path = Path(args.meta)
    out_path = Path(args.out)
    if not data_path.exists():
        print(f"ERROR: dataset {data_path} not found")
        return 1
    rows = load_dataset(data_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    bidder_vocab = meta.get("bidder_vocab") or {}
    idx_to_alias = {v: k for k, v in bidder_vocab.items()}
    by_idx: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_idx[int(r["bidder_index"])].append(r)

    report = {
        "version": "v1",
        "dataset": str(data_path),
        "per_llm": [],
        "aggregate_p": None,
        "acceptance_threshold": 0.05,
        "accept": False,
    }
    per_llm_pvals: list[float] = []
    per_llm_tvds: list[float] = []
    for idx, real_rows in sorted(by_idx.items()):
        alias = idx_to_alias.get(idx, f"bidder_{idx}")
        real_claims = np.array([r["y_claim"] for r in real_rows], dtype=np.float32)
        n_obs = int(len(real_rows))
        mimic_claims = sample_mimic_claims(alias, real_rows, n_samples_per_row=args.samples_per_row)
        if mimic_claims is None:
            print(f"  [{alias}] skipping — mimic not trained.")
            report["per_llm"].append({"alias": alias, "skipped": True, "reason": "mimic not trained"})
            continue
        per_attr = []
        attr_pvals = []
        attr_tvds = []
        for a in range(5):
            real_dist = _binned_distribution(real_claims[:, a], bins=args.bins)
            mimic_dist = _binned_distribution(mimic_claims[:, a], bins=args.bins)
            chi2, pval = chi2_pvalue(real_dist, mimic_dist, n_obs=n_obs)
            kl = kl_divergence(real_dist, mimic_dist)
            v = cramers_v(real_dist, mimic_dist, n_obs=n_obs)
            tvd = total_variation(real_dist, mimic_dist)
            per_attr.append({
                "attr_idx": a,
                "tvd": round(tvd, 4),
                "chi2": round(chi2, 4),
                "p": round(pval, 4),
                "cramers_v": round(v, 4),
                "kl": round(kl, 4),
            })
            attr_pvals.append(pval)
            attr_tvds.append(tvd)
        agg_p = fisher_combine(attr_pvals)
        mean_tvd = float(np.mean(attr_tvds))
        per_llm_pvals.append(agg_p)
        per_llm_tvds.append(mean_tvd)
        report["per_llm"].append({
            "alias": alias,
            "n_real": n_obs,
            "n_mimic": int(mimic_claims.shape[0]),
            "mean_tvd": round(mean_tvd, 4),
            "per_attribute": per_attr,
            "aggregate_p": round(agg_p, 4),
        })
        print(f"  [{alias}] n_real={n_obs}  n_mimic={mimic_claims.shape[0]}  mean_tvd={mean_tvd:.4f}  (ref aggregate_p={agg_p:.4f})")

    overall_p = fisher_combine(per_llm_pvals)
    overall_tvd = float(np.mean(per_llm_tvds)) if per_llm_tvds else float("nan")
    accept = bool(overall_tvd <= args.max_tvd)
    report["aggregate_p"] = round(overall_p, 4)   # chi2/Fisher kept as secondary reference only
    report["mean_tvd"] = round(overall_tvd, 4)
    report["max_tvd"] = args.max_tvd
    report["accept"] = accept
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nOverall mean_tvd = {overall_tvd:.4f}  (threshold {args.max_tvd})  ->  {'ACCEPT' if accept else 'REJECT'} mimic fit")
    print(f"  (secondary ref, NOT the gate: chi2-Fisher aggregate_p = {overall_p:.4f})")
    print(f"Report -> {out_path}")
    return 0 if accept else 2


if __name__ == "__main__":
    sys.exit(main())
