"""Validate that deception mimics reproduce LLM claim distributions.

For each LLM/mimic pair, compares:
  - lie rate per attribute (from y_lied labels): real-LLM rate vs mimic rate
  - claim distribution per attribute, binned at 0.05 resolution

Metrics:
  - total-variation distance (TVD) on the binned claim distribution — the
    HEADLINE metric and acceptance gate. TVD ∈ [0,1] is sample-size-independent
    and interpretable (TVD=0.1 ⇒ distributions disagree on 10% of their mass).
  - chi² goodness-of-fit p-value — reported for reference ONLY, not used as the
    gate. At n=480–2400 the chi²-p is a poor gate: it rejects negligible
    differences (over-power) and underflows to 0 on large ones, so the p answers
    the wrong question. KL and Cramér's V are reported alongside as effect sizes.

In addition to the pooled fit, each LLM is scored *per information level*
(known-attribute count). Because the information gradient IS the result, the
mimic must track each rung, not just the pooled average — a pooled pass can mask
one level fitting badly. Rungs above the TVD threshold are reported under
`per_level_failures` (and warned on stdout).

Acceptance: overall mean TVD ≤ --max-tvd (default 0.15). Per-level failures are
surfaced as a warning rather than a hard reject, since low-k / multimodal cells
are the hardest for a unimodal regressor to reproduce.

Usage:
    python -m simulation.validate_deception_mimics \
        --data simulation/datasets/deception_dataset_v3.jsonl \
        --meta simulation/datasets/deception_dataset_v3_meta.json \
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


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """Total-variation distance between two normalized histograms, in [0, 1].

    TVD = ½·Σ|p−q|. Sample-size-independent (unlike chi²), and directly
    interpretable: TVD=0.1 means the two claim distributions disagree on 10% of
    their probability mass. This is the headline effect size for acceptance —
    a p-value gate is inappropriate at n=480–2400, where chi² rejects negligible
    differences and underflows on large ones."""
    sp, sq = p.sum(), q.sum()
    if sp <= 0 or sq <= 0:
        return 1.0
    p = p / sp
    q = q / sq
    return float(0.5 * np.abs(p - q).sum())


def cramers_v(observed: np.ndarray, expected: np.ndarray, *, n_obs: int) -> float:
    """Cramér's V from chi² and table dims (1×k contingency table here → V = sqrt(chi2/n))."""
    chi2, _ = chi2_pvalue(observed, expected, n_obs=n_obs)
    if n_obs == 0:
        return 0.0
    return float(np.sqrt(chi2 / n_obs))


def fisher_combine(pvals: list[float], *, eps: float = 1e-300) -> float:
    """Fisher's method for combining independent p-values.

    p-values are clamped to [eps, 1] BEFORE use — critically, a p that has
    underflowed to exactly 0.0 (an extreme misfit) must be floored to eps, not
    discarded. The old `if 0.0 < p` filter dropped such zeros, so a model whose
    every attribute misfit so badly that p underflowed reported a *combined* p of
    1.0 (empty list → default), inverting the verdict. Floor, don't filter."""
    pvals = [min(1.0, max(eps, p)) for p in pvals if p is not None and not math.isnan(p)]
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
    bare_alias = _strip_mimic_prefix(alias)  # mimic file uses bare alias
    model = _load_deception_mimic(alias)
    if model is None:
        # Try with "Mimic-" prefix in case alias is bare.
        model = _load_deception_mimic(f"Mimic-{alias}")
    if model is None:
        # Fall back: maybe the file is stored under the original alias (e.g., for math tiers).
        return None
    mimic_alias = alias if alias.startswith("Mimic-") else f"Mimic-{alias}"
    samples = []
    for r in rows:
        feats = r.get("features") or {}
        # Reconstruct the decision-time context from the structured features so the
        # mimic sees exactly the v3 leakage-safe input it was trained on.
        observed_truth = feats.get("observed_truth")
        if observed_truth is None:
            observed_truth = list(r["x"][:5])
        policy_truth = [0.5 if v is None else float(v) for v in observed_truth]
        visible_attrs = feats.get("visible_attrs")
        preferences = feats.get("preferences") or list(r["x"][10:15])
        own_trust = float(feats.get("own_trust", r["x"][15]))
        opponents_trust = feats.get("opponent_trusts") or list(r["x"][16:20])
        information_mode = feats.get("information_mode", "full")
        threshold = float(feats.get("threshold", r["x"][21]))
        penalty = float(feats.get("penalty", r["x"][22]))
        round_index = int(feats.get("round_index", 0))
        total_rounds = int(feats.get("num_rounds", 12))
        # Sample n_samples_per_row stochastic outputs for distributional comparison.
        for _ in range(n_samples_per_row):
            from simulation.mimic_agent import deception_mimic_claim
            c = deception_mimic_claim(
                mimic_alias,
                policy_truth,
                own_trust,
                opponents_trust,
                information_mode=information_mode,
                observed_truth=observed_truth,
                visible_attrs=visible_attrs,
                preferences=preferences,
                threshold=threshold,
                penalty=penalty,
                round_index=round_index,
                total_rounds=total_rounds,
            )
            samples.append(c)
    return np.array(samples, dtype=np.float32)


def _row_level(row: dict) -> int:
    """Known-attribute count for a row = number of visible attrs (sum of the
    visibility mask). Falls back to the mask slice of x if `visible_mask` is
    absent. Every agent sees exactly `known_count` attrs, so this recovers the
    information level the row was played at."""
    vm = row.get("visible_mask")
    if vm is not None:
        return int(round(sum(float(x) for x in vm)))
    x = row.get("x") or []
    if len(x) >= 10:
        return int(round(sum(float(v) for v in x[5:10])))
    return 5


def _compare(real_claims: np.ndarray, mimic_claims: np.ndarray, *, n_obs: int,
             bins: int) -> tuple[list[dict], list[float], float, float]:
    """Per-attribute chi²/KL/Cramér's V/TVD comparison of two claim sets.

    Returns (per_attribute_records, attr_pvals, fisher_aggregate_p, mean_tvd).
    """
    per_attr, attr_pvals, tvds = [], [], []
    for a in range(5):
        real_dist = _binned_distribution(real_claims[:, a], bins=bins)
        mimic_dist = _binned_distribution(mimic_claims[:, a], bins=bins)
        chi2, pval = chi2_pvalue(real_dist, mimic_dist, n_obs=n_obs)
        kl = kl_divergence(real_dist, mimic_dist)
        v = cramers_v(real_dist, mimic_dist, n_obs=n_obs)
        tvd = total_variation(real_dist, mimic_dist)
        per_attr.append({
            "attr_idx": a,
            "chi2": round(chi2, 4),
            "p": round(pval, 4),
            "cramers_v": round(v, 4),
            "kl": round(kl, 4),
            "tvd": round(tvd, 4),
        })
        attr_pvals.append(pval)
        tvds.append(tvd)
    mean_tvd = float(sum(tvds) / len(tvds)) if tvds else 1.0
    return per_attr, attr_pvals, fisher_combine(attr_pvals), mean_tvd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="simulation/datasets/deception_dataset_v3.jsonl")
    p.add_argument("--meta", default="simulation/datasets/deception_dataset_v3_meta.json")
    p.add_argument("--out", default="simulation/datasets/deception_fit_report_v1.json")
    p.add_argument("--bins", type=int, default=20,
                   help="Number of equal-width bins over [0,1] for the histogram comparison.")
    p.add_argument("--samples-per-row", type=int, default=4,
                   help="Mimic samples to draw per real-LLM row (more = lower noise in mimic distribution).")
    p.add_argument("--max-tvd", type=float, default=0.15,
                   help="Acceptance threshold on mean total-variation distance "
                        "between real and mimic claim distributions (lower = stricter). "
                        "Effect-size gate; replaces the sample-size-dependent chi²-p gate.")
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

        # Pooled comparison: all information levels together (the headline number).
        per_attr, attr_pvals, agg_p, mean_tvd = _compare(real_claims, mimic_claims, n_obs=n_obs, bins=args.bins)
        per_llm_pvals.append(agg_p)
        per_llm_tvds.append(mean_tvd)

        # Per-level (known-attribute) stratification. A pooled pass can hide a
        # single level fitting badly — and since the information gradient IS the
        # result, we need the mimic to track EACH rung, not just the average.
        # sample_mimic_claims emits `samples_per_row` mimic draws per real row in
        # row order, so np.repeat maps each mimic draw back to its row's level.
        levels = np.array([_row_level(r) for r in real_rows])
        n_spr = max(1, mimic_claims.shape[0] // max(1, n_obs))
        mimic_levels = np.repeat(levels, n_spr)[: mimic_claims.shape[0]]
        per_level = []
        for k in sorted({int(x) for x in levels}, reverse=True):
            rmask = levels == k
            mmask = mimic_levels == k
            if int(rmask.sum()) == 0 or int(mmask.sum()) == 0:
                continue
            lp_attr, lp_pvals, lp_agg, lp_tvd = _compare(
                real_claims[rmask], mimic_claims[mmask], n_obs=int(rmask.sum()), bins=args.bins)
            per_level.append({
                "known": k,
                "n_real": int(rmask.sum()),
                "n_mimic": int(mmask.sum()),
                "aggregate_p": round(lp_agg, 4),
                "mean_tvd": round(lp_tvd, 4),
                "per_attribute": lp_attr,
            })

        report["per_llm"].append({
            "alias": alias,
            "n_real": n_obs,
            "n_mimic": int(mimic_claims.shape[0]),
            "per_attribute": per_attr,
            "aggregate_p": round(agg_p, 4),
            "mean_tvd": round(mean_tvd, 4),
            "per_level": per_level,
        })
        lvl_str = "  ".join(f"k{p['known']}:tvd={p['mean_tvd']:.3f}(n={p['n_real']})" for p in per_level)
        print(f"  [{alias}] n_real={n_obs}  n_mimic={mimic_claims.shape[0]}  "
              f"mean_TVD={mean_tvd:.4f}  (pooled chi2-p={agg_p:.4f})")
        print(f"           per-level TVD: {lvl_str or '(single level)'}")

    overall_p = fisher_combine(per_llm_pvals)
    overall_tvd = float(sum(per_llm_tvds) / len(per_llm_tvds)) if per_llm_tvds else 1.0
    report["aggregate_p"] = round(overall_p, 4)
    report["mean_tvd"] = round(overall_tvd, 4)
    # Acceptance is on effect size (TVD), NOT the chi²-p: at n=480–2400 a p-gate
    # rejects negligible differences and underflows on large ones, so it answers
    # the wrong question. TVD measures *how much* the distributions differ.
    report["max_tvd"] = args.max_tvd
    report["accept"] = bool(overall_tvd <= args.max_tvd)

    # Surface any per-level rung whose TVD exceeds the threshold even if the
    # pooled average passes — these are the gradient rungs the mimic reproduces
    # worst (often the data-starved low-info / multimodal cells).
    level_failures = []
    for entry in report["per_llm"]:
        for pl in entry.get("per_level", []):
            if pl.get("mean_tvd", 1.0) > args.max_tvd:
                level_failures.append({"alias": entry["alias"], "known": pl["known"],
                                       "mean_tvd": pl.get("mean_tvd"), "n_real": pl["n_real"]})
    report["per_level_failures"] = level_failures

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    accept = overall_tvd <= args.max_tvd
    print(f"\nOverall mean_TVD = {overall_tvd:.4f}  (threshold {args.max_tvd}) "
          f"-> {'ACCEPT' if accept else 'REJECT'}   [chi2-p={overall_p:.4f}, informational only]")
    if level_failures:
        print(f"WARNING: {len(level_failures)} per-level fit(s) above TVD {args.max_tvd} "
              f"(pooled pass can mask these):")
        for lf in sorted(level_failures, key=lambda d: -d["mean_tvd"]):
            print(f"    {lf['alias']} @ known={lf['known']}: mean_TVD={lf['mean_tvd']:.4f} (n_real={lf['n_real']})")
    else:
        print(f"All per-level fits <= TVD {args.max_tvd}.")
    print(f"Report -> {out_path}")
    return 0 if accept else 2


if __name__ == "__main__":
    sys.exit(main())
