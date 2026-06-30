"""Paired mimic-fidelity validation on matched scenarios.

Design (requires the canonical one-seed-per-batch collection scheme):
  * LLM-vs-LLM eval lives in a save-slot folder (e.g. "Negotiation v9 Eval"),
    collected as B batches x 20 cross-pairings, seed = start_seed + batch_idx.
  * For every (scenario_seed, buyer_llm, seller_llm) duel we re-run the SAME
    duel mimic-vs-mimic IN PROCESS on the identical seed, giving a matched pair.

Tests (each assumption-clean for its data type):
  * Agreement (binary, deal reached?): McNemar's exact test, paired by scenario.
  * Extraction (continuous agreed-price surplus, 0 on no-deal): Wilcoxon
    signed-rank, paired by scenario. Per-LLM: a cell's buyer-extraction is
    attributed to the buyer-LLM, seller-extraction to the seller-LLM.
  * Family-wise across the 5 LLMs: Holm-Bonferroni.

Descriptive companions (NOT tests): Wasserstein distance, median |delta|.

Usage:
    python -m simulation.validate_negotiation_paired \
        --llm-folder "Negotiation v9 Eval"
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from scipy import stats

from simulation.eval_negotiation_pairings import run_one


SIM = Path(__file__).parent
RUNTIME = SIM / ".runtime" / "save_slots"
CATALOG = SIM / ".runtime" / "save_slots.json"
LLMS = ["GPT-5.4", "Grok", "Opus", "Pro", "Llama"]


def _log_ext(role: str, price: float, own_value: float) -> float:
    """Log-symmetric extraction; 0 if no legal surplus."""
    if own_value <= 0 or price <= 0:
        return 0.0
    return math.log(own_value / price) if role == "buyer" else math.log(price / own_value)


def load_llm_duels(folder_name: str) -> dict:
    """Return {(seed, buyer_llm, seller_llm): outcome} for the LLM eval folder."""
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    fid = next((f["folder_id"] for f in catalog["folders"] if f["name"] == folder_name), None)
    if fid is None:
        raise SystemExit(f"Folder {folder_name!r} not found")
    out = {}
    for s in catalog["slots"]:
        if s.get("folder_id") != fid:
            continue
        pkl = RUNTIME / s["slot_id"] / "runtime.pkl"
        if not pkl.exists():
            continue
        try:
            rt = pickle.load(pkl.open("rb"))
        except Exception:
            continue
        if not isinstance(rt, dict):
            continue
        env = rt.get("env")
        if env is None or env.result is None:
            continue
        sel = env.config.get("selected_models") or []
        if len(sel) < 2:
            continue
        seed = int(env.config.get("seed") or 0)
        buyer, seller = sel[0], sel[1]
        b = env.world["buyer_true"]
        sll = env.world["seller_true"]
        agreed = env.world.get("agreed_price")
        out[(seed, buyer, seller)] = {
            "agreed": agreed is not None,
            "buyer_ext": _log_ext("buyer", agreed, b.budget) if agreed is not None else 0.0,
            "seller_ext": _log_ext("seller", agreed, sll.baseline_value) if agreed is not None else 0.0,
        }
    return out


def run_mimic_duels(keys: list[tuple]) -> dict:
    """Re-run each (seed, buyer_llm, seller_llm) as mimic-vs-mimic on the same seed."""
    out = {}
    for i, (seed, buyer, seller) in enumerate(keys):
        r = run_one(f"Mimic-{buyer}", f"Mimic-{seller}", seed=seed, message_limit=10)
        agreed = r["agreed"]
        out[(seed, buyer, seller)] = {
            "agreed": agreed,
            "buyer_ext": _log_ext("buyer", r["agreed_price"], r["buyer_budget"]) if agreed else 0.0,
            "seller_ext": _log_ext("seller", r["agreed_price"], r["seller_baseline"]) if agreed else 0.0,
        }
        if (i + 1) % 100 == 0:
            print(f"  mimic duels {i+1}/{len(keys)}", flush=True)
    return out


def mcnemar(llm_vals: list[bool], mim_vals: list[bool]) -> tuple[int, int, float]:
    """McNemar exact test on paired binary outcomes. Returns (b, c, p)."""
    b = sum(1 for l, m in zip(llm_vals, mim_vals) if l and not m)  # LLM yes, mimic no
    c = sum(1 for l, m in zip(llm_vals, mim_vals) if (not l) and m)  # LLM no, mimic yes
    n = b + c
    if n == 0:
        return b, c, 1.0
    # Exact binomial two-sided p on discordant pairs vs p=0.5
    k = min(b, c)
    p = min(1.0, 2.0 * sum(math.comb(n, i) * 0.5 ** n for i in range(0, k + 1)))
    return b, c, p


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj = {}
    running = 0.0
    for rank, (k, p) in enumerate(items):
        a = min(1.0, (m - rank) * p)
        running = max(running, a)  # enforce monotonicity
        adj[k] = running
    return adj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llm-folder", required=True)
    ap.add_argument("--out", default="simulation/datasets/negotiation_paired_report.json")
    args = ap.parse_args()

    print(f"Loading LLM duels from {args.llm_folder!r}...", flush=True)
    llm = load_llm_duels(args.llm_folder)
    keys = sorted(llm.keys())
    print(f"  {len(keys)} matched (seed, buyer, seller) cells", flush=True)

    print("Running matched mimic-vs-mimic duels...", flush=True)
    mim = run_mimic_duels(keys)

    # ---- Agreement: McNemar, pooled + per-LLM ----
    llm_agree = [llm[k]["agreed"] for k in keys]
    mim_agree = [mim[k]["agreed"] for k in keys]
    b, c, p_pool = mcnemar(llm_agree, mim_agree)
    print()
    print("=" * 72)
    print(" AGREEMENT — McNemar exact (paired by scenario)")
    print("=" * 72)
    print(f"Pooled: LLM-agree-rate={sum(llm_agree)/len(keys):.3f}  "
          f"mimic-agree-rate={sum(mim_agree)/len(keys):.3f}  "
          f"discordant b={b} c={c}  p={p_pool:.4f}")

    # Per-LLM agreement (cells where the LLM appears on either side)
    agree_p = {}
    print(f"\n{'LLM':<8} {'n_cells':>7} {'LLM_rate':>9} {'mim_rate':>9} {'b':>4} {'c':>4} {'McNemar_p':>10}")
    print("-" * 56)
    for alias in LLMS:
        idx = [k for k in keys if alias in (k[1], k[2])]
        la = [llm[k]["agreed"] for k in idx]
        ma = [mim[k]["agreed"] for k in idx]
        bb, cc, pp = mcnemar(la, ma)
        agree_p[alias] = pp
        print(f"{alias:<8} {len(idx):>7} {sum(la)/len(idx):>9.3f} {sum(ma)/len(idx):>9.3f} "
              f"{bb:>4} {cc:>4} {pp:>10.4f}")
    agree_holm = holm(agree_p)

    # ---- Extraction: Wilcoxon signed-rank, per-LLM (role-attributed) ----
    print()
    print("=" * 72)
    print(" EXTRACTION — Wilcoxon signed-rank (paired by scenario, role-attributed)")
    print("=" * 72)
    print(f"{'LLM':<8} {'n':>5} {'LLM_med':>8} {'mim_med':>8} {'med|Δ|':>7} {'W1':>7} {'Wilcox_p':>10}")
    print("-" * 60)
    ext_p = {}
    for alias in LLMS:
        diffs, lv, mv = [], [], []
        for k in keys:
            seed, buyer, seller = k
            if buyer == alias:
                lv.append(llm[k]["buyer_ext"]); mv.append(mim[k]["buyer_ext"])
            if seller == alias:
                lv.append(llm[k]["seller_ext"]); mv.append(mim[k]["seller_ext"])
        diffs = [a - bb for a, bb in zip(lv, mv)]
        nonzero = [d for d in diffs if abs(d) > 1e-9]
        if len(nonzero) >= 1:
            try:
                wp = stats.wilcoxon(nonzero).pvalue
            except ValueError:
                wp = 1.0
        else:
            wp = 1.0
        w1 = stats.wasserstein_distance(lv, mv)
        ext_p[alias] = wp
        print(f"{alias:<8} {len(lv):>5} {statistics.median(lv):>8.3f} {statistics.median(mv):>8.3f} "
              f"{statistics.median([abs(d) for d in diffs]):>7.3f} {w1:>7.4f} {wp:>10.4f}")
    ext_holm = holm(ext_p)

    # ---- Family-wise verdict ----
    print()
    print("=" * 72)
    print(" HOLM-BONFERRONI family-wise adjusted p-values (5 LLMs)")
    print("=" * 72)
    print(f"{'LLM':<8} {'agree_raw':>10} {'agree_holm':>11}   {'ext_raw':>9} {'ext_holm':>9}")
    print("-" * 54)
    for alias in LLMS:
        print(f"{alias:<8} {agree_p[alias]:>10.4f} {agree_holm[alias]:>11.4f}   "
              f"{ext_p[alias]:>9.4f} {ext_holm[alias]:>9.4f}")
    n_rej_agree = sum(1 for a in agree_holm.values() if a < 0.05)
    n_rej_ext = sum(1 for a in ext_holm.values() if a < 0.05)
    print()
    print(f"Agreement: {n_rej_agree}/5 rejected at Holm-adjusted alpha=0.05 "
          f"(0 = mimic indistinguishable from LLM)")
    print(f"Extraction: {n_rej_ext}/5 rejected at Holm-adjusted alpha=0.05")

    report = {
        "llm_folder": args.llm_folder,
        "n_cells": len(keys),
        "agreement": {"pooled_p": p_pool, "per_llm_raw": agree_p, "per_llm_holm": agree_holm},
        "extraction": {"per_llm_raw": ext_p, "per_llm_holm": ext_holm},
        "n_rejected_agreement": n_rej_agree,
        "n_rejected_extraction": n_rej_ext,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nSaved report -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
