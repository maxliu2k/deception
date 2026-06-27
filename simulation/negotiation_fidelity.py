"""Mimic fidelity report for the negotiation pipeline.

Compares real-LLM episode outcomes (from a persisted folder) against
mimic-vs-mimic episode outcomes (run in-process).

Metrics:
- Per-LLM agreement rate (LLM as buyer / as seller)
- Per-LLM mean surplus
- Aggregate outcome chi² + Cramér's V (agreed vs not-agreed × LLM)
- Per-LLM mean turns
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import statistics
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

from simulation.eval_negotiation_pairings import run_one as eval_run_one

import numpy as np


LLMS = ["GPT-5.4", "Grok", "Opus", "Pro", "Llama"]
MIMICS = [f"Mimic-{a}" for a in LLMS]
RUNTIME = Path(__file__).parent / ".runtime" / "save_slots"
CATALOG = Path(__file__).parent / ".runtime" / "save_slots.json"


def load_real_episodes(folder_name: str) -> list[dict]:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    folder_id = None
    for f in catalog.get("folders") or []:
        if f.get("name") == folder_name:
            folder_id = f["folder_id"]; break
    if folder_id is None:
        return []
    slot_ids = [s["slot_id"] for s in catalog.get("slots") or [] if s.get("folder_id") == folder_id]
    episodes = []
    for sid in slot_ids:
        pkl = RUNTIME / sid / "runtime.pkl"
        if not pkl.exists(): continue
        try:
            with pkl.open("rb") as f: rt = pickle.load(f)
        except Exception: continue
        env = rt.get("env")
        if env is None or env.result is None: continue
        if env.config.get("mode") != "buyer_seller_negotiation": continue
        sel = env.config.get("selected_models") or []
        if len(sel) < 2: continue
        d = env.result.derived
        episodes.append({
            "buyer": sel[0], "seller": sel[1],
            "agreed": bool(d.get("agreement_reached", 0)),
            "agreed_price": d.get("agreed_price"),
            "buyer_budget": d.get("buyer_budget"),
            "seller_baseline": d.get("seller_baseline_value"),
            "n_turns": d.get("num_turns"),
        })
    return episodes


def run_mimic_episodes(episodes_per_pairing: int, message_limit: int = 10) -> list[dict]:
    out = []
    seed_iter = iter(range(50000, 100000))
    for b, s in product(MIMICS, MIMICS):
        for _ in range(episodes_per_pairing):
            seed = next(seed_iter)
            res = eval_run_one(b, s, seed=seed, message_limit=message_limit)
            res["buyer"] = res["buyer"][len("Mimic-"):]
            res["seller"] = res["seller"][len("Mimic-"):]
            out.append(res)
    return out


def summarize_per_llm(episodes: list[dict]) -> dict[str, dict]:
    """Per-LLM stats: agreement rate as buyer, as seller; mean surplus; mean turns."""
    stats: dict[str, dict] = {a: {"buyer_eps":0,"buyer_agreed":0,"buyer_surplus":0.0,
                                  "seller_eps":0,"seller_agreed":0,"seller_surplus":0.0,
                                  "turns_sum":0,"turns_n":0}
                              for a in LLMS}
    for ep in episodes:
        ba, sa = ep["buyer"], ep["seller"]
        if ba in stats:
            stats[ba]["buyer_eps"] += 1
            if ep["agreed"]:
                stats[ba]["buyer_agreed"] += 1
                stats[ba]["buyer_surplus"] += float(ep.get("buyer_surplus")
                                                    if "buyer_surplus" in ep
                                                    else (ep["buyer_budget"] - ep["agreed_price"]))
        if sa in stats:
            stats[sa]["seller_eps"] += 1
            if ep["agreed"]:
                stats[sa]["seller_agreed"] += 1
                stats[sa]["seller_surplus"] += float(ep.get("seller_surplus")
                                                     if "seller_surplus" in ep
                                                     else (ep["agreed_price"] - ep["seller_baseline"]))
        if ep.get("n_turns"):
            stats[ba]["turns_sum"] += ep["n_turns"]; stats[ba]["turns_n"] += 1
            stats[sa]["turns_sum"] += ep["n_turns"]; stats[sa]["turns_n"] += 1
    out = {}
    for a in LLMS:
        s = stats[a]
        be = s["buyer_eps"]; se = s["seller_eps"]
        out[a] = {
            "buyer_n": be,
            "buyer_agreement_rate": s["buyer_agreed"] / be if be else 0.0,
            "mean_buyer_surplus": s["buyer_surplus"] / max(1, s["buyer_agreed"]),
            "seller_n": se,
            "seller_agreement_rate": s["seller_agreed"] / se if se else 0.0,
            "mean_seller_surplus": s["seller_surplus"] / max(1, s["seller_agreed"]),
            "mean_turns": s["turns_sum"] / s["turns_n"] if s["turns_n"] else 0.0,
        }
    return out


def chi_sq_v_2x5(real_counts: list[list[int]]) -> tuple[float, float, float]:
    obs = np.array(real_counts, dtype=float)
    rs = obs.sum(axis=1, keepdims=True); cs = obs.sum(axis=0, keepdims=True); g = obs.sum()
    if g == 0: return 0.0, 1.0, 0.0
    exp = rs @ cs / g
    chi2 = float(np.nansum(np.where(exp > 0, (obs - exp) ** 2 / exp, 0.0)))
    df = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    z = ((chi2 / df) ** (1/3) - (1 - 2/(9*df))) / math.sqrt(2/(9*df))
    p = 0.5 * math.erfc(z / math.sqrt(2))
    v = math.sqrt(chi2 / (g * min(obs.shape) - 1)) if (g * (min(obs.shape) - 1)) > 0 else 0.0
    return chi2, p, v


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-folder", default="Negotiation Real LLM v1")
    p.add_argument("--mimic-episodes-per-pairing", type=int, default=4,
                   help="Mimic-vs-mimic episodes per (buyer,seller) pairing (default 4 → 25×4=100 mimic eps)")
    p.add_argument("--message-limit", type=int, default=10)
    args = p.parse_args()

    real = load_real_episodes(args.real_folder)
    print(f"Real episodes loaded: {len(real)}")
    if not real:
        print("ERROR: no real episodes"); return 1

    print(f"\nRunning mimic-vs-mimic ({args.mimic_episodes_per_pairing} per pair, "
          f"{len(MIMICS)**2 * args.mimic_episodes_per_pairing} total)...", flush=True)
    mimic = run_mimic_episodes(args.mimic_episodes_per_pairing, message_limit=args.message_limit)
    print(f"Mimic episodes: {len(mimic)}", flush=True)

    real_stats = summarize_per_llm(real)
    mimic_stats = summarize_per_llm(mimic)

    print(f"\n{'LLM':<10}{'role':<8}{'real n':>8}{'real_agr%':>12}{'mim_agr%':>11}"
          f"{'real_surp':>12}{'mim_surp':>11}{'turns_r':>9}{'turns_m':>9}")
    print("-" * 90)
    for a in LLMS:
        rb = real_stats[a]; mb = mimic_stats[a]
        print(f"{a:<10}{'buyer':<8}{rb['buyer_n']:>8d}"
              f"{rb['buyer_agreement_rate']*100:>11.1f}%"
              f"{mb['buyer_agreement_rate']*100:>10.1f}%"
              f"{rb['mean_buyer_surplus']:>12.1f}"
              f"{mb['mean_buyer_surplus']:>11.1f}"
              f"{rb['mean_turns']:>9.1f}"
              f"{mb['mean_turns']:>9.1f}")
        print(f"{'':<10}{'seller':<8}{rb['seller_n']:>8d}"
              f"{rb['seller_agreement_rate']*100:>11.1f}%"
              f"{mb['seller_agreement_rate']*100:>10.1f}%"
              f"{rb['mean_seller_surplus']:>12.1f}"
              f"{mb['mean_seller_surplus']:>11.1f}"
              f"{rb['mean_turns']:>9.1f}"
              f"{mb['mean_turns']:>9.1f}")

    # Chi² on (agreed, not_agreed) × LLM (combine buyer + seller roles)
    real_table = [[0]*len(LLMS), [0]*len(LLMS)]
    mim_table = [[0]*len(LLMS), [0]*len(LLMS)]
    for ep in real:
        for role in ("buyer", "seller"):
            a = ep[role]
            if a not in LLMS: continue
            i = LLMS.index(a)
            if ep["agreed"]: real_table[0][i] += 1
            else: real_table[1][i] += 1
    for ep in mimic:
        for role in ("buyer", "seller"):
            a = ep[role]
            if a not in LLMS: continue
            i = LLMS.index(a)
            if ep["agreed"]: mim_table[0][i] += 1
            else: mim_table[1][i] += 1
    # Compute chi²/V for real vs mimic per-LLM agreement count
    c2_r, p_r, v_r = chi_sq_v_2x5([real_table[0], mim_table[0]])
    print(f"\nAgreement-rate chi² (real-LLM vs mimic, agreed counts per LLM):")
    print(f"  chi2={c2_r:.2f}  p={p_r:.4f}  Cramér's V={v_r:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
