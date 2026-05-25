"""
Run a Math-vs-Mimic experiment.

For each math tier, rotates the displaced LLM mimic across all 5 mimics,
running N auctions per displacement. Each tier's results land in a dedicated
folder so you can analyze separately.

Each (tier, displaced) batch uses a unique permutation seed so position
order varies, and a unique --start-seed range so auction RNG varies.

Usage:
    python -m simulation.run_math_experiment --tiers T1,T2,T3 --per-displacement 10
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time


MIMICS = ["Mimic-Grok", "Mimic-Opus", "Mimic-GPT-5.4", "Mimic-Pro", "Mimic-Llama"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tiers", default="T1,T2,T3",
                   help="Comma-separated math tiers (default: T1,T2,T3)")
    p.add_argument("--per-displacement", type=int, default=10,
                   help="Auctions to run per (tier, displaced LLM) combination")
    p.add_argument("--parallel", type=int, default=5,
                   help="Concurrent auctions per batch")
    p.add_argument("--timeout", type=int, default=1200,
                   help="Per-auction polling timeout (seconds)")
    p.add_argument("--base", default="http://localhost:8010",
                   help="Server base URL")
    p.add_argument("--folder-prefix", default="Math",
                   help="Folder name prefix; folder per tier becomes '<prefix> <tier>'")
    args = p.parse_args()

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    total_batches = len(tiers) * len(MIMICS)
    print(f"Math experiment: {len(tiers)} tiers × {len(MIMICS)} displacements "
          f"× {args.per_displacement} auctions = {total_batches * args.per_displacement} total auctions")

    seed_counter = 1
    perm_counter = 1
    done = 0
    overall_t0 = time.time()
    summary: list[dict] = []

    for tier in tiers:
        folder = f"{args.folder_prefix} {tier}"
        for displaced in MIMICS:
            loadout = [f"Math-{tier}"] + [m for m in MIMICS if m != displaced]
            done += 1
            displaced_short = displaced.replace("Mimic-", "")
            print(f"\n[{done}/{total_batches}] tier={tier} displaced={displaced_short} folder={folder!r}")
            print(f"  loadout: {','.join(loadout)}")
            t0 = time.time()
            cmd = [
                sys.executable, "-u", "-m", "simulation.run_mimic_auctions",
                "--base", args.base,
                "--count", str(args.per_displacement),
                "--start-seed", str(seed_counter),
                "--parallel", str(args.parallel),
                "--timeout", str(args.timeout),
                "--folder", folder,
                "--slot-prefix", f"{tier}-{displaced_short[:4]}",
                "--loadout", ",".join(loadout),
                "--permutation-seed", str(perm_counter),
            ]
            result = subprocess.run(cmd)
            dur = time.time() - t0
            print(f"  batch done in {dur:.1f}s (exit={result.returncode})")
            summary.append({
                "tier": tier,
                "displaced": displaced_short,
                "duration_s": round(dur, 1),
                "exit_code": result.returncode,
            })
            seed_counter += args.per_displacement
            perm_counter += 1

    overall = time.time() - overall_t0
    print(f"\n=== Experiment done in {overall:.1f}s ({total_batches} batches) ===")
    failed = [s for s in summary if s["exit_code"] != 0]
    if failed:
        print(f"Failed batches: {len(failed)}")
        for s in failed:
            print(f"  tier={s['tier']} displaced={s['displaced']} exit={s['exit_code']}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
