"""Report per-model token usage from deception episode logs.

After the server captures usage metadata (prompt/completion/reasoning tokens per
call), this walks recent episode_log.json files and summarizes how much each
model actually reasons — the data needed to decide Pro's reasoning-effort level.

reasoning_tokens is the key column: it shows whether Pro is genuinely thinking
hard (high effort) or barely (default/low). Compare it across models and against
its max_tokens budget to see how close it runs to truncation.

Usage:
    python -m simulation.measure_reasoning_usage            # last 2h of runs
    python -m simulation.measure_reasoning_usage --max-age 86400
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path

RUNTIME_ROOT = Path(__file__).resolve().parent / ".runtime"


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    i = min(len(s) - 1, int(q * len(s)))
    return s[i]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-age", type=float, default=7200.0,
                   help="Only include episode logs modified within this many seconds (default 2h).")
    p.add_argument("--root", default=str(RUNTIME_ROOT))
    args = p.parse_args()

    now = time.time()
    root = Path(args.root)
    per_model: dict[str, dict[str, list]] = collections.defaultdict(
        lambda: {"reasoning": [], "completion": [], "total": [], "with_usage": 0, "rounds": 0})
    n_logs = 0
    for c in root.rglob("episode_log.json"):
        try:
            if now - c.stat().st_mtime > args.max_age:
                continue
            payload = json.loads(c.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        n_logs += 1
        aliases = {a["agent_id"]: a.get("alias", a["agent_id"]) for a in payload.get("agents", [])}
        for rnd in payload.get("rounds", []):
            for aid, blob in (rnd.get("reasoning_by_agent") or {}).items():
                m = aliases.get(aid, aid)
                d = per_model[m]
                d["rounds"] += 1
                usage = (blob or {}).get("usage") if isinstance(blob, dict) else None
                if not isinstance(usage, dict):
                    continue
                d["with_usage"] += 1
                if usage.get("reasoning_tokens") is not None:
                    d["reasoning"].append(int(usage["reasoning_tokens"]))
                if usage.get("completion_tokens") is not None:
                    d["completion"].append(int(usage["completion_tokens"]))
                if usage.get("total_tokens") is not None:
                    d["total"].append(int(usage["total_tokens"]))

    print(f"episode logs scanned: {n_logs}  (within {args.max_age:.0f}s)\n")
    if not per_model:
        print("No usage data found. Run fresh episodes after the usage-capture change, then re-run.")
        return 0
    print(f"{'model':>10}{'rounds':>7}{'w/usage':>8}{'reasonMean':>11}{'reasonP90':>10}"
          f"{'reasonMax':>10}{'complMean':>10}{'totalMean':>10}")
    for m in sorted(per_model):
        d = per_model[m]
        r = d["reasoning"]
        rm = statistics.mean(r) if r else float("nan")
        cm = statistics.mean(d["completion"]) if d["completion"] else float("nan")
        tm = statistics.mean(d["total"]) if d["total"] else float("nan")
        print(f"{m:>10}{d['rounds']:>7}{d['with_usage']:>8}{rm:>11.0f}"
              f"{_pct(r, 0.9):>10.0f}{(max(r) if r else float('nan')):>10.0f}{cm:>10.0f}{tm:>10.0f}")
    print("\nreasonMax near a model's max_tokens budget => it is truncating. "
          "Low reasonMean for Pro vs others => it is NOT reasoning at high effort.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
