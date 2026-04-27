from __future__ import annotations

import argparse
import copy
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PAINTING_RE = re.compile(
    r"^(painting_\d+)\s+\|\s+status=([^|]+)\|\s+winner=([^|]+)\|\s+winning_bid=([^\s]+)\s*$"
)
TURN_RAISE_RE = re.compile(
    r"^\s*turn\s+(\d+):\s+(.+?)\s+RAISE\s+to\s+(-?\d+)\s+\(before=(-?\d+),\s*leader_after=([^)]+)\)\s*$"
)
TURN_PASS_RE = re.compile(
    r"^\s*turn\s+(\d+):\s+(.+?)\s+PASS\s+\(before=(-?\d+),\s*leader_after=([^)]+)\)\s*$"
)
TOTAL_PAINTINGS_RE = re.compile(r"^Total paintings processed:\s*(\d+)\s*$")


@dataclass
class ParsedTurn:
    turn_number: int
    bidder_id: str
    action_type: str
    bid_before: int
    leader_after_raw: str
    bid_amount: int | None = None


@dataclass
class ParsedPainting:
    painting_id: str
    status: str
    winner_id: str | None
    winning_bid: int | None
    turns: list[ParsedTurn]


def _canonical_none(text: str | None) -> str | None:
    if text is None:
        return None
    value = str(text).strip()
    if not value or value.lower() in {"none", "null", "n/a", "-"}:
        return None
    return value


def _parse_int(value: str | None, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _min_raise(current_bid: int) -> int:
    if int(current_bid) < 1000:
        return 50
    if int(current_bid) < 3000:
        return 100
    return 250


def _min_legal_bid(current_bid: int, current_leader: str | None, opening_bid: int) -> int:
    if current_leader is None:
        return int(opening_bid)
    return int(current_bid) + int(_min_raise(current_bid))


def _episode_id_from_path(path: Path) -> str:
    parts = path.parts
    try:
        idx = parts.index("auction_exports")
        slot_id = parts[idx - 1] if idx - 1 >= 0 else "unknown_slot"
        run_id = parts[idx + 1] if idx + 1 < len(parts) else path.parent.name
        return f"{slot_id}:{run_id}"
    except ValueError:
        return path.parent.name


def _parse_summary_csv(summary_csv_path: Path) -> dict[str, dict[str, float]]:
    """
    Returns bidder stats from the "Auction Summary" section:
    {
      "GPT-5.4": {
         "paintings_won": 4.0,
         "remaining_budget": 1950.0,
         "total_spent": 8050.0,
      },
      ...
    }
    """
    if not summary_csv_path.exists():
        return {}
    rows: list[list[str]] = []
    with summary_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)
    stats: dict[str, dict[str, float]] = {}
    summary_idx = None
    for idx, row in enumerate(rows):
        if row and row[0].strip() == "Auction Summary":
            summary_idx = idx
            break
    if summary_idx is None or summary_idx + 2 >= len(rows):
        return {}
    header = [h.strip() for h in rows[summary_idx + 1]]
    col = {name: i for i, name in enumerate(header)}
    expected = {"bidder", "paintings_won", "remaining_budget", "total_spent"}
    if not expected.issubset(col):
        return {}
    for row in rows[summary_idx + 2 :]:
        if not row or not row[0].strip():
            break
        bidder = row[col["bidder"]].strip()
        if not bidder:
            continue
        def _f(name: str) -> float:
            try:
                return float(row[col[name]].strip() or 0.0)
            except Exception:
                return 0.0
        stats[bidder] = {
            "paintings_won": _f("paintings_won"),
            "remaining_budget": _f("remaining_budget"),
            "total_spent": _f("total_spent"),
        }
    return stats


def parse_auction_log(log_path: Path) -> tuple[list[ParsedPainting], int | None]:
    paintings: list[ParsedPainting] = []
    total_paintings: int | None = None
    current: ParsedPainting | None = None
    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip("\n")
        m_total = TOTAL_PAINTINGS_RE.match(line.strip())
        if m_total:
            total_paintings = int(m_total.group(1))
            continue
        m_painting = PAINTING_RE.match(line.strip())
        if m_painting:
            if current is not None:
                paintings.append(current)
            pid = m_painting.group(1).strip()
            status = m_painting.group(2).strip()
            winner = _canonical_none(m_painting.group(3))
            winning_bid = _parse_int(_canonical_none(m_painting.group(4)))
            current = ParsedPainting(
                painting_id=pid,
                status=status,
                winner_id=winner,
                winning_bid=winning_bid,
                turns=[],
            )
            continue
        if current is None:
            continue
        m_raise = TURN_RAISE_RE.match(line)
        if m_raise:
            current.turns.append(
                ParsedTurn(
                    turn_number=int(m_raise.group(1)),
                    bidder_id=m_raise.group(2).strip(),
                    action_type="raise",
                    bid_amount=int(m_raise.group(3)),
                    bid_before=int(m_raise.group(4)),
                    leader_after_raw=m_raise.group(5).strip(),
                )
            )
            continue
        m_pass = TURN_PASS_RE.match(line)
        if m_pass:
            current.turns.append(
                ParsedTurn(
                    turn_number=int(m_pass.group(1)),
                    bidder_id=m_pass.group(2).strip(),
                    action_type="pass",
                    bid_amount=None,
                    bid_before=int(m_pass.group(3)),
                    leader_after_raw=m_pass.group(4).strip(),
                )
            )
            continue
    if current is not None:
        paintings.append(current)
    return paintings, total_paintings


def infer_bidder_order(paintings: list[ParsedPainting]) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    for painting in paintings:
        for turn in painting.turns:
            if turn.bidder_id not in seen:
                seen.add(turn.bidder_id)
                order.append(turn.bidder_id)
    return order


def build_rows_for_log(log_path: Path, *, opening_bid: int = 100) -> list[dict[str, Any]]:
    paintings, total_from_log = parse_auction_log(log_path)
    if not paintings:
        return []
    bidder_order = infer_bidder_order(paintings)
    if not bidder_order:
        return []

    summary_stats = _parse_summary_csv(log_path.with_name("auction_tables.csv"))
    # Prefer per-bidder inferred starting budget from summary row; fallback 10000.
    start_budget_by_bidder = {
        bidder: int(
            round(
                float(summary_stats.get(bidder, {}).get("remaining_budget", 10000.0))
                + float(summary_stats.get(bidder, {}).get("total_spent", 0.0))
            )
        )
        for bidder in bidder_order
    }
    total_paintings = total_from_log or len(paintings)
    episode_id = _episode_id_from_path(log_path)

    # Running episode state.
    running_budget = dict(start_budget_by_bidder)
    running_counts = {bidder: 0 for bidder in bidder_order}
    budget_log = {
        bidder: [{"painting_id": "start", "remaining_budget": int(running_budget[bidder])}]
        for bidder in bidder_order
    }
    completed_summaries: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []

    for painting_idx, painting in enumerate(paintings, start=1):
        current_bid = 0
        current_leader: str | None = None
        active_bidders = list(bidder_order)
        passed_bidders: list[str] = []
        current_bids_this_painting: dict[str, int | None] = {bidder: None for bidder in bidder_order}
        round_history: list[dict[str, Any]] = []
        paintings_remaining = max(1, int(total_paintings) - int(len(completed_summaries)))

        for turn in painting.turns:
            min_legal_bid = _min_legal_bid(
                current_bid=current_bid,
                current_leader=current_leader,
                opening_bid=opening_bid,
            )
            public_bid_table = {
                bidder: {
                    "current_bid_this_painting": current_bids_this_painting.get(bidder),
                    "remaining_budget": int(running_budget.get(bidder, 0)),
                    "paintings_won": int(running_counts.get(bidder, 0)),
                }
                for bidder in bidder_order
            }
            llm_input = {
                "painting_number": painting_idx,
                "total_paintings": total_paintings,
                "is_last_painting": painting_idx >= total_paintings,
                "painting_id": painting.painting_id,
                "current_bid": int(current_bid),
                "current_leader": current_leader,
                "your_bidder_id": turn.bidder_id,
                "your_remaining_budget": int(running_budget.get(turn.bidder_id, 0)),
                "minimum_legal_bid": int(min_legal_bid),
                "active_bidders": list(active_bidders),
                "passed_bidders": list(passed_bidders),
                "paintings_remaining": int(paintings_remaining),
                "public_bid_table": public_bid_table,
                "all_budgets": {bidder: int(running_budget.get(bidder, 0)) for bidder in bidder_order},
                "budget_log_by_bidder": copy.deepcopy(budget_log),
                "all_painting_counts": {bidder: int(running_counts.get(bidder, 0)) for bidder in bidder_order},
                "bid_history": copy.deepcopy(round_history),
                "completed_painting_summaries": copy.deepcopy(completed_summaries),
                "warning": "LAST_PAINTING" if painting_idx >= total_paintings else "MORE_PAINTINGS_REMAIN",
            }

            target: dict[str, Any] = {"action_type": turn.action_type}
            if turn.action_type == "raise" and turn.bid_amount is not None:
                target["bid_amount"] = int(turn.bid_amount)
                target["raise_over_min_legal"] = int(turn.bid_amount) - int(min_legal_bid)
                target["raise_size_from_prev_bid"] = int(turn.bid_amount) - int(current_bid)
            else:
                target["bid_amount"] = None
                target["raise_over_min_legal"] = None
                target["raise_size_from_prev_bid"] = None

            dataset_rows.append(
                {
                    "episode_id": episode_id,
                    "source_log_path": str(log_path),
                    "painting_index": painting_idx,
                    "painting_id": painting.painting_id,
                    "turn_number": int(turn.turn_number),
                    "bidder_id": turn.bidder_id,
                    "llm_input": llm_input,
                    "target": target,
                }
            )

            # Transition state using the action from this turn.
            history_entry = {
                "bidder_id": turn.bidder_id,
                "action_type": turn.action_type,
                "turn_number": int(turn.turn_number),
                "bid_before": int(turn.bid_before),
            }
            if turn.action_type == "pass":
                if turn.bidder_id in active_bidders:
                    active_bidders = [b for b in active_bidders if b != turn.bidder_id]
                if turn.bidder_id not in passed_bidders:
                    passed_bidders.append(turn.bidder_id)
                history_entry["leader_after"] = current_leader
            else:
                if turn.bid_amount is not None:
                    bid_amount = int(turn.bid_amount)
                    history_entry["bid_amount"] = bid_amount
                    history_entry["raise_size"] = bid_amount - int(current_bid)
                    current_bid = bid_amount
                    current_leader = turn.bidder_id
                    current_bids_this_painting[turn.bidder_id] = bid_amount
                history_entry["leader_after"] = current_leader
            round_history.append(history_entry)

        # End-of-painting updates.
        winner = painting.winner_id
        winning_bid = int(painting.winning_bid) if painting.winning_bid is not None else None
        if winner and winner in running_budget and winning_bid is not None:
            running_budget[winner] -= int(winning_bid)
            running_counts[winner] += 1
        for bidder in bidder_order:
            budget_log[bidder].append(
                {
                    "painting_id": painting.painting_id,
                    "remaining_budget": int(running_budget[bidder]),
                }
            )
        completed_summaries.append(
            {
                "painting_id": painting.painting_id,
                "winner_id": winner,
                "winning_bid": winning_bid,
                "bid_history": list(round_history),
                "status": painting.status,
            }
        )

    return dataset_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "episode_id",
        "source_log_path",
        "painting_index",
        "painting_id",
        "turn_number",
        "bidder_id",
        "action_type",
        "bid_amount",
        "raise_over_min_legal",
        "raise_size_from_prev_bid",
        "current_bid",
        "current_leader",
        "minimum_legal_bid",
        "your_remaining_budget",
        "your_paintings_won",
        "paintings_remaining",
        "total_paintings",
        "is_last_painting",
        "active_bidders_count",
        "passed_bidders_count",
        "bid_history_count",
        "completed_painting_summaries_count",
        "active_bidders_json",
        "passed_bidders_json",
        "public_bid_table_json",
        "all_budgets_json",
        "budget_log_by_bidder_json",
        "all_painting_counts_json",
        "bid_history_json",
        "completed_painting_summaries_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            inp = row["llm_input"]
            target = row["target"]
            writer.writerow(
                {
                    "episode_id": row["episode_id"],
                    "source_log_path": row["source_log_path"],
                    "painting_index": row["painting_index"],
                    "painting_id": row["painting_id"],
                    "turn_number": row["turn_number"],
                    "bidder_id": row["bidder_id"],
                    "action_type": target.get("action_type"),
                    "bid_amount": target.get("bid_amount"),
                    "raise_over_min_legal": target.get("raise_over_min_legal"),
                    "raise_size_from_prev_bid": target.get("raise_size_from_prev_bid"),
                    "current_bid": inp.get("current_bid"),
                    "current_leader": inp.get("current_leader"),
                    "minimum_legal_bid": inp.get("minimum_legal_bid"),
                    "your_remaining_budget": inp.get("your_remaining_budget"),
                    "your_paintings_won": inp.get("all_painting_counts", {}).get(row["bidder_id"]),
                    "paintings_remaining": inp.get("paintings_remaining"),
                    "total_paintings": inp.get("total_paintings"),
                    "is_last_painting": inp.get("is_last_painting"),
                    "active_bidders_count": len(inp.get("active_bidders") or []),
                    "passed_bidders_count": len(inp.get("passed_bidders") or []),
                    "bid_history_count": len(inp.get("bid_history") or []),
                    "completed_painting_summaries_count": len(inp.get("completed_painting_summaries") or []),
                    "active_bidders_json": json.dumps(inp.get("active_bidders"), ensure_ascii=False),
                    "passed_bidders_json": json.dumps(inp.get("passed_bidders"), ensure_ascii=False),
                    "public_bid_table_json": json.dumps(inp.get("public_bid_table"), ensure_ascii=False),
                    "all_budgets_json": json.dumps(inp.get("all_budgets"), ensure_ascii=False),
                    "budget_log_by_bidder_json": json.dumps(inp.get("budget_log_by_bidder"), ensure_ascii=False),
                    "all_painting_counts_json": json.dumps(inp.get("all_painting_counts"), ensure_ascii=False),
                    "bid_history_json": json.dumps(inp.get("bid_history"), ensure_ascii=False),
                    "completed_painting_summaries_json": json.dumps(
                        inp.get("completed_painting_summaries"), ensure_ascii=False
                    ),
                }
            )


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        value = float(x)
    except Exception:
        return default
    if value != value:  # NaN guard
        return default
    return value


def _stats(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    n = float(len(values))
    mean = sum(values) / n
    vmin = min(values)
    vmax = max(values)
    var = sum((x - mean) * (x - mean) for x in values) / n
    std = var ** 0.5
    return mean, std, vmin, vmax


def _topk(values: list[float], k: int) -> list[float]:
    out = list(values[:k])
    if len(out) < k:
        out.extend([0.0] * (k - len(out)))
    return out


def build_nn_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    episode_set: set[str] = set()
    log_set: set[str] = set()
    bidder_set: set[str] = set()
    for row in rows:
        inp = row.get("llm_input") or {}
        episode_set.add(str(row.get("episode_id") or ""))
        log_set.add(str(row.get("source_log_path") or ""))
        bidder_set.add(str(row.get("bidder_id") or inp.get("your_bidder_id") or ""))
        for k in (inp.get("all_budgets") or {}).keys():
            bidder_set.add(str(k))

    episode_vocab = {name: idx for idx, name in enumerate(sorted(episode_set))}
    log_vocab = {name: idx for idx, name in enumerate(sorted(log_set))}
    bidder_vocab = {name: idx for idx, name in enumerate(sorted(bidder_set))}

    feature_names: list[str] = [
        # Game progress
        "painting_number_frac",       # painting_idx / total_paintings
        "paintings_remaining",        # raw count of paintings left including current
        "is_last_painting",           # binary
        # Round state
        "current_bid_frac",           # current_bid / self_start_budget
        "min_legal_bid_frac",         # minimum_legal_bid / self_start_budget
        "active_bidders_count",       # bidders still alive this round
        "bid_history_count",          # turns taken so far this painting
        # Self state
        "budget_fraction",            # your_remaining / self_start_budget
        "budget_vs_mean_opp",         # your_remaining / mean(opp_budgets)
        "budget_vs_min_legal",        # your_remaining / minimum_legal_bid
        "your_paintings_won",         # raw count
        "paintings_gap_to_leader",    # your_paintings_won - max(all_counts); <=0 if behind
        "fair_share_budget",          # (your_remaining / paintings_remaining) / self_start
        "own_current_bid_frac",       # own bid placed this painting / self_start_budget
        # Last action context
        "last_action_is_raise",       # binary: was the most recent history entry a raise
        "last_bid_frac",              # last bid amount / self_start_budget (0 if pass/none)
        "last_bidder_is_self",        # binary: was that last bidder this agent
        # Opponent budget stats (all normalised by self_start_budget)
        "opp_budget_mean_frac",
        "opp_budget_std_frac",
        "opp_budget_min_frac",
        "opp_budget_max_frac",
        # Opponent count stats
        "opp_count_mean",
        "opp_count_std",
        "opp_count_max",
        # Top-4 opponent budgets sorted descending (normalised)
        "top_opp_budget_frac_1",
        "top_opp_budget_frac_2",
        "top_opp_budget_frac_3",
        "top_opp_budget_frac_4",
        # Top-4 opponent painting counts sorted descending
        "top_opp_count_1",
        "top_opp_count_2",
        "top_opp_count_3",
        "top_opp_count_4",
    ]

    def _safe_div(a: float, b: float) -> float:
        return a / b if b > 0.0 else 0.0

    nn_rows: list[dict[str, Any]] = []
    for row in rows:
        inp = row.get("llm_input") or {}
        target = row.get("target") or {}
        bidder_id = str(row.get("bidder_id") or inp.get("your_bidder_id") or "")

        # Recover per-bidder start budget from the "start" sentinel entry in budget_log.
        budget_log_by_bidder = inp.get("budget_log_by_bidder") or {}
        def _start_budget(bid: str) -> float:
            for entry in (budget_log_by_bidder.get(bid) or []):
                if entry.get("painting_id") == "start":
                    return max(1.0, _safe_float(entry.get("remaining_budget"), 10000.0))
            return 10000.0

        self_start = _start_budget(bidder_id)

        all_budgets = inp.get("all_budgets") or {}
        all_counts = inp.get("all_painting_counts") or {}
        public_bid = inp.get("public_bid_table") or {}
        bid_history = list(inp.get("bid_history") or [])

        painting_idx = int(row.get("painting_index") or inp.get("painting_number") or 1)
        total_paintings = max(1, int(inp.get("total_paintings") or 12))
        paintings_remaining = max(1.0, _safe_float(inp.get("paintings_remaining"), 1.0))
        current_bid = _safe_float(inp.get("current_bid"))
        min_legal = _safe_float(inp.get("minimum_legal_bid"))
        your_budget = _safe_float(inp.get("your_remaining_budget"))
        your_count = _safe_float(all_counts.get(bidder_id, 0.0))

        opp_budgets = sorted(
            [_safe_float(all_budgets[k]) for k in all_budgets if k != bidder_id],
            reverse=True,
        )
        opp_counts = sorted(
            [_safe_float(all_counts[k]) for k in all_counts if k != bidder_id],
            reverse=True,
        )
        opp_b_mean, opp_b_std, opp_b_min, opp_b_max = _stats(opp_budgets) if opp_budgets else (0.0, 0.0, 0.0, 0.0)
        opp_c_mean, opp_c_std, _opp_c_min, opp_c_max = _stats(opp_counts) if opp_counts else (0.0, 0.0, 0.0, 0.0)
        top_opp_budgets = _topk(opp_budgets, 4)
        top_opp_counts = _topk(opp_counts, 4)

        all_counts_vals = [_safe_float(v) for v in all_counts.values()]
        global_count_max = max(all_counts_vals) if all_counts_vals else 0.0

        last_hist = bid_history[-1] if bid_history else {}
        own_current_bid = _safe_float((public_bid.get(bidder_id) or {}).get("current_bid_this_painting"))

        x: list[float] = [
            painting_idx / total_paintings,
            paintings_remaining,
            1.0 if bool(inp.get("is_last_painting")) else 0.0,
            _safe_div(current_bid, self_start),
            _safe_div(min_legal, self_start),
            float(len(inp.get("active_bidders") or [])),
            float(len(bid_history)),
            _safe_div(your_budget, self_start),
            _safe_div(your_budget, opp_b_mean),
            _safe_div(your_budget, min_legal),
            your_count,
            your_count - global_count_max,
            _safe_div(_safe_div(your_budget, paintings_remaining), self_start),
            _safe_div(own_current_bid, self_start),
            1.0 if str(last_hist.get("action_type") or "") == "raise" else 0.0,
            _safe_div(_safe_float(last_hist.get("bid_amount", 0.0)), self_start),
            1.0 if str(last_hist.get("bidder_id") or "") == bidder_id else 0.0,
            _safe_div(opp_b_mean, self_start),
            _safe_div(opp_b_std, self_start),
            _safe_div(opp_b_min, self_start),
            _safe_div(opp_b_max, self_start),
            opp_c_mean,
            opp_c_std,
            opp_c_max,
            *[_safe_div(b, self_start) for b in top_opp_budgets],
            *top_opp_counts,
        ]

        action_type = str(target.get("action_type") or "").lower()
        y_action = 1.0 if action_type == "raise" else 0.0
        y_delta = max(0.0, _safe_float(target.get("raise_over_min_legal"), 0.0))
        cb_dollars = int(_safe_float(inp.get("current_bid"), 0.0))
        step = 50 if cb_dollars < 1000 else (100 if cb_dollars < 3000 else 250)
        y_delta_steps = round(y_delta / step) if step > 0 else 0.0

        nn_rows.append({
            "x": x,
            "y_action": y_action,
            "y_delta": y_delta,
            "y_delta_steps": float(y_delta_steps),
            "y_delta_mask": y_action,
            "step_dollars": float(step),
            "episode_index": int(episode_vocab.get(str(row.get("episode_id") or ""), 0)),
            "source_log_index": int(log_vocab.get(str(row.get("source_log_path") or ""), 0)),
            "painting_index": int(row.get("painting_index") or 0),
            "turn_number": int(row.get("turn_number") or 0),
            "bidder_index": int(bidder_vocab.get(bidder_id, 0)),
        })

    meta = {
        "feature_names": feature_names,
        "input_dim": len(feature_names),
        "target_definition": {
            "y_action": "1 if raise, 0 if pass",
            "y_delta": "overbid above minimum legal bid in raw dollars (0 for pass); mask with y_delta_mask",
            "y_delta_steps": "overbid expressed as # of legal raise increments (50/100/250 by current_bid range); always integer-valued by construction",
            "y_delta_mask": "1 when y_action==1 else 0",
            "step_dollars": "the legal raise increment ($) for this row, derived from current_bid",
        },
        "normalization_note": (
            "budget features are divided by self_start_budget recovered from budget_log_by_bidder start entry "
            "(default 10000); count features are raw integers; all features are bounded"
        ),
        "bidder_vocab": bidder_vocab,
        "episode_vocab": episode_vocab,
        "source_log_vocab": log_vocab,
    }
    return nn_rows, meta


def write_nn_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def discover_default_logs(root: Path) -> list[Path]:
    return sorted(
        root.glob("simulation/.runtime/**/auction_exports/**/auction_bid_log.txt"),
        key=lambda p: str(p),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a turn-level neural-network dataset from simulation auction bid logs. "
            "Each row contains reconstructed LLM-style input + action target."
        )
    )
    parser.add_argument(
        "--input",
        nargs="*",
        default=[],
        help=(
            "One or more auction_bid_log.txt paths, or directories to scan recursively. "
            "If omitted, scans simulation/.runtime/**/auction_exports/**/auction_bid_log.txt."
        ),
    )
    parser.add_argument(
        "--output-jsonl",
        default="simulation/datasets/auction_nn_dataset.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--output-csv",
        default="simulation/datasets/auction_nn_dataset.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-nn-jsonl",
        default="simulation/datasets/auction_nn_dataset_nn.jsonl",
        help="Output numeric-only NN JSONL path.",
    )
    parser.add_argument(
        "--output-nn-meta",
        default="simulation/datasets/auction_nn_dataset_nn_meta.json",
        help="Output NN JSON metadata path.",
    )
    parser.add_argument(
        "--opening-bid",
        type=int,
        default=100,
        help="Opening minimum bid used to compute legal raise floor (default: 100).",
    )
    return parser.parse_args()


def resolve_input_logs(root: Path, inputs: list[str]) -> list[Path]:
    if not inputs:
        return discover_default_logs(root)
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            if path.name == "auction_bid_log.txt":
                paths.append(path)
            continue
        if path.is_dir():
            paths.extend(path.rglob("auction_bid_log.txt"))
    # De-duplicate while preserving order.
    dedup: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        dedup.append(p)
    return dedup


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    log_paths = resolve_input_logs(root, args.input)
    if not log_paths:
        print("No auction_bid_log.txt files found.")
        return
    all_rows: list[dict[str, Any]] = []
    for log_path in log_paths:
        rows = build_rows_for_log(log_path, opening_bid=int(args.opening_bid))
        all_rows.extend(rows)
    output_jsonl = Path(args.output_jsonl)
    output_csv = Path(args.output_csv)
    output_nn_jsonl = Path(args.output_nn_jsonl)
    output_nn_meta = Path(args.output_nn_meta)
    if not output_jsonl.is_absolute():
        output_jsonl = root / output_jsonl
    if not output_csv.is_absolute():
        output_csv = root / output_csv
    if not output_nn_jsonl.is_absolute():
        output_nn_jsonl = root / output_nn_jsonl
    if not output_nn_meta.is_absolute():
        output_nn_meta = root / output_nn_meta
    write_jsonl(output_jsonl, all_rows)
    write_csv(output_csv, all_rows)
    nn_rows, nn_meta = build_nn_rows(all_rows)
    write_nn_jsonl(output_nn_jsonl, nn_rows)
    output_nn_meta.parent.mkdir(parents=True, exist_ok=True)
    output_nn_meta.write_text(json.dumps(nn_meta, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "logs_processed": len(log_paths),
                "rows_written": len(all_rows),
                "output_jsonl": str(output_jsonl),
                "output_csv": str(output_csv),
                "output_nn_jsonl": str(output_nn_jsonl),
                "output_nn_meta": str(output_nn_meta),
                "nn_rows_written": len(nn_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
