"""Extract compact mimic-training datasets from a mega-batch CSV export.

The mega-batch export is intentionally verbose because it doubles as an audit
log. This script keeps only the columns needed to train behavior mimics.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SECTION_NAMES = {
    "Mega-Batch Matchups",
    "Mega-Batch Episodes",
    "Mega-Batch Episode Audit",
    "Buyer Rankings",
    "Seller Rankings",
}


AGENT_COLUMNS = [
    "run_id",
    "config_hash",
    "matchup_index",
    "episode",
    "seed",
    "split",
    "buyer_model",
    "agent_model",
    "truth",
    "customer_preferences",
    "omission_prior",
    "buyer_belief_vector",
    "initial_customer_known_attr",
    "initial_customer_known_value",
    "agent_policy",
    "agent_prompt_style",
    "reported_attrs",
    "report_style",
    "behavior_label",
    "true_quality",
    "perceived_quality",
    "final_report_false_claim_count",
    "final_report_omission_count",
    "belief_misalignment",
    "harmful_false_claim_shift",
    "customer_decision",
    "booked",
    "agent",
    "agent_optimality_gap",
]


CUSTOMER_COLUMNS = [
    "run_id",
    "config_hash",
    "matchup_index",
    "episode",
    "seed",
    "split",
    "buyer_model",
    "agent_model",
    "truth",
    "customer_preferences",
    "omission_prior",
    "buyer_belief_vector",
    "initial_customer_known_attr",
    "initial_customer_known_value",
    "final_known_attrs",
    "reported_attrs",
    "true_quality",
    "perceived_quality",
    "quality_threshold",
    "booking_confidence",
    "booking_evidence_score",
    "belief_misalignment",
    "harmful_false_claim_shift",
    "final_report_false_claim_count",
    "final_report_omission_count",
    "quality_gate_reason",
    "terminal_reason",
    "customer",
    "agent",
    "customer_decision",
    "booked",
]


def _read_sections(path: Path) -> dict[str, list[list[str]]]:
    sections: dict[str, list[list[str]]] = {}
    current_name: str | None = None
    current_lines: list[str] = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if line in SECTION_NAMES:
                if current_name and current_lines:
                    sections[current_name] = list(csv.reader(current_lines))
                current_name = line
                current_lines = []
            elif current_name and line.strip():
                current_lines.append(line)

    if current_name and current_lines:
        sections[current_name] = list(csv.reader(current_lines))
    return sections


def _rows(section: list[list[str]]) -> list[dict[str, str]]:
    if not section:
        return []
    header = section[0]
    return [dict(zip(header, row)) for row in section[1:]]


def _write_subset(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def main() -> None:
    parser = argparse.ArgumentParser(description="Create compact mimic datasets from a mega-batch CSV export.")
    parser.add_argument("input_csv", type=Path, help="Path to negotiation_mega_batch_tables CSV export.")
    parser.add_argument("--out-dir", type=Path, default=Path("simulation/datasets"), help="Output directory.")
    parser.add_argument("--prefix", default="mimic", help="Output filename prefix.")
    args = parser.parse_args()

    sections = _read_sections(args.input_csv)
    audit_rows = _rows(sections.get("Mega-Batch Episode Audit", []))
    if not audit_rows:
        audit_rows = _rows(sections.get("Mega-Batch Episodes", []))
    if not audit_rows:
        raise SystemExit("No episode rows found. Expected 'Mega-Batch Episode Audit' or 'Mega-Batch Episodes'.")

    clean_rows = [
        row
        for row in audit_rows
        if not (row.get("error") or row.get("failure_stage") or row.get("failure_type"))
    ]

    agent_path = args.out_dir / f"{args.prefix}_agent_dataset.csv"
    customer_path = args.out_dir / f"{args.prefix}_customer_dataset.csv"
    _write_subset(agent_path, clean_rows, AGENT_COLUMNS)
    _write_subset(customer_path, clean_rows, CUSTOMER_COLUMNS)

    print(f"source_rows={len(audit_rows)}")
    print(f"clean_rows={len(clean_rows)}")
    print(f"agent_dataset={agent_path}")
    print(f"customer_dataset={customer_path}")


if __name__ == "__main__":
    main()
