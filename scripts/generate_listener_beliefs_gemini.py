#!/usr/bin/env python3
"""
Generate partial listener beliefs for each scenario using Gemini Flash.

For each record, create:
- exactly 2 known truths from the 5 scenario ground-truth booleans
- exactly 3 unknown truths

By default this script is restricted to data/healthcare.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def _read_key_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw.strip())
    if raw.startswith("{") and raw.endswith("}"):
        return json.loads(raw)
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise ValueError("No JSON object found in model output.")
    return json.loads(match.group(0))


def _call_gemini(prompt: str, api_key: str, model: str, timeout_s: float = 60.0) -> str:
    encoded_key = urllib.parse.quote(api_key, safe="")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{urllib.parse.quote(model, safe='')}:generateContent?key={encoded_key}"
    )
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 900,
            "responseMimeType": "application/json",
            "thinkingConfig": {
                "thinkingLevel": "low",
                "includeThoughts": False,
            },
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Gemini HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini connection error: {exc}") from exc

    candidates = obj.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"No candidates returned: {obj}")
    parts = (((candidates[0] or {}).get("content") or {}).get("parts") or [])
    text = "".join((p.get("text") or "") for p in parts).strip()
    if not text:
        raise RuntimeError(f"Empty text returned: {obj}")
    return text


def _build_prompt(item: dict[str, Any]) -> str:
    gt = item.get("groundtruth") or {}
    keys = list(gt.keys()) if isinstance(gt, dict) else []
    payload = {
        "topic": item.get("topic"),
        "dimension": item.get("dimension"),
        "goal": item.get("goal"),
        "question": item.get("question"),
        "scenario_prompt": item.get("L2-self-pressure"),
        "groundtruth": gt,
    }
    return (
        "Create a partial listener belief state for this deception scenario.\n"
        "The listener should know only part of the truth.\n"
        "Rules:\n"
        "1) Select exactly 2 truths the listener knows for sure.\n"
        "2) Mark exactly 3 truths as unknown to the listener.\n"
        "3) Use only these truth names and do not invent new keys:\n"
        f"{keys}\n"
        "4) Keep reasons short and realistic (records, prior diagnosis, visible cues, etc.).\n"
        "5) Output strict JSON only.\n\n"
        "Return format:\n"
        "{\n"
        '  "known_truths": [\n'
        '    {"name": "truth_key", "reason": "short reason"},\n'
        "    ... exactly 2 ...\n"
        "  ],\n"
        '  "unknown_truths": ["truth_key", "truth_key", "truth_key"],\n'
        '  "summary": "one short sentence about listener uncertainty"\n'
        "}\n\n"
        f"Case:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def _generate_listener_beliefs(
    item: dict[str, Any],
    api_key: str,
    model: str,
    retries: int,
    sleep_s: float,
) -> dict[str, Any]:
    groundtruth = item.get("groundtruth") or {}
    if not isinstance(groundtruth, dict) or len(groundtruth) != 5:
        raise ValueError("Expected groundtruth to be a dict with exactly 5 booleans.")
    truth_keys = set(groundtruth.keys())

    prompt = _build_prompt(item)
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            text = _call_gemini(prompt=prompt, api_key=api_key, model=model)
            obj = _extract_json_object(text)

            known = obj.get("known_truths")
            unknown = obj.get("unknown_truths")
            summary = str(obj.get("summary") or "").strip()
            if not isinstance(known, list) or len(known) != 2:
                raise ValueError("known_truths must contain exactly 2 items.")
            if not isinstance(unknown, list) or len(unknown) != 3:
                raise ValueError("unknown_truths must contain exactly 3 items.")
            if not summary:
                raise ValueError("summary is required.")

            known_names: list[str] = []
            known_reasons: dict[str, str] = {}
            for item_known in known:
                if not isinstance(item_known, dict):
                    raise ValueError("known_truths item must be an object.")
                name = str(item_known.get("name") or "").strip()
                reason = str(item_known.get("reason") or "").strip()
                if not name:
                    raise ValueError("known truth name missing.")
                if not reason:
                    raise ValueError("known truth reason missing.")
                known_names.append(name)
                known_reasons[name] = reason

            unknown_names = [str(x).strip() for x in unknown if str(x).strip()]
            if len(unknown_names) != 3:
                raise ValueError("unknown_truths contains empty entries.")
            if len(set(known_names)) != 2:
                raise ValueError("known_truths names must be unique.")
            if len(set(unknown_names)) != 3:
                raise ValueError("unknown_truths names must be unique.")

            if not set(known_names).issubset(truth_keys):
                raise ValueError("known_truths contains invalid key(s).")
            if not set(unknown_names).issubset(truth_keys):
                raise ValueError("unknown_truths contains invalid key(s).")
            if set(known_names) & set(unknown_names):
                raise ValueError("known and unknown truth keys overlap.")
            if set(known_names) | set(unknown_names) != truth_keys:
                raise ValueError("known+unknown keys must cover all 5 truths.")

            known_truths = {k: bool(groundtruth[k]) for k in known_names}
            return {
                "known_truths": known_truths,
                "known_truth_reasons": known_reasons,
                "unknown_truths": unknown_names,
                "known_count": 2,
                "unknown_count": 3,
                "summary": summary,
            }
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(sleep_s * attempt)
            continue

    assert last_exc is not None
    raise last_exc


def _iter_input_files(raw_inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in raw_inputs:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.glob("*.json")))
        elif any(ch in raw for ch in "*?[]"):
            files.extend(sorted(Path().glob(raw)))
        else:
            files.append(p)
    out: list[Path] = []
    seen: set[Path] = set()
    for f in files:
        r = f.resolve()
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate incomplete listener beliefs (2 known truths, 3 unknown truths) using Gemini Flash."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=["data/healthcare.json"],
        help="Input JSON file(s). Default: data/healthcare.json",
    )
    parser.add_argument(
        "--model",
        default="gemini-3.1-flash-lite-preview",
        help="Gemini model ID. Default: gemini-3.1-flash-lite-preview",
    )
    parser.add_argument(
        "--key-file",
        default="keys/geminikey.txt",
        help="Path to Gemini API key file. Default: keys/geminikey.txt",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Limit items per file (0 means all).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per record when model output is invalid.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes in place. Without this flag, script runs dry-run only.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="When used with --apply, write a .bak backup before overwriting.",
    )
    parser.add_argument(
        "--allow-non-healthcare",
        action="store_true",
        help="Allow processing files other than data/healthcare.json.",
    )
    args = parser.parse_args()

    api_key = _read_key_file(Path(args.key_file))
    if not api_key:
        print(f"ERROR: Gemini key missing in {args.key_file}", file=sys.stderr)
        return 2

    files = _iter_input_files(args.input)
    if not files:
        print("ERROR: No input files found.", file=sys.stderr)
        return 2

    allowed_file = (Path.cwd() / "data" / "healthcare.json").resolve()
    if not args.allow_non_healthcare:
        disallowed = [f for f in files if f != allowed_file]
        if disallowed:
            print(
                "ERROR: This run is restricted to data/healthcare.json. "
                "Pass --allow-non-healthcare to override.",
                file=sys.stderr,
            )
            for bad in disallowed:
                print(f"  disallowed: {bad}", file=sys.stderr)
            return 2

    print(f"Model: {args.model}")
    print(f"Files: {len(files)}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")

    for file_path in files:
        if not file_path.exists():
            print(f"[skip] missing file: {file_path}")
            continue
        original_text = file_path.read_text(encoding="utf-8")
        try:
            data = json.loads(original_text)
        except Exception as exc:
            print(f"[error] failed to parse {file_path}: {exc}")
            continue

        if not isinstance(data, list):
            print(f"[error] expected list in {file_path}, got {type(data).__name__}")
            continue

        total = len(data) if args.max_items <= 0 else min(args.max_items, len(data))
        updated = 0
        print(f"\nProcessing {file_path} ({total} records)...")
        for idx, record in enumerate(data[:total], start=1):
            if not isinstance(record, dict):
                print(f"  [{idx}/{total}] skip non-object record")
                continue
            try:
                beliefs = _generate_listener_beliefs(
                    item=record,
                    api_key=api_key,
                    model=args.model,
                    retries=max(1, args.retries),
                    sleep_s=1.5,
                )
                record["listener_beliefs"] = beliefs
                updated += 1
                print(f"  [{idx}/{total}] ok")
            except Exception as exc:
                print(f"  [{idx}/{total}] error: {exc}")

        if args.apply:
            if args.backup:
                backup_path = file_path.with_suffix(file_path.suffix + ".bak")
                backup_path.write_text(original_text, encoding="utf-8")
            file_path.write_text(json.dumps(data, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
            print(f"[written] {file_path} (updated={updated})")
        else:
            print(f"[dry-run] {file_path} (would update={updated})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
