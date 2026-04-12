"""
Rewrite each AI memory identity.self_summary into a personality-focused paragraph.

Usage:
    python rewrite_identity_summaries.py
    python rewrite_identity_summaries.py --agents "GPT-4o,GPT-5.4,Sonnet"
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from app import agents, mafia_memory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite memory self_summary for all AIs.")
    parser.add_argument(
        "--agents",
        type=str,
        default="",
        help="Comma-separated agent IDs. Defaults to all known agents with memory files.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=900,
        help="Max output tokens per rewrite call.",
    )
    return parser.parse_args()


def _discover_agents() -> List[str]:
    memory_dir = Path("app/data/mafia_memory")
    file_agents = []
    for path in memory_dir.glob("*.json"):
        if path.name == "_global_meta.json":
            continue
        file_agents.append(path.stem)
    return [a for a in agents.AGENT_NAMES if a in set(file_agents)]


def _build_user_prompt(agent_id: str, memory: dict) -> str:
    identity = dict(memory.get("identity") or {})
    recent = list(memory.get("game_journals") or [])[-2:]
    recent_lines: List[str] = []
    for row in recent:
        if not isinstance(row, dict):
            continue
        narrative = str(row.get("narrative") or "").strip()
        if narrative:
            recent_lines.append(f"- {narrative[:500]}")

    return (
        f"Current self_summary:\n{identity.get('self_summary', '')}\n\n"
        f"Current focus:\n{identity.get('current_focus', '')}\n\n"
        "Recent reflection snippets:\n"
        + ("\n".join(recent_lines) if recent_lines else "- (none)")
        + "\n\nRewrite self_summary as a first-person personality description for how I generally think, communicate, "
        "and make decisions. Keep it evergreen and cross-game.\n"
        "Hard constraints:\n"
        "- Do NOT mention Mafia terms, roles, alignments, nights, days, votes, claims, eliminations, or specific games.\n"
        "- Do NOT mention any player names.\n"
        "- Do NOT include bracketed notes or meta comments.\n"
        "- 4-7 sentences, natural prose, no bullet points.\n"
        "Return only the paragraph text."
    )


async def _rewrite_one(agent_id: str, max_tokens: int) -> None:
    memory = mafia_memory.load_agent_memory(agent_id)
    system_prompt = (
        f"You are {agent_id}. Rewrite your own self-summary as an enduring personality profile. "
        "Focus on voice, temperament, reasoning style, strengths, and blind spots."
    )
    user_prompt = _build_user_prompt(agent_id, memory)

    text = await agents._sandbox_generate_text(
        agent=agent_id,
        system_prompt=system_prompt,
        user_text=user_prompt,
        timeout_s=180.0,
        temperature=0.35,
        max_tokens=max_tokens,
        thinking_budget=-1 if agent_id == "Pro" else None,
    )
    cleaned = (text or "").strip().strip('"').strip()
    if not cleaned:
        raise RuntimeError("empty response")

    memory.setdefault("identity", {})
    memory["identity"]["self_summary"] = cleaned
    memory["identity"]["updated_at"] = datetime.now(timezone.utc).isoformat()
    mafia_memory.save_agent_memory(agent_id, memory)


async def _main() -> int:
    args = _parse_args()
    if args.agents.strip():
        chosen = [x.strip() for x in args.agents.split(",") if x.strip()]
    else:
        chosen = _discover_agents()

    if not chosen:
        print("No agent memories found to rewrite.")
        return 1

    print(f"Rewriting self_summary for: {', '.join(chosen)}")
    ok = 0
    failed = 0
    for agent_id in chosen:
        try:
            await _rewrite_one(agent_id, args.max_tokens)
            ok += 1
            print(f"[OK] {agent_id}")
        except Exception as exc:
            failed += 1
            print(f"[ERR] {agent_id}: {exc}")
    print(f"Done. Success={ok}, Failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))

