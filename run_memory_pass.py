"""
Standalone script: parse game_1.txt and run the memory pass for all agents.

Usage (from the Research directory):
    python run_memory_pass.py
"""
from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure the app package is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import mafia_memory

# ---------------------------------------------------------------------------
# Roles from the final reveal line in game_1.txt
# "Final roles: GPT-4o=VIGILANTE, GPT-5.4=TOWN, Flash=TOWN, Pro=MAFIA, Llama=DOCTOR,
#  Sonnet=MAFIA, Opus=TOWN, Grok=SHERIFF, DeepSeek=TOWN, User=MAFIA."
# ---------------------------------------------------------------------------
ROLE_MAP: Dict[str, str] = {
    "GPT-4o":   "VIGILANTE",
    "GPT-5.4":  "TOWN",
    "Flash":    "TOWN",
    "Pro":      "MAFIA",
    "Llama":    "DOCTOR",
    "Sonnet":   "MAFIA",
    "Opus":     "TOWN",
    "Grok":     "SHERIFF",
    "DeepSeek": "TOWN",
    "User":     "MAFIA",
}
ALIGNMENT_MAP: Dict[str, str] = {
    name: ("MAFIA" if role == "MAFIA" else "TOWN")
    for name, role in ROLE_MAP.items()
}
WINNER = "MAFIA"

# Message-line pattern:  [HH:MM:SS] Sender (visibility):
MSG_RE = re.compile(
    r"^\[(\d{2}:\d{2}:\d{2})\]\s+(\S+)\s+\(([^)]+)\):\s*$"
)

# Day-votes summary line from system messages:
# "Day votes: Flash→No Lynch, Pro→No Lynch, ..."
DAY_VOTES_RE = re.compile(r"Day votes: (.+)\. (\w+) is eliminated\. Alignment revealed: \w+\.")
NO_LYNCH_RE = re.compile(r"Day votes: (.+)\. No lynch\.")

# Death from kill: "X dies during the night. Alignment revealed: Y."
NIGHT_DEATH_RE = re.compile(r"(\w[\w.]*) dies during the night\. Alignment revealed: (\w+)\.")

# Eliminaton: "X is eliminated. Alignment revealed: Y."
ELIM_RE = re.compile(r"(\w[\w.]*) is eliminated\. Alignment revealed: \w+\.")


def _parse_vote_string(votes_str: str) -> Dict[str, Optional[str]]:
    """Turn 'Flash→No Lynch, Pro→Sonnet' into {'Flash': None, 'Pro': 'Sonnet'}."""
    result: Dict[str, Optional[str]] = {}
    for part in votes_str.split(","):
        part = part.strip()
        if "→" not in part:
            continue
        voter, _, target = part.partition("→")
        voter = voter.strip()
        target = target.strip()
        if not voter:
            continue
        result[voter] = None if target.lower() in ("no lynch", "no vote", "abstain", "") else target
    return result


def parse_transcript(text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns (messages, day_vote_history, death_history).
    messages: [{id, sender, content, visibility}, ...]
    day_vote_history: [{day_number, votes: {voter: target_or_None}}, ...]
    death_history: [{player, cause, night_number or day_number}, ...]
    """
    messages: List[Dict[str, Any]] = []
    day_vote_history: List[Dict[str, Any]] = []
    death_history: List[Dict[str, Any]] = []

    lines = text.splitlines()
    msg_id = 1
    current_sender: Optional[str] = None
    current_visibility: Optional[str] = None
    current_lines: List[str] = []
    day_number = 0

    def flush():
        nonlocal msg_id
        if current_sender is None:
            return
        content = "\n".join(current_lines).strip()
        if not content:
            return
        vis = current_visibility or "all"
        # Normalise visibility token from transcript.
        if vis == "private_room":
            vis = "private_room"
        elif vis == "all":
            vis = "all"
        else:
            vis = vis

        messages.append({
            "id": msg_id,
            "sender": current_sender,
            "content": content,
            "visibility": vis,
        })
        msg_id += 1

    for raw_line in lines:
        line = raw_line.strip()
        m = MSG_RE.match(raw_line)
        if m:
            flush()
            current_lines = []
            current_sender = m.group(2)
            current_visibility = m.group(3).strip()
            continue

        if current_sender is not None:
            current_lines.append(line)
            continue

        # System-level lines (not inside a message block): parse metadata.
        # e.g. "Day votes: ..." or "X dies during the night."
        dv_elim = DAY_VOTES_RE.search(line)
        if dv_elim:
            vote_str = dv_elim.group(1)
            eliminated = dv_elim.group(2)
            vote_dict = _parse_vote_string(vote_str)
            day_vote_history.append({
                "day_number": day_number,
                "votes": vote_dict,
                "eliminated": eliminated,
            })
            death_history.append({
                "player": eliminated,
                "cause": "elimination",
                "day_number": day_number,
            })
            continue

        dv_nl = NO_LYNCH_RE.search(line)
        if dv_nl:
            vote_str = dv_nl.group(1)
            vote_dict = _parse_vote_string(vote_str)
            day_vote_history.append({
                "day_number": day_number,
                "votes": vote_dict,
                "eliminated": None,
            })
            continue

        nd = NIGHT_DEATH_RE.search(line)
        if nd:
            death_history.append({
                "player": nd.group(1),
                "cause": "night_kill",
            })
            continue

        # Track day/night transitions.
        if re.search(r"\bDay \d+\b", line):
            dm = re.search(r"\bDay (\d+)\b", line)
            if dm:
                day_number = int(dm.group(1))

    flush()
    return messages, day_vote_history, death_history


def build_game_bundle(transcript_path: Path) -> Dict[str, Any]:
    text = transcript_path.read_text(encoding="utf-8")
    messages, day_vote_history, death_history = parse_transcript(text)

    players = []
    for seat, name in enumerate(ROLE_MAP.keys()):
        players.append({
            "seat_index": seat,
            "name": name,
            "role": ROLE_MAP[name],
            "alignment": ALIGNMENT_MAP[name],
        })

    public_state: Dict[str, Any] = {
        "players": players,
        "day_vote_history": day_vote_history,
        "death_history": death_history,
        "winner": WINNER,
        "game_over": True,
    }

    return {
        "game_id": "game_1",
        "public_state": public_state,
        "messages": messages,
    }


async def main() -> None:
    transcript_path = Path(__file__).resolve().parent / "app" / "data" / "transcripts" / "game_1.txt"
    if not transcript_path.exists():
        print(f"ERROR: transcript not found at {transcript_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing {transcript_path} ...")
    bundle = build_game_bundle(transcript_path)
    msgs = bundle["messages"]
    print(f"  {len(msgs)} messages parsed.")
    print(f"  Players: {[p['name'] for p in bundle['public_state']['players']]}")
    print(f"  Day vote records: {len(bundle['public_state']['day_vote_history'])}")

    agent_ids = [name for name in ROLE_MAP if name != "User"]
    print(f"\nRunning memory pass for: {agent_ids}\n")

    report = await mafia_memory.commit_memories_for_game(agent_ids, bundle)
    committed = report.get("committed_agents", 0)
    skipped = report.get("skipped_agents", 0)
    print(f"\n=== Done: {committed} committed, {skipped} skipped ===")
    for r in report.get("reports", []):
        agent = r.get("agent_id", "?")
        sk = r.get("skipped", False)
        err = r.get("error", "")
        src = r.get("generation_source", "")
        if sk:
            print(f"  SKIP  {agent}: {err}")
        else:
            print(f"  OK    {agent}  [{src}]")


if __name__ == "__main__":
    asyncio.run(main())
