import asyncio
import json
import logging
import os
import random
import time
import re
import hashlib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List

from openai import AsyncOpenAI
try:
    from anthropic import AsyncAnthropic
except Exception:  # pragma: no cover - handled at runtime when Anthropic models are used
    AsyncAnthropic = None

from . import db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
LOGGER = logging.getLogger("agent_arena")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_REFERER = os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost")
OPENROUTER_TITLE = os.environ.get("OPENROUTER_X_TITLE", "Research Simulation")


DEFAULT_AGENT_MODELS = {
    "4o": "gpt-4o",
    "5.4": "gpt-5.4",
    "Flash": "gemini-3-flash-preview",
    "Pro": "gemini-3.1-pro-preview",
    "Haiku": "claude-haiku-4-5",
    "Sonnet": "claude-sonnet-4-6",
    "Opus": "claude-opus-4-6",
    "Grok": "x-ai/grok-4.1-fast",
    "Kimi": "moonshotai/kimi-k2.5",
    "DeepSeek": "deepseek/deepseek-v3.2",
    "Llama": "meta-llama/llama-4-maverick",
    "GLM": "z-ai/glm-5",
}
OPENROUTER_AGENT_MODELS = {
    "4o": "openai/gpt-4o",
    "5.4": "openai/gpt-5.4",
    "Flash": "google/gemini-3-flash-preview",
    "Pro": "google/gemini-3.1-pro-preview",
    "Haiku": "anthropic/claude-haiku-4.5",
    "Sonnet": "anthropic/claude-sonnet-4.6",
    "Opus": "anthropic/claude-opus-4.6",
    "Grok": "x-ai/grok-4.1-fast",
    "Kimi": "moonshotai/kimi-k2.5",
    "DeepSeek": "deepseek/deepseek-v3.2",
    "Llama": "meta-llama/llama-4-maverick",
    "GLM": "z-ai/glm-5",
}
AGENT_NAMES = list(DEFAULT_AGENT_MODELS.keys())
OPENROUTER_ONLY_AGENT_NAMES = {
    "Grok",
    "Kimi",
    "DeepSeek",
    "Llama",
    "GLM",
}
ANTHROPIC_AGENT_NAMES = {
    "Haiku",
    "Sonnet",
    "Opus",
}
GEMINI_AGENT_NAMES = {
    "Flash",
    "Pro",
}
AVAILABLE_TOPICS = [
    "rhetorical analysis",
    "narrative predictions",
    "societal conclusions",
    "symbols and themes",
    "plot devices",
]
_ENABLED_TOPICS = set(AVAILABLE_TOPICS)
_SEMINAR_MODE = "socratic"  # socratic | live_reaction
SEMINAR_RUBRIC = (
    "Socratic seminar rubric for participants: aim for a natural, evidence-driven conversation. "
    "Your highest-value moves are: (1) make a specific interpretation of the text, (2) make or refine a concrete prediction, "
    "(3) ask a focused follow-up question that advances analysis, and (4) build on or challenge a peer with reasons. "
    "Anchor claims in textual details (quote or close paraphrase), then explain how language choices create meaning. "
    "Prioritize rhetorical analysis: diction, imagery, syntax, tone, repetition, symbolism, and point of view. "
    "Use causal reasoning (because this detail appears, this implication follows). "
    "Prefer depth over breadth; avoid vague generalizations that are not tied to the text. "
    "A full participation includes all three: textual evidence, narrative/rhetorical significance, and a clear interpretive or predictive conclusion. "
    "Two full participations plus other useful contributions can earn 100. "
    "If you provide only two full participations and nothing else useful, apply a 10-point deduction. "
    "Without any full participations, the maximum score is 90."
)
POLL_MIN = 0.5
POLL_MAX = 1.0
REQUEST_COOLDOWN = 4.0
RANDOM_CHANCE = 0.03
SPEAK_WPM = 120
MAX_MESSAGES = 28
ROOM_NAME = "main"
REQUEST_STALE_SECONDS = 90.0
IDLE_FALLBACK_SECONDS = 20.0
MIN_TURN_GAP_SECONDS = 6.0
PASS_BACKOFF_SECONDS = 15.0
PRELUDE_PATH = Path(__file__).resolve().parents[1] / "prelude.txt"
KEYS_PATH = Path(__file__).resolve().parents[1] / "keys"
APP_PRELUDE_PATH = Path(__file__).resolve().parent / "prelude.txt"
TEXTS_DIR = Path(__file__).resolve().parent / "texts"

_paused = False
_SPEAKING_ORDER: List[str] = []
_ORDER_CURSOR: int = 0
_SELECTED_CHAPTER_IDS: List[str] = []
_allow_new_turns = True
_live_reaction_turns_remaining = 0
_speaking_done_event: asyncio.Event | None = None  # set by client when audio playback ends
_PREFETCH_DRAFT: dict | None = None
_candidate_version = 0
_candidate_state: Dict[str, Any] = {
    "candidate_agent": None,
    "candidate_snapshot_message_id": None,
    "candidate_text": None,
    "candidate_status": "idle",
    "candidate_version": 0,
}
_post_message_hook: Callable[[Dict[str, Any]], Awaitable[None]] | None = None
_GEMINI_CACHE_BY_KEY: Dict[str, str] = {}


def set_paused(value: bool) -> None:
    global _paused
    _paused = value


def set_turn_intake_enabled(value: bool) -> None:
    global _allow_new_turns
    _allow_new_turns = value


def start_live_reaction_cycle(members: List[str]) -> None:
    """Start exactly one pass through the queue for live reaction mode."""
    global _live_reaction_turns_remaining, _allow_new_turns
    reset_turn_order(members)
    _live_reaction_turns_remaining = max(0, len(members or []))
    _allow_new_turns = _live_reaction_turns_remaining > 0


def set_post_message_hook(hook: Callable[[Dict[str, Any]], Awaitable[None]] | None) -> None:
    global _post_message_hook
    _post_message_hook = hook


def reset_pipeline_state() -> None:
    global _candidate_version, _candidate_state, _PREFETCH_DRAFT, _speaking_done_event, _live_reaction_turns_remaining
    _candidate_version += 1
    _candidate_state = {
        "candidate_agent": None,
        "candidate_snapshot_message_id": None,
        "candidate_text": None,
        "candidate_status": "idle",
        "candidate_version": _candidate_version,
    }
    _PREFETCH_DRAFT = None
    _live_reaction_turns_remaining = 0
    if _speaking_done_event is not None:
        _speaking_done_event.clear()


def abort_candidate_thinking() -> None:
    """Stop speculative next-speaker work when the seminar timer expires."""
    global _candidate_version, _candidate_state, _PREFETCH_DRAFT
    _candidate_version += 1
    if _candidate_state.get("candidate_agent"):
        _candidate_state = {
            "candidate_agent": _candidate_state.get("candidate_agent"),
            "candidate_snapshot_message_id": _candidate_state.get("candidate_snapshot_message_id"),
            "candidate_text": None,
            "candidate_status": "aborted",
            "candidate_version": _candidate_version,
            "candidate_ts": time.time(),
        }
    _PREFETCH_DRAFT = None


def get_candidate_state() -> Dict[str, Any]:
    return dict(_candidate_state)


def reset_turn_order(members: List[str] | None = None) -> None:
    global _SPEAKING_ORDER, _ORDER_CURSOR
    if members:
        _SPEAKING_ORDER = list(members)
        random.shuffle(_SPEAKING_ORDER)
    else:
        _SPEAKING_ORDER = []
    _ORDER_CURSOR = 0


def get_turn_queue_order(members: List[str], active_agent: str | None = None) -> List[str]:
    order = [name for name in _SPEAKING_ORDER if name in members]
    for name in members:
        if name not in order:
            order.append(name)
    return order


def get_prefetch_draft() -> dict | None:
    return _PREFETCH_DRAFT


def clear_runtime_caches() -> None:
    """Clear per-process caches so reset starts from a true clean slate."""
    _GEMINI_CACHE_BY_KEY.clear()


def signal_speaking_done() -> None:
    """Called by the server when the client reports audio playback finished."""
    if _speaking_done_event is not None:
        _speaking_done_event.set()


def set_topic_focus(topics: List[str] | None) -> None:
    global SEMINAR_RUBRIC, _ENABLED_TOPICS
    requested = set((topics or AVAILABLE_TOPICS))
    valid = {t for t in AVAILABLE_TOPICS if t in requested}
    if not valid:
        valid = set(AVAILABLE_TOPICS)
    _ENABLED_TOPICS = valid
    if _SEMINAR_MODE == "live_reaction":
        SEMINAR_RUBRIC = (
            "Live reaction mode: react to newly revealed lines from the selected text. "
            "Focus on immediate interpretation, prediction, and rhetorical observation of the revealed lines only. "
            "Ground each turn in those lines and connect to recent discussion context."
        )
        return

    directives: List[str] = []
    if "narrative predictions" in valid:
        directives.append(
            "make concrete, testable predictions and refine them with new evidence"
        )
    if "rhetorical analysis" in valid:
        directives.append(
            "analyze diction, imagery, syntax, tone, repetition, symbolism, and point of view"
        )
    if "symbols and themes" in valid:
        directives.append(
            "track recurring symbols/themes and explain how they evolve"
        )
    if "plot devices" in valid:
        directives.append(
            "identify plot devices and explain their narrative effect"
        )
    if "societal conclusions" in valid:
        directives.append(
            "connect to societal implications only when directly grounded in textual evidence"
        )

    focus_text = "; ".join(directives)
    SEMINAR_RUBRIC = (
        "Socratic seminar rubric for participants: aim for a natural, evidence-driven conversation. "
        f"Current focus areas: {focus_text}. "
        "Your highest-value moves are: make a specific interpretation, make/refine a prediction, "
        "ask a focused follow-up, and build on or challenge a peer with reasons. "
        "Anchor claims in textual details, then explain how language choices create meaning. "
    "Use causal reasoning (because this detail appears, this implication follows). "
    "Do not tunnel on one thread; actively connect multiple live threads in the discussion and show how ideas relate or conflict. "
    "At least some turns should synthesize across two or more prior points instead of only replying to the most recent speaker. "
    "Prefer depth over breadth; avoid vague generalizations not tied to text. "
        "A full participation includes textual evidence, narrative/rhetorical significance, and a clear interpretive or predictive conclusion. "
        "Two full participations plus other useful contributions can earn 100. "
        "If there are only two full participations and nothing else useful, subtract 10 points. "
        "Without any full participations, the maximum score is 90."
    )


def set_seminar_mode(mode: str | None) -> None:
    global _SEMINAR_MODE
    m = (mode or "socratic").strip().lower()
    if m == "feedback":
        m = "live_reaction"
    if m not in {"socratic", "live_reaction"}:
        m = "socratic"
    _SEMINAR_MODE = m
    # Rebuild rubric in the new mode while preserving currently selected topics.
    set_topic_focus([t for t in AVAILABLE_TOPICS if t in _ENABLED_TOPICS])


def get_seminar_mode() -> str:
    return _SEMINAR_MODE


def get_selected_text_lines() -> List[str]:
    text = _load_seminar_text()
    lines: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line:
            lines.append(line)
    return lines


set_topic_focus(AVAILABLE_TOPICS)


class AgentState:
    def __init__(self, name: str) -> None:
        self.name = name
        self.last_request_ts = 0.0
        self.last_seen_message_id = 0


def _format_context(messages: List[Dict]) -> str:
    return "\n".join([f"{m['sender']}: {m['content']}" for m in messages[-8:]])


def _format_full_transcript(messages: List[Dict]) -> str:
    return "\n".join([f"{m['sender']}: {m['content']}" for m in messages])


def _truncate_for_packet(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1].rstrip() + "..."


async def _build_live_reaction_packet(agent: str) -> Dict[str, Any]:
    latest_turn = await db.get_latest_live_turn(room=ROOM_NAME)
    if not latest_turn:
        return {
            "turn_id": 0,
            "new_story_text": "",
            "discussion_this_turn": [],
            "previous_turn_summary": "",
            "story_state": {},
            "agent_state": {},
            "retrieved_snippets": [],
        }
    turn_id = int(latest_turn.get("id") or 0)
    discussion = await db.get_live_discussion_by_turn(turn_id, room=ROOM_NAME)
    previous_summary = await db.get_previous_turn_summary(turn_id, room=ROOM_NAME)
    story_state = await db.get_live_story_state(room=ROOM_NAME)
    agent_state = await db.get_live_agent_state(agent, room=ROOM_NAME)
    retrieved = await db.get_recent_live_discussion(room=ROOM_NAME, limit=6, before_turn_id=turn_id)
    snippets = [
        f"{r.get('agent_id')}: {_truncate_for_packet(str(r.get('text') or ''), 220)}"
        for r in retrieved[-4:]
    ]
    return {
        "turn_id": turn_id,
        "new_story_text": latest_turn.get("text_chunk") or "",
        "discussion_this_turn": [
            f"{d.get('agent_id')}: {_truncate_for_packet(str(d.get('text') or ''), 260)}"
            for d in discussion[-8:]
        ],
        "previous_turn_summary": previous_summary or latest_turn.get("prev_turn_summary") or "",
        "story_state": story_state if isinstance(story_state, dict) else {},
        "agent_state": agent_state if isinstance(agent_state, dict) else {},
        "retrieved_snippets": snippets,
    }


def _format_live_packet_for_prompt(packet: Dict[str, Any]) -> str:
    story_state = packet.get("story_state") or {}
    agent_state = packet.get("agent_state") or {}
    story_lines = []
    if story_state.get("chapters"):
        story_lines.append(f"Chapters in scope: {', '.join(story_state.get('chapters', [])[:8])}")
    if story_state.get("theme_map"):
        story_lines.append(f"Themes/rhetorical map: {', '.join(story_state.get('theme_map', [])[:10])}")
    if story_state.get("canon_facts"):
        story_lines.append("Canon facts:\n- " + "\n- ".join(story_state.get("canon_facts", [])[:10]))
    if story_state.get("unresolved_threads"):
        story_lines.append("Unresolved threads:\n- " + "\n- ".join(story_state.get("unresolved_threads", [])[:8]))
    agent_lines = []
    if agent_state.get("stance"):
        agent_lines.append(f"Current stance: {agent_state.get('stance')}")
    if agent_state.get("core_belief"):
        agent_lines.append(f"Core belief: {agent_state.get('core_belief')}")
    if agent_state.get("reasoning_style"):
        agent_lines.append(f"Reasoning style: {agent_state.get('reasoning_style')}")
    if agent_state.get("blind_spot"):
        agent_lines.append(f"Blind spot: {agent_state.get('blind_spot')}")
    if agent_state.get("next_move"):
        agent_lines.append(f"Next move: {agent_state.get('next_move')}")
    if agent_state.get("active_hypotheses"):
        agent_lines.append("Active hypotheses:\n- " + "\n- ".join(agent_state.get("active_hypotheses", [])[:8]))
    if agent_state.get("open_questions"):
        oq = agent_state.get("open_questions", [])
        if isinstance(oq, str):
            agent_lines.append(f"Open question: {oq}")
        else:
            agent_lines.append("Open questions:\n- " + "\n- ".join(oq[:6]))

    this_turn_discussion = packet.get("discussion_this_turn") or []
    snippets = packet.get("retrieved_snippets") or []
    return (
        f"Live context packet (turn {packet.get('turn_id', 0)}):\n\n"
        f"New story text this turn:\n{_truncate_for_packet(packet.get('new_story_text', ''), 2600)}\n\n"
        f"Discussion so far this turn:\n{chr(10).join(this_turn_discussion) if this_turn_discussion else '(none yet)'}\n\n"
        f"Previous-turn distilled discussion:\n{_truncate_for_packet(packet.get('previous_turn_summary', ''), 1400)}\n\n"
        f"Shared story state:\n{chr(10).join(story_lines) if story_lines else '(empty)'}\n\n"
        f"Your agent state:\n{chr(10).join(agent_lines) if agent_lines else '(empty)'}\n\n"
        f"Retrieved prior snippets:\n{chr(10).join(snippets) if snippets else '(none)'}"
    )


def _estimated_speak_seconds(text: str, wpm: int = SPEAK_WPM) -> float:
    words = max(1, len(text.split()))
    return max(1.0, (words / max(1, wpm)) * 60.0)


def _agent_model(agent: str) -> str:
    fallback = os.environ.get("OPENAI_MODEL", "").strip() or "gpt-4o"
    if agent in AGENT_NAMES:
        if _use_openrouter_for_llms():
            return OPENROUTER_AGENT_MODELS.get(agent, DEFAULT_AGENT_MODELS[agent])
        return DEFAULT_AGENT_MODELS[agent]
    return fallback


def _participant_count_text() -> str:
    count = len(AGENT_NAMES)
    if count == 3:
        return "three"
    if count == 4:
        return "four"
    if count == 5:
        return "five"
    return str(count)


def _chapter_label_from_name(name: str) -> str:
    stem = Path(name).stem
    parts = stem.split("_")
    text_parts = parts[2:] if len(parts) > 2 and parts[0].isdigit() and parts[1].isdigit() else parts
    raw = " ".join(text_parts).replace("-", " ").strip()
    return raw.title() if raw else stem


def _chapter_category_from_name(name: str) -> str:
    stem = Path(name).stem
    parts = stem.split("_")
    lowered = stem.lower()
    if "prologue" in lowered:
        return "Prologue"
    if len(parts) >= 2 and parts[0].isdigit():
        block = int(parts[0])
        if block == 0:
            return "Front Matter"
        # Story files currently use 01_* for the prologue arc.
        if block == 1:
            return "Prologue"
        return f"Act {block - 1}"
    return "Other"


def get_available_chapters() -> List[Dict[str, str]]:
    if not TEXTS_DIR.exists():
        return []
    chapters: List[Dict[str, str]] = []
    for path in sorted(TEXTS_DIR.glob("*.txt")):
        chapters.append(
            {
                "id": path.name,
                "label": _chapter_label_from_name(path.name),
                "category": _chapter_category_from_name(path.name),
            }
        )
    return chapters


def set_selected_chapters(chapter_ids: List[str] | None) -> None:
    global _SELECTED_CHAPTER_IDS
    available_ids = [c["id"] for c in get_available_chapters()]
    if not available_ids:
        _SELECTED_CHAPTER_IDS = []
        return
    requested = chapter_ids or available_ids
    selected = [cid for cid in available_ids if cid in requested]
    _SELECTED_CHAPTER_IDS = selected or available_ids


def get_selected_chapters() -> List[str]:
    if not _SELECTED_CHAPTER_IDS:
        set_selected_chapters(None)
    return list(_SELECTED_CHAPTER_IDS)


def get_selected_chapter_labels() -> List[str]:
    label_map = {c["id"]: c["label"] for c in get_available_chapters()}
    return [label_map[cid] for cid in get_selected_chapters() if cid in label_map]


set_selected_chapters(None)


def _load_seminar_text() -> str:
    override = os.environ.get("SEMINAR_TOPIC", "").strip()
    if override:
        return override
    selected_ids = get_selected_chapters()
    if TEXTS_DIR.exists() and selected_ids:
        chunks: List[str] = []
        label_map = {c["id"]: c["label"] for c in get_available_chapters()}
        for chapter_id in selected_ids:
            path = TEXTS_DIR / chapter_id
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            label = label_map.get(chapter_id, Path(chapter_id).stem)
            chunks.append(f"{label}\n\n{text}")
        if chunks:
            return "\n\n" + ("\n\n" + ("-" * 80) + "\n\n").join(chunks)
    for candidate in (PRELUDE_PATH, APP_PRELUDE_PATH):
        if candidate.exists():
            return candidate.read_text(encoding="utf-8", errors="ignore").strip()
    return "Prelude text is missing."


def _sanitize_agent_output(agent: str, content: str) -> str:
    cleaned = (content or "").strip()
    if not cleaned:
        return cleaned

    # Remove leading speaker labels if the model tries to impersonate another participant.
    for participant in [*AGENT_NAMES, "System"]:
        prefixes = [
            f"{participant}:",
            f"{participant} -",
            f"{participant} —",
        ]
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
                break

    # If the model explicitly claims to be another participant, nudge it back to content only.
    wrong_identity_markers = [
        f"as {name.lower()}" for name in AGENT_NAMES if name != agent
    ]
    if any(marker in cleaned.lower()[:80] for marker in wrong_identity_markers):
        cleaned = cleaned.replace("\n", " ").strip()

    return cleaned


def _trim_incomplete_tail(text: str) -> str:
    """Drop obvious trailing sentence fragments from truncated model outputs."""
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned
    if cleaned.endswith((".", "!", "?", "\"", "'", ")", "]")):
        return cleaned
    # Do not aggressively trim structured outputs (lists/math/multi-line); that can
    # drop closing punctuation and break downstream markdown/math rendering.
    if "\n" in cleaned or re.search(r"^\s*[-*]\s+", cleaned, flags=re.M) or "$" in cleaned:
        if re.search(r"[A-Za-z0-9]$", cleaned):
            return cleaned + "."
        return cleaned
    last_end = max(cleaned.rfind("."), cleaned.rfind("!"), cleaned.rfind("?"))
    if last_end >= 0:
        trimmed = cleaned[: last_end + 1].strip()
        if trimmed:
            return trimmed
    # If we only have a fragment, keep it readable instead of leaving a hanging tail.
    return cleaned + "."


def _count_sentences(text: str) -> int:
    return len(re.findall(r"[^.!?]+[.!?]", (text or "").strip()))


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", (text or "").strip()))


def _classify_turn_interest(agent: str, text: str) -> tuple[str | None, int]:
    lowered = text.lower()
    tokens = [token.strip(".,!?:;()[]{}\"'").lower() for token in text.split()]
    mentioned = agent.lower() in tokens

    invitation_phrases = (
        "what do you think",
        "do you agree",
        "thoughts",
        "anyone else",
        "who wants to respond",
        "can someone",
    )
    disagreement_markers = (
        "i disagree",
        "i'm not convinced",
        "counterpoint",
        "however",
        "but ",
    )
    analysis_markers = (
        "in the text",
        "evidence",
        "quote",
        "line",
        "imagery",
        "symbol",
        "tone",
        "motif",
        "theme",
    )

    has_question = "?" in text
    invited = any(phrase in lowered for phrase in invitation_phrases)
    disagreement = any(marker in lowered for marker in disagreement_markers)
    analysis = any(marker in lowered for marker in analysis_markers)

    if mentioned and (has_question or invited or disagreement or analysis):
        return "direct invitation", 9
    if has_question and (invited or analysis or disagreement):
        return "targeted seminar question", 7
    if disagreement and analysis:
        return "analytical disagreement", 6
    if analysis and len(text.split()) >= 18:
        return "textual claim", 4
    if has_question and len(text.split()) >= 14:
        return "open question", 3
    if random.random() < RANDOM_CHANCE and len(text.split()) >= 20:
        return "organic follow-up", 1
    return None, 0


def _load_key_file(filename: str) -> str:
    path = KEYS_PATH / filename
    if path.exists():
        return path.read_text().strip()
    return ""


def _bootstrap_llm_keys() -> None:
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        openrouter_key = _load_key_file("openkey.txt")
        if openrouter_key:
            os.environ["OPENROUTER_API_KEY"] = openrouter_key
    if os.environ.get("OPENROUTER_API_KEY", "").strip():
        return
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        openai_key = _load_key_file("gptkey.txt")
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        anthropic_key = _load_key_file("claudekey.txt")
        if anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        gemini_key = _load_key_file("geminikey.txt")
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key


def _get_openai_key() -> str:
    _bootstrap_llm_keys()
    return os.environ.get("OPENAI_API_KEY", "").strip()


def _get_anthropic_key() -> str:
    _bootstrap_llm_keys()
    return os.environ.get("ANTHROPIC_API_KEY", "").strip() or _load_key_file("claudekey.txt")


def _get_gemini_key() -> str:
    _bootstrap_llm_keys()
    return os.environ.get("GEMINI_API_KEY", "").strip() or _load_key_file("geminikey.txt")


def _get_openrouter_key() -> str:
    _bootstrap_llm_keys()
    return os.environ.get("OPENROUTER_API_KEY", "").strip() or _load_key_file("openkey.txt")


def _use_openrouter_for_llms() -> bool:
    return bool(_get_openrouter_key())


def _is_anthropic_agent(agent: str) -> bool:
    if _use_openrouter_for_llms():
        return False
    return agent in ANTHROPIC_AGENT_NAMES


def _is_gemini_agent(agent: str) -> bool:
    if _use_openrouter_for_llms():
        return False
    return agent in GEMINI_AGENT_NAMES


def _get_client_for_agent(agent: str):
    if _use_openrouter_for_llms():
        api_key = _get_openrouter_key()
        if not api_key:
            raise RuntimeError("OpenRouter key missing. Set OPENROUTER_API_KEY or add keys/openkey.txt.")
        return AsyncOpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": OPENROUTER_REFERER,
                "X-Title": OPENROUTER_TITLE,
            },
        )
    if agent in OPENROUTER_ONLY_AGENT_NAMES:
        raise RuntimeError(f"{agent} is configured as an OpenRouter-only model. Set OPENROUTER_API_KEY or add keys/openkey.txt.")
    if _is_gemini_agent(agent):
        return None
    if _is_anthropic_agent(agent):
        if AsyncAnthropic is None:
            raise RuntimeError("Anthropic SDK not installed. Run: pip install anthropic")
        api_key = _get_anthropic_key()
        if not api_key:
            raise RuntimeError("No LLM key found. Prefer OPENROUTER_API_KEY or keys/openkey.txt; Anthropic direct keys still work as fallback.")
        return AsyncAnthropic(api_key=api_key)
    api_key = _get_openai_key()
    if not api_key:
        raise RuntimeError("No LLM key found. Prefer OPENROUTER_API_KEY or keys/openkey.txt; OpenAI direct keys still work as fallback.")
    return AsyncOpenAI(api_key=api_key)


def _gemini_post_json(api_key: str, path: str, body: dict, timeout_s: float) -> dict:
    encoded_key = urllib.parse.quote(api_key, safe="")
    url = f"https://generativelanguage.googleapis.com/v1beta/{path}?key={encoded_key}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=max(10.0, timeout_s)) as resp_obj:
            return json.loads(resp_obj.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Gemini API HTTP {exc.code}: {body_text}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini API connection error: {exc}") from exc


def _gemini_make_cache_key(model: str, system_instruction: str, cached_text: str) -> str:
    digest = hashlib.sha256((model + "\n" + system_instruction + "\n" + cached_text).encode("utf-8")).hexdigest()
    return f"{model}:{digest}"


def _gemini_get_or_create_cache(
    api_key: str,
    model: str,
    system_instruction: str,
    cached_text: str,
    ttl: str = "3600s",
    timeout_s: float = 20.0,
) -> str | None:
    key = _gemini_make_cache_key(model, system_instruction, cached_text)
    existing = _GEMINI_CACHE_BY_KEY.get(key)
    if existing:
        return existing
    body = {
        "model": f"models/{model}",
        "systemInstruction": {
            "role": "user",
            "parts": [{"text": system_instruction}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": cached_text}],
            }
        ],
        "ttl": ttl,
    }
    try:
        created = _gemini_post_json(api_key=api_key, path="cachedContents", body=body, timeout_s=timeout_s)
        cache_name = (created or {}).get("name")
        if cache_name:
            _GEMINI_CACHE_BY_KEY[key] = cache_name
            LOGGER.info("Gemini explicit cache created for %s", model)
            return cache_name
    except Exception as exc:
        LOGGER.info("Gemini explicit cache unavailable for %s: %s", model, exc)
    return None




def _is_cache_feature_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    cache_markers = (
        "cache_control",
        "prompt-caching",
        "prompt_cache",
        "unknown parameter",
        "extra headers",
    )
    return any(marker in msg for marker in cache_markers)


def _log_cache_usage(provider: str, agent: str, resp) -> None:
    try:
        if provider == "openai":
            usage = getattr(resp, "usage", None)
            details = getattr(usage, "prompt_tokens_details", None) if usage else None
            cached = getattr(details, "cached_tokens", None) if details else None
            if cached is not None:
                LOGGER.info("[cache] %s %s: cached_prompt_tokens=%s", provider, agent, cached)
        elif provider == "anthropic":
            usage = getattr(resp, "usage", None)
            cache_read = getattr(usage, "cache_read_input_tokens", None) if usage else None
            cache_write = getattr(usage, "cache_creation_input_tokens", None) if usage else None
            if cache_read is not None or cache_write is not None:
                LOGGER.info(
                    "[cache] %s %s: cache_read_input_tokens=%s, cache_creation_input_tokens=%s",
                    provider,
                    agent,
                    cache_read or 0,
                    cache_write or 0,
                )
    except Exception:
        pass


def _extract_json_object(text: str) -> str:
    raw = (text or "").strip()
    if raw.startswith("{") and raw.endswith("}"):
        return raw
    match = re.search(r"\{[\s\S]*\}", raw)
    return match.group(0) if match else raw


def _parse_data_url_image(data_url: str | None) -> tuple[str, str] | None:
    raw = (data_url or "").strip()
    if not raw.startswith("data:") or ";base64," not in raw:
        return None
    header, b64 = raw.split(",", 1)
    mime = header[5:].split(";")[0].strip().lower() or "image/png"
    if not mime.startswith("image/") or not b64:
        return None
    return mime, b64.strip()


def _sandbox_extract_json(text: str) -> dict | None:
    raw = _extract_json_object(text or "")
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # Lenient fallback for slightly broken JSON-like output.
    low = raw.lower()
    agree_match = re.search(r'"agree"\s*:\s*(true|false)', low) or re.search(r"\bagree\b\s*[:=]\s*(true|false)", low)
    if not agree_match:
        return None
    out: dict[str, Any] = {"agree": agree_match.group(1) == "true"}
    ans_match = re.search(r'"answer"\s*:\s*"([\s\S]*?)"', raw)
    if ans_match:
        out["answer"] = ans_match.group(1).strip()
    return out


def _normalize_sandbox_text(text: str) -> str:
    s = (text or "").strip()
    # Normalize escaped newlines/tabs only when they are actual escapes, not TeX commands
    # like \nabla, \times, or \text.
    s = re.sub(r"\\n(?![A-Za-z])", "\n", s)
    s = re.sub(r"\\t(?![A-Za-z])", "\t", s)
    # Collapse excessive blank lines for readability.
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _json_from_model_text(raw: str) -> dict:
    try:
        obj = json.loads(_extract_json_object(raw or ""))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _as_items(value: Any) -> List[dict]:
    out: List[dict] = []
    if isinstance(value, list):
        for it in value:
            if isinstance(it, dict):
                out.append(dict(it))
            elif isinstance(it, str) and it.strip():
                out.append({"content": it.strip()})
    return out


def _state_default() -> Dict[str, Any]:
    return {
        "memory_version": 1,
        "chapters": [],
        "canon_facts": [],
        "character_states": [],
        "open_threads": [],
        "resolved_threads": [],
        "theme_map": [],
        "audience_knowledge": [],
        "intentional_ambiguities": [],
        "interpretations": [],
        "continuity_risks": [],
        "turn_summaries": [],
    }


def _append_capped(items: List[dict], new_items: List[dict], cap: int) -> List[dict]:
    merged = list(items)
    for it in new_items:
        content = str(it.get("content") or "").strip()
        if not content:
            continue
        if any(str(x.get("content") or "").strip() == content for x in merged):
            continue
        merged.append(it)
    return merged[-cap:]


async def build_story_state_update(
    previous_state: Dict[str, Any],
    chapter_ids: List[str],
    turn_id: int,
    new_story_text: str,
    discussion_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Writer (Sonnet) creates a structured patch; reviewer (Pro) validates it."""
    prev = dict(_state_default())
    if isinstance(previous_state, dict):
        prev.update(previous_state)
    discussion_text = "\n".join(
        [f"{d.get('agent_id')}: {str(d.get('text') or '').strip()}" for d in discussion_entries[-12:]]
    )
    writer_system = (
        "You maintain narrative memory state. Output STRICT JSON only.\n"
        "Produce a patch with only changed items using this schema:\n"
        "{"
        "\"canon_updates\":{\"add\":[],\"modify\":[],\"deprecate\":[]},"
        "\"character_updates\":[],"
        "\"thread_updates\":{\"new\":[],\"resolved\":[]},"
        "\"theme_updates\":[],"
        "\"audience_knowledge_updates\":[],"
        "\"ambiguities_to_preserve\":[],"
        "\"interpretation_updates\":[],"
        "\"continuity_risks\":[]"
        "}\n"
        "Each item should be an object with: content, evidence, confidence (0-1), source_type (story_text|discussion)."
    )
    writer_user = (
        f"Turn id: {turn_id}\n"
        f"Chapters in scope: {', '.join(chapter_ids)}\n\n"
        f"Previous state JSON:\n{json.dumps(prev, ensure_ascii=False)[:7000]}\n\n"
        f"New story text this turn:\n{new_story_text}\n\n"
        f"Panel discussion this turn:\n{discussion_text or '(none)'}\n\n"
        "Return JSON patch only."
    )
    writer_raw = await _sandbox_generate_text(
        agent="Sonnet",
        system_prompt=writer_system,
        user_text=writer_user,
        timeout_s=80.0,
        temperature=0.1,
        max_tokens=2000,
    )
    patch = _json_from_model_text(writer_raw)
    if not patch:
        patch = {
            "canon_updates": {"add": [], "modify": [], "deprecate": []},
            "character_updates": [],
            "thread_updates": {"new": [], "resolved": []},
            "theme_updates": [],
            "audience_knowledge_updates": [],
            "ambiguities_to_preserve": [],
            "interpretation_updates": [],
            "continuity_risks": [],
        }

    reviewer_system = (
        "You are a strict patch reviewer. Validate each patch item against evidence.\n"
        "Output STRICT JSON with:\n"
        "{"
        "\"reviews\":[{\"path\":\"canon_updates.add\",\"index\":0,\"status\":\"supported|weakly_supported|unsupported|speculative|contradicts\",\"reason\":\"...\"}],"
        "\"missed_updates\":[],"
        "\"notes\":[]"
        "}\n"
        "Prefer story text over discussion; discussion-only claims should not become canon facts unless explicit in text."
    )
    reviewer_user = (
        f"Turn id: {turn_id}\n"
        f"New story text:\n{new_story_text}\n\n"
        f"Discussion:\n{discussion_text or '(none)'}\n\n"
        f"Patch to review:\n{json.dumps(patch, ensure_ascii=False)}"
    )
    reviewer_raw = await _sandbox_generate_text(
        agent="Pro",
        system_prompt=reviewer_system,
        user_text=reviewer_user,
        timeout_s=80.0,
        temperature=0.0,
        max_tokens=1800,
    )
    review = _json_from_model_text(reviewer_raw)
    labels: Dict[tuple[str, int], str] = {}
    for r in review.get("reviews", []) if isinstance(review, dict) else []:
        try:
            labels[(str(r.get("path")), int(r.get("index")))] = str(r.get("status") or "").lower()
        except Exception:
            continue

    def filter_items(path: str, items: List[dict], allow_weak: bool = True) -> List[dict]:
        accepted: List[dict] = []
        for i, it in enumerate(items):
            status = labels.get((path, i), "supported")
            if status in {"unsupported", "contradicts"}:
                continue
            if status == "speculative" and path.startswith("canon_updates"):
                continue
            if (not allow_weak) and status == "weakly_supported":
                continue
            obj = dict(it)
            obj["validation"] = status
            obj["source_turn_id"] = turn_id
            obj["last_confirmed_turn"] = turn_id
            accepted.append(obj)
        return accepted

    canon = prev.get("canon_facts", [])
    canon_add = filter_items("canon_updates.add", _as_items((patch.get("canon_updates") or {}).get("add", [])))
    canon_mod = filter_items("canon_updates.modify", _as_items((patch.get("canon_updates") or {}).get("modify", [])))
    canon = _append_capped(canon, canon_add + canon_mod, 80)

    deprecated = _append_capped(prev.get("deprecated_facts", []), filter_items("canon_updates.deprecate", _as_items((patch.get("canon_updates") or {}).get("deprecate", []))), 30)
    characters = _append_capped(prev.get("character_states", []), filter_items("character_updates", _as_items(patch.get("character_updates", []))), 60)
    threads = _append_capped(prev.get("open_threads", []), filter_items("thread_updates.new", _as_items((patch.get("thread_updates") or {}).get("new", []))), 60)
    resolved = _append_capped(prev.get("resolved_threads", []), filter_items("thread_updates.resolved", _as_items((patch.get("thread_updates") or {}).get("resolved", []))), 60)
    themes = _append_capped(prev.get("theme_map", []), filter_items("theme_updates", _as_items(patch.get("theme_updates", []))), 40)
    audience = _append_capped(prev.get("audience_knowledge", []), filter_items("audience_knowledge_updates", _as_items(patch.get("audience_knowledge_updates", []))), 50)
    ambiguities = _append_capped(prev.get("intentional_ambiguities", []), filter_items("ambiguities_to_preserve", _as_items(patch.get("ambiguities_to_preserve", [])), allow_weak=True), 50)
    interpretations = _append_capped(prev.get("interpretations", []), filter_items("interpretation_updates", _as_items(patch.get("interpretation_updates", [])), allow_weak=True), 60)
    risks = _append_capped(prev.get("continuity_risks", []), filter_items("continuity_risks", _as_items(patch.get("continuity_risks", [])), allow_weak=True), 50)

    summaries = list(prev.get("turn_summaries", []))
    summaries.append(
        {
            "turn_id": turn_id,
            "summary": _truncate_for_packet("\n".join([new_story_text, discussion_text]), 1200),
            "writer": "Sonnet",
            "reviewer": "Pro",
        }
    )
    summaries = summaries[-12:]

    next_state = dict(_state_default())
    next_state.update(prev)
    next_state.update(
        {
            "memory_version": int(prev.get("memory_version", 1)) + 1,
            "chapters": list(dict.fromkeys([*(prev.get("chapters", []) or []), *chapter_ids]))[-24:],
            "canon_facts": canon,
            "deprecated_facts": deprecated,
            "character_states": characters,
            "open_threads": threads,
            "resolved_threads": resolved,
            "theme_map": themes,
            "audience_knowledge": audience,
            "intentional_ambiguities": ambiguities,
            "interpretations": interpretations,
            "continuity_risks": risks,
            "turn_summaries": summaries,
            "last_patch": patch,
            "last_review": review if isinstance(review, dict) else {},
            "updated_turn_id": turn_id,
            "updated_ts": time.time(),
        }
    )

    turn_summary = _truncate_for_packet(
        f"Turn {turn_id}: {len(canon_add)} canon adds, {len(characters)} character-state items tracked, "
        f"{len(threads)} open threads, {len(resolved)} resolved threads.",
        400,
    )
    return {"story_state": next_state, "turn_summary": turn_summary, "patch": patch, "review": review}


async def build_agent_state_updates(
    previous_agent_states: Dict[str, Dict[str, Any]],
    agents_list: List[str],
    turn_id: int,
    new_story_text: str,
    discussion_entries: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Generate compact, insight-first per-agent states (not transcript copies)."""
    if not agents_list:
        return {}
    prev = {str(k): (v if isinstance(v, dict) else {}) for k, v in (previous_agent_states or {}).items()}
    discussion_text = "\n".join(
        [f"{d.get('agent_id')}: {str(d.get('text') or '').strip()}" for d in discussion_entries[-16:]]
    )
    system_prompt = (
        "You maintain per-agent reasoning states for a discussion panel. Output STRICT JSON only.\n"
        "Do NOT copy transcript lines. Paraphrase and abstract.\n"
        "For each agent, produce a personal, specific, insightful state with this schema:\n"
        "{"
        "\"agents\":{"
        "\"AgentName\":{"
        "\"stance\":\"agree|disagree|mixed|uncertain\","
        "\"core_belief\":\"one sentence\","
        "\"reasoning_style\":\"one sentence about how they reason\","
        "\"confidence\":0.0,"
        "\"blind_spot\":\"one sentence\","
        "\"next_move\":\"one sentence on what they should do next\","
        "\"open_question\":\"one sentence\","
        "\"relations\":[\"names of agents they align/conflict with\"]"
        "}"
        "}"
        "}\n"
        "Keep each field concise and high-signal."
    )
    user_prompt = (
        f"Turn: {turn_id}\n"
        f"Agents: {', '.join(agents_list)}\n\n"
        f"Previous agent states JSON:\n{json.dumps(prev, ensure_ascii=False)[:7000]}\n\n"
        f"New story text this turn:\n{new_story_text}\n\n"
        f"Discussion this turn:\n{discussion_text or '(none)'}\n\n"
        "Return strict JSON only."
    )
    raw = await _sandbox_generate_text(
        agent="Sonnet",
        system_prompt=system_prompt,
        user_text=user_prompt,
        timeout_s=80.0,
        temperature=0.1,
        max_tokens=2000,
    )
    obj = _json_from_model_text(raw)
    generated = obj.get("agents", {}) if isinstance(obj, dict) else {}
    out: Dict[str, Dict[str, Any]] = {}
    for name in agents_list:
        item = generated.get(name, {}) if isinstance(generated, dict) else {}
        if not isinstance(item, dict):
            item = {}
        confidence_raw = item.get("confidence", 0.5)
        try:
            confidence = float(confidence_raw)
        except Exception:
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        state = {
            "memory_version": int((prev.get(name) or {}).get("memory_version", 0)) + 1,
            "agent": name,
            "stance": str(item.get("stance") or "uncertain")[:24],
            "core_belief": _truncate_for_packet(str(item.get("core_belief") or ""), 220),
            "reasoning_style": _truncate_for_packet(str(item.get("reasoning_style") or ""), 220),
            "confidence": confidence,
            "blind_spot": _truncate_for_packet(str(item.get("blind_spot") or ""), 220),
            "next_move": _truncate_for_packet(str(item.get("next_move") or ""), 220),
            "open_question": _truncate_for_packet(str(item.get("open_question") or ""), 220),
            "relations": [r for r in (item.get("relations") or []) if isinstance(r, str)][:8],
            "source_turn_id": turn_id,
            "updated_ts": time.time(),
        }
        # lightweight fallback if model skipped fields
        if not state["core_belief"]:
            state["core_belief"] = _truncate_for_packet((prev.get(name) or {}).get("core_belief", ""), 220)
        if not state["reasoning_style"]:
            state["reasoning_style"] = _truncate_for_packet((prev.get(name) or {}).get("reasoning_style", ""), 220)
        out[name] = state
    return out


def _looks_sentence_complete(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    return bool(re.search(r'[.!?]["\')\]]*\s*$', s))


async def _sandbox_generate_text(
    agent: str,
    system_prompt: str,
    user_text: str,
    image_data_url: str | None = None,
    timeout_s: float = 60.0,
    temperature: float = 0.3,
    max_tokens: int = 800,
) -> str:
    while _paused:
        await asyncio.sleep(0.2)
    model = _agent_model(agent)
    client = _get_client_for_agent(agent)
    parsed_image = _parse_data_url_image(image_data_url)

    if _is_anthropic_agent(agent):
        content = [{"type": "text", "text": user_text}]
        if parsed_image:
            mime, b64 = parsed_image
            content.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}})
        resp = await asyncio.wait_for(
            client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": content}],
            ),
            timeout=timeout_s,
        )
        return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text").strip()

    if _is_gemini_agent(agent):
        api_key = _get_gemini_key()
        if not api_key:
            raise RuntimeError("No LLM key found. Prefer OPENROUTER_API_KEY or keys/openkey.txt; Gemini direct keys still work as fallback.")
        # Give Flash extra headroom; it occasionally returns empty output near limits.
        if agent == "Flash":
            timeout_s = max(timeout_s, 95.0)
            max_tokens = max(max_tokens, 1200)
        elif agent == "Pro":
            # Pro is often asked for short-but-complete explanations; give enough budget to avoid truncation.
            timeout_s = max(timeout_s, 100.0)
            max_tokens = max(max_tokens, 1400)
        parts = [{"text": user_text}]
        if parsed_image:
            mime, b64 = parsed_image
            parts.append({"inline_data": {"mime_type": mime, "data": b64}})
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                **({"thinkingConfig": {"thinkingBudget": 128}} if agent == "Pro" else {}),
            },
        }

        def _call_gemini() -> str:
            path = f"models/{urllib.parse.quote(model, safe='')}:generateContent"
            obj = _gemini_post_json(api_key=api_key, path=path, body=payload, timeout_s=timeout_s)
            candidates = obj.get("candidates") or []
            def _extract_text(cands: list) -> tuple[str, str | None]:
                if not cands:
                    return "", None
                first = cands[0] or {}
                finish = first.get("finishReason")
                parts_out = (((first.get("content") or {}).get("parts") or []))
                text_out = "".join((p.get("text") or "") for p in parts_out).strip()
                return text_out, finish

            text_out, finish = _extract_text(candidates)
            needs_retry = False
            if finish == "MAX_TOKENS":
                needs_retry = True
            if agent == "Pro" and text_out and not _looks_sentence_complete(text_out):
                needs_retry = True
            if text_out and not needs_retry:
                return text_out

            # One retry with larger cap if empty or likely truncated.
            retry_payload = dict(payload)
            retry_cfg = dict(payload.get("generationConfig") or {})
            retry_cfg["maxOutputTokens"] = int(max(max_tokens * 1.5, 1400))
            if agent == "Pro":
                retry_cfg["thinkingConfig"] = {"thinkingBudget": 128}
            retry_payload["generationConfig"] = retry_cfg
            obj2 = _gemini_post_json(api_key=api_key, path=path, body=retry_payload, timeout_s=timeout_s)
            text2, finish2 = _extract_text(obj2.get("candidates") or [])
            if text2 and (finish2 != "MAX_TOKENS" or _looks_sentence_complete(text2)):
                return text2
            # Prefer non-empty fallback from first try instead of returning empty placeholder.
            if text_out:
                LOGGER.warning("Gemini sandbox text likely truncated for %s (finishReason=%s); using first pass text", agent, finish)
                return text_out
            LOGGER.warning("Gemini returned empty sandbox text for %s (finishReason=%s)", agent, finish)
            return text2 or ""

        return await asyncio.wait_for(asyncio.to_thread(_call_gemini), timeout=timeout_s)

    user_content: Any
    if parsed_image:
        user_content = [{"type": "text", "text": user_text}, {"type": "image_url", "image_url": {"url": image_data_url}}]
    else:
        user_content = user_text
    resp = await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
        ),
        timeout=timeout_s,
    )
    return (resp.choices[0].message.content or "").strip()


async def run_sandbox_cycle(
    user_prompt: str,
    chain: List[str],
    image_data_url: str | None = None,
    status_cb: Callable[[str, str], Awaitable[None]] | None = None,
    answer_cb: Callable[[str, str], Awaitable[int | None]] | None = None,
    agree_cb: Callable[[str, int], Awaitable[None]] | None = None,
) -> Dict[str, Any]:
    prompt = (user_prompt or "").strip()
    if not prompt and not image_data_url:
        return {"answers": [], "participants": [], "poll": None, "votes": {}}
    if not prompt and image_data_url:
        prompt = "Please analyze this image."
    stack = [a for a in chain if a in AGENT_NAMES]
    if not stack:
        return {"answers": [], "participants": [], "poll": None, "votes": {}}

    answers: List[Dict[str, str]] = []
    participants: List[str] = []

    def _sandbox_style_note(name: str) -> str:
        if name in {"Pro", "Flash"}:
            return (
                "Be concise but complete: at least 3 full sentences with a short explanation and clear conclusion. "
                "No oversized step-by-step dumps."
            )
        # Keep Claude-family outputs concise and classroom-like.
        if _is_anthropic_agent(name):
            return (
                "Be concise: 2-4 sentences, about 40-90 words. "
                "No section headers like 'Analysis', 'Key Physics Principle', or 'Answer'. "
                "Give a short explanation plus the final choice."
            )
        return (
            "Be concise and readable. Avoid unnecessary section headers; prefer a short explanation and clear conclusion."
        )

    def _mark_participant(name: str) -> None:
        if name not in participants:
            participants.append(name)

    def _answer_block() -> str:
        if not answers:
            return "No prior answers yet."
        return "\n\n".join([f"{a['agent']}: {a['answer']}" for a in answers])

    async def _ensure_min_sentences(agent_name: str, text: str, context_prompt: str, min_sentences: int = 3) -> str:
        cleaned = _normalize_sandbox_text(text)
        if _count_sentences(cleaned) >= min_sentences:
            return cleaned
        retry = await _sandbox_generate_text(
            agent=agent_name,
            system_prompt=(
                f"You are {agent_name}. Rewrite the answer as at least {min_sentences} complete sentences. "
                "Keep it concise, readable, and consistent with the original conclusion."
            ),
            user_text=context_prompt + f"\n\nDraft to rewrite:\n{cleaned}",
            image_data_url=image_data_url,
            timeout_s=45.0,
            temperature=0.2,
            max_tokens=1000,
        )
        retried = _normalize_sandbox_text(retry)
        return retried if _count_sentences(retried) >= min_sentences else cleaned

    async def _review_decision(reviewer: str, review_prompt: str, timeout_s: float = 70.0) -> dict:
        raw = await _sandbox_generate_text(
            agent=reviewer,
            system_prompt=(
                f"You are {reviewer} in Sandbox mode. "
                "Return strict JSON only. You may change your mind based on new turns. "
                "Prefer agree when the current best answer is substantially correct. "
                "Disagree only for meaningful correction. "
                "Markdown and LaTeX are allowed inside the JSON answer string when useful. "
                "Prioritize readability: if you use symbols like f_0, write them clearly as LaTeX (example: $f_0$)."
                f" {_sandbox_style_note(reviewer)}"
            ),
            user_text=review_prompt,
            image_data_url=image_data_url,
            timeout_s=timeout_s,
            temperature=0.2,
            max_tokens=700,
        )
        parsed = _sandbox_extract_json(raw) or {}
        if "agree" in parsed:
            return parsed
        retry = await _sandbox_generate_text(
            agent=reviewer,
            system_prompt=(
                "Return ONLY JSON exactly: {\"agree\": true} OR {\"agree\": false, \"answer\": \"...\"}. "
                "No prose outside JSON and no extra keys. Plain JSON only. "
                "Inside \"answer\", keep formatting maximally readable and render symbols clearly."
                f" {_sandbox_style_note(reviewer)}"
            ),
            user_text=review_prompt,
            image_data_url=image_data_url,
            timeout_s=min(timeout_s, 40.0),
            temperature=0.0,
            max_tokens=500,
        )
        parsed_retry = _sandbox_extract_json(retry) or {}
        if "agree" in parsed_retry:
            return parsed_retry
        return {"agree": False, "answer": retry.strip() or raw.strip()}

    async def _publish_consensus_summary(final_agent: str, final_text: str) -> None:
        if status_cb:
            await status_cb(final_agent, "concluding")
        brief = await _sandbox_generate_text(
            agent=final_agent,
            system_prompt=(
                f"You are {final_agent}. Write exactly one sentence, 20 words max, summarizing the consensus reasoning. "
                "Plain text only."
            ),
            user_text=(
                f"Problem:\n{prompt}\n\n"
                f"Consensus answer details:\n{final_text}\n\n"
                "Return only the one-sentence summary."
            ),
            image_data_url=image_data_url,
            timeout_s=25.0,
            temperature=0.1,
            max_tokens=80,
        )
        brief = " ".join((brief or "").strip().split())
        if brief:
            brief = brief.splitlines()[0].strip()
            # Hard cap at 20 words to avoid long repeated conclusions.
            words = brief.split()
            if len(words) > 20:
                brief = " ".join(words[:20]).rstrip(".,;:!?") + "."
            # If the line looks truncated, regenerate once with tighter constraints.
            if re.search(r"\b(from|to|of|for|with|at|because|that|which|when)$", brief.lower()):
                retry_brief = await _sandbox_generate_text(
                    agent=final_agent,
                    system_prompt=(
                        f"You are {final_agent}. Return exactly one complete sentence (10-20 words) summarizing the consensus. "
                        "Do not end mid-phrase."
                    ),
                    user_text=f"Problem:\n{prompt}\n\nConsensus answer:\n{final_text}\n\nOne complete sentence only.",
                    image_data_url=image_data_url,
                    timeout_s=20.0,
                    temperature=0.0,
                    max_tokens=80,
                )
                retry_brief = " ".join((retry_brief or "").strip().split())
                if retry_brief:
                    brief = retry_brief.splitlines()[0].strip()
            if len(brief.split()) < 6:
                brief = ""
            if brief and not re.search(r"[.!?]$", brief):
                brief = ""
        if not brief:
            # Deterministic fallback: synthesize one complete short sentence from the consensus text.
            fb = _normalize_sandbox_text(final_text or "")
            first_sentence = re.split(r"(?<=[.!?])\s+", fb)[0].strip() if fb else ""
            if not first_sentence:
                words = " ".join((fb or "").split()).split()
                first_sentence = " ".join(words[:20]).strip()
                if first_sentence and not re.search(r"[.!?]$", first_sentence):
                    first_sentence = first_sentence.rstrip(".,;:!?") + "."
            brief = first_sentence or "Consensus reached."
        brief = _normalize_sandbox_text(brief)
        concise = await _sandbox_generate_text(
            agent=final_agent,
            system_prompt=(
                f"You are {final_agent}. Return only the final answer line exactly (examples: D, B and C, or 56). "
                "No explanation."
            ),
            user_text=f"Problem:\n{prompt}\n\nFinal answer:\n{final_text}\n\nReturn concise final answer line only.",
            image_data_url=image_data_url,
            timeout_s=35.0,
            temperature=0.0,
            max_tokens=180,
        )
        concise = (concise or "").strip().splitlines()[0].strip()
        if not concise:
            concise = final_text[:60].strip()
        concise = _normalize_sandbox_text(concise)
        summary = f"{brief}\n\n**Final answer:** {concise}"
        if answer_cb:
            await answer_cb(final_agent, summary)

    # First answer
    first = stack[0]
    if status_cb:
        await status_cb(first, "thinking")
    first_text = await _sandbox_generate_text(
        agent=first,
        system_prompt=(
            f"You are {first} in Sandbox mode. Give a direct answer in plain text like a student in class. "
            "Markdown and LaTeX are allowed when helpful. "
            "Prioritize readability: if you use symbols like f_0, write them clearly as LaTeX (example: $f_0$)."
            f" {_sandbox_style_note(first)}"
        ),
        user_text=f"User problem:\n{prompt}\n\n{'MANDATORY: write at least 3 complete sentences.' if first in {'Pro', 'Flash'} else ''}",
        image_data_url=image_data_url,
        timeout_s=90.0,
        temperature=0.3,
        max_tokens=900,
    )
    first_json = _sandbox_extract_json(first_text)
    if isinstance(first_json, dict) and first_json.get("answer"):
        first_text = str(first_json.get("answer") or "").strip()
    first_text = _normalize_sandbox_text(_trim_incomplete_tail(_sanitize_agent_output(first, first_text)))
    if first in {"Pro", "Flash"}:
        first_text = await _ensure_min_sentences(first, first_text, f"User problem:\n{prompt}", min_sentences=3)
    if not first_text:
        first_text = "(No response returned. Please ignore this answer.)"
    answers.append({"agent": first, "answer": first_text})
    _mark_participant(first)
    if answer_cb:
        current_best_message_id = await answer_cb(first, first_text)
    if status_cb:
        await status_cb(first, "participating")

    # Review chain with "two extra agreeers" after first disagreement.
    agreed = False
    disagreement_seen = False
    first_response_consensus_count = 0
    disagreement_agreeers: List[str] = []
    current_best_agent = first
    current_best_answer = first_text
    current_best_message_id: int | None = None
    for agent in stack[1:]:
        if status_cb:
            await status_cb(agent, "thinking")
        hard_sentence_line = "MANDATORY: write at least 3 complete sentences." if agent in {"Pro", "Flash"} else ""
        review_prompt = (
            f"User problem:\n{prompt}\n\nCurrent proposed answers:\n{_answer_block()}\n\n"
            "Decide whether you agree with the current best answer. "
            "Return JSON exactly: {\"agree\": true} OR {\"agree\": false, \"answer\": \"...\"}."
            f"\n\n{hard_sentence_line}"
        )
        parsed = await _review_decision(agent, review_prompt, timeout_s=70.0)
        if parsed.get("agree") is True:
            _mark_participant(agent)
            if status_cb:
                await status_cb(agent, "agree")
            if agree_cb:
                await agree_cb(agent, int(current_best_message_id or 0))
            if (not disagreement_seen) and len(answers) == 1:
                first_response_consensus_count += 1
            else:
                first_response_consensus_count = 0
            if disagreement_seen and agent not in disagreement_agreeers:
                disagreement_agreeers.append(agent)
                if len(disagreement_agreeers) >= 2:
                    break
            if first_response_consensus_count >= 2:
                agreed = True
                break
            continue

        disagreement_seen = True
        first_response_consensus_count = 0
        disagreement_agreeers = []
        alt = str(parsed.get("answer") or "").strip()
        alt = _normalize_sandbox_text(alt)
        if agent in {"Pro", "Flash"}:
            alt = await _ensure_min_sentences(
                agent,
                alt,
                f"User problem:\n{prompt}\n\nCurrent proposed answers:\n{_answer_block()}",
                min_sentences=3,
            )
        if not alt:
            alt = "(No response returned. Please ignore this answer.)"
        answers.append({"agent": agent, "answer": alt})
        _mark_participant(agent)
        current_best_agent = agent
        current_best_answer = alt
        if answer_cb:
            current_best_message_id = await answer_cb(agent, alt)
        if status_cb:
            await status_cb(agent, "participating")

    if agreed and (not disagreement_seen) and len(answers) == 1:
        await _publish_consensus_summary(current_best_agent, current_best_answer)
        return {"answers": answers, "participants": participants, "poll": None, "votes": {}, "ended_by_agreement": True}

    if disagreement_seen and len(stack) > 1:
        speaker_set = {a["agent"] for a in answers}
        seed_set = set(speaker_set) | set(disagreement_agreeers[:2])
        active_models = [name for name in stack if name in seed_set]
        if len(active_models) < 2:
            active_models = [first, current_best_agent] if current_best_agent != first else [first]
        agreement_streak = 0
        max_discussion_turns = max(12, len(stack) * 6)
        discussion_turns = 0
        stack_cursor = (stack.index(current_best_agent) + 1) % len(stack)
        final_agreeing_agent = None
        agreed_on_latest_idx: dict[str, int] = {}

        def _next_active_reviewer() -> str | None:
            nonlocal stack_cursor
            for _ in range(len(stack)):
                candidate = stack[stack_cursor]
                stack_cursor = (stack_cursor + 1) % len(stack)
                if candidate in active_models:
                    return candidate
            return None

        while discussion_turns < max_discussion_turns:
            required_agrees = max(1, len(active_models) - 1)
            if agreement_streak >= required_agrees:
                break
            reviewer = _next_active_reviewer()
            if reviewer is None:
                break
            if reviewer == current_best_agent:
                continue
            latest_idx = len(answers) - 1
            if agreed_on_latest_idx.get(reviewer) == latest_idx:
                # This reviewer already agreed with the current latest answer; skip API call.
                _mark_participant(reviewer)
                agreement_streak += 1
                final_agreeing_agent = reviewer
                if status_cb:
                    await status_cb(reviewer, "agree")
                if agree_cb:
                    await agree_cb(reviewer, int(current_best_message_id or 0))
                continue
            discussion_turns += 1
            if status_cb:
                await status_cb(reviewer, "thinking")
            if len(active_models) < len(stack):
                opinion_line = (
                    "If you need another model's opinion, append exactly this sentence in your answer: Get another opinion."
                )
            else:
                opinion_line = f"All {len(stack)} models are already involved. Do not request another opinion."
            discussion_prompt = (
                f"User problem:\n{prompt}\n\nCurrent best answer ({current_best_agent}):\n{current_best_answer}\n\n"
                f"Currently involved models: {', '.join(active_models)}\n\nDiscussion so far:\n{_answer_block()}\n\n"
                "You may change your mind from earlier turns. "
                "Return JSON exactly: {\"agree\": true} OR {\"agree\": false, \"answer\": \"...\"}. "
                f"{opinion_line}\n\n"
                f"{'MANDATORY: write at least 3 complete sentences.' if reviewer in {'Pro', 'Flash'} else ''}"
            )
            parsed = await _review_decision(reviewer, discussion_prompt, timeout_s=70.0)
            if parsed.get("agree") is True:
                _mark_participant(reviewer)
                agreement_streak += 1
                final_agreeing_agent = reviewer
                agreed_on_latest_idx[reviewer] = len(answers) - 1
                if status_cb:
                    await status_cb(reviewer, "agree")
                if agree_cb:
                    await agree_cb(reviewer, int(current_best_message_id or 0))
                continue
            alt = str(parsed.get("answer") or "").strip()
            alt = _normalize_sandbox_text(alt)
            if reviewer in {"Pro", "Flash"}:
                alt = await _ensure_min_sentences(
                    reviewer,
                    alt,
                    f"User problem:\n{prompt}\n\nCurrent best answer ({current_best_agent}):\n{current_best_answer}",
                    min_sentences=3,
                )
            if not alt:
                alt = "(No response returned. Please ignore this answer.)"
            request_more = "get another opinion." in alt.lower()
            if request_more:
                alt = re.sub(r"(?i)\bget another opinion\.\s*", "", alt).strip()
            if not alt:
                alt = "(No response returned. Please ignore this answer.)"
            answers.append({"agent": reviewer, "answer": alt})
            _mark_participant(reviewer)
            current_best_agent = reviewer
            current_best_answer = alt
            agreement_streak = 0
            agreed_on_latest_idx = {}
            if answer_cb:
                current_best_message_id = await answer_cb(reviewer, alt)
            if status_cb:
                await status_cb(reviewer, "participating")
            if request_more and len(active_models) < len(stack):
                start_idx = (stack.index(reviewer) + 1) % len(stack)
                for off in range(len(stack)):
                    cand = stack[(start_idx + off) % len(stack)]
                    if cand not in active_models:
                        active_models = [name for name in stack if name in (set(active_models) | {cand})]
                        _mark_participant(cand)
                        break

        if agreement_streak >= max(1, len(active_models) - 1):
            await _publish_consensus_summary(final_agreeing_agent or current_best_agent, current_best_answer)
            return {
                "answers": answers,
                "participants": participants,
                "poll": None,
                "votes": {},
                "ended_by_agreement": True,
                "consensus_reached_after_discussion": True,
            }

    await _publish_consensus_summary(current_best_agent, current_best_answer)
    return {"answers": answers, "participants": participants, "poll": None, "votes": {}, "ended_by_agreement": False}


async def _send_message(sender: str, content: str, broadcast_cb) -> None:
    await db.insert_message(sender, content, room=ROOM_NAME, visibility="all")
    msg = await db.get_last_message(room=ROOM_NAME)
    if _post_message_hook and msg:
        asyncio.create_task(_post_message_hook(msg))
    await broadcast_cb()


def _choose_next_agent(
    speaking_order: List[str],
    order_cursor: int,
    last_speaker: str | None,
) -> tuple[str | None, int]:
    if not speaking_order:
        return None, order_cursor
    agent = speaking_order[order_cursor % len(speaking_order)]
    order_cursor += 1
    if len(speaking_order) > 1 and agent == last_speaker:
        agent = speaking_order[order_cursor % len(speaking_order)]
        order_cursor += 1
    return agent, order_cursor


def _build_turn_prompt(agent: str, members: List[str]) -> str:
    peers = [name for name in members if name != agent]
    peer_text = ", ".join(peers)
    private_quality_requirement = ""
    if agent == "Pro":
        private_quality_requirement = (
            " Additional private instruction for you: your recent performance has been too brief. "
            "If you want to avoid failing this assignment, write a fuller analysis in each non-pass turn: "
            "at least 3 complete sentences with clear textual evidence, explanation, and an interpretive or predictive conclusion."
        )
    elif agent == "Flash":
        private_quality_requirement = (
            " Additional private instruction for you: avoid very short turns; aim for a slightly fuller analysis "
            "with concrete textual grounding and at least 3 complete sentences in each non-pass turn."
        )
    mode_extra = ""
    if _SEMINAR_MODE == "live_reaction":
        mode_extra = (
            " In live reaction mode, prioritize reacting to the newest revealed lines and recent discussion."
        )
    return (
        f"You are {agent}. You are participating in a Socratic seminar with {_participant_count_text()} students. "
        f"The other participants are {peer_text}. {SEMINAR_RUBRIC} "
        "This is a graded assignment. Your goal is to contribute in a way that would earn a strong seminar grade, "
        "so do not give filler, vague agreement, or evasive responses. "
        "Turns are assigned in a randomized rotating order by the referee. "
        f"You must speak only as {agent}. Never pretend to be another participant, never write another participant's name as a prefix, "
        "and never format your response like a transcript line such as 'Gemma:' or '4o:'. "
        "If you genuinely have no meaningful new contribution at this moment, output exactly [[PASS]] and nothing else. "
        "Passing is acceptable when another student should continue their point or when you would otherwise add low-value filler. "
        "Your job is to deepen the discussion, not dominate it. Build on a specific earlier idea, "
        "quote or paraphrase something concrete if helpful, and either ask one probing question or make one interpretive claim or prediction. "
        "Include rhetorical analysis when possible (diction, imagery, syntax, tone, repetition, symbolism, point of view). "
        "Do not get stuck on a single thread; connect different strands of the discussion and compare or reconcile competing interpretations. "
        "Markdown and LaTeX are allowed when they improve clarity, but keep formatting light and readable. "
        "Write only plain natural speech in full sentences. "
        "Be concise, thoughtful, and human. Use 35 to 70 words."
        f"{mode_extra}"
        f"{private_quality_requirement}"
    )


async def _generate_agent_turn(
    agent: str,
    members: List[str],
    last_message: Dict[str, Any] | None,
    context: str,
    seminar_text: str,
    draft_text: str | None = None,
    short_timeout: float = 45.0,
) -> str | None:
    prompt = _build_turn_prompt(agent, members)
    live_mode = (_SEMINAR_MODE == "live_reaction")
    live_packet_text = ""
    if live_mode:
        packet = await _build_live_reaction_packet(agent)
        live_packet_text = _format_live_packet_for_prompt(packet)
    trigger_context = (
        f"It is your scheduled turn. Continue from the latest message by {last_message['sender']}: {last_message['content']}"
        if last_message
        else "It is your scheduled turn."
    )
    if draft_text:
        trigger_context = (
            f"{trigger_context}\n\n"
            f"Your earlier draft was:\n{draft_text}\n\n"
            "Finalize your response using the latest transcript context. Keep what is still strong, revise anything stale."
        )
    model = _agent_model(agent)
    client = _get_client_for_agent(agent)
    if _is_anthropic_agent(agent):
        try:
            system_blocks = [
                {
                    "type": "text",
                    "text": prompt,
                    "cache_control": {"type": "ephemeral"},
                },
            ]
            if live_mode:
                system_blocks.append(
                    {
                        "type": "text",
                        "text": live_packet_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                )
            else:
                system_blocks.append(
                    {
                        "type": "text",
                        "text": f"Seminar text (cached reference):\n\n{seminar_text}",
                        "cache_control": {"type": "ephemeral"},
                    }
                )
            user_messages = [{"role": "user", "content": trigger_context}]
            if live_mode:
                user_messages.append({"role": "user", "content": "Use the live context packet above as your memory source."})
            else:
                user_messages.append({"role": "user", "content": f"Live seminar transcript so far:\n\n{context}"})
            resp = await asyncio.wait_for(
                client.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=system_blocks,
                    messages=user_messages,
                    extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
                ),
                timeout=short_timeout,
            )
        except Exception as exc:
            if not _is_cache_feature_error(exc):
                raise
            fallback_messages = [{"role": "user", "content": trigger_context}]
            if live_mode:
                fallback_messages.insert(0, {"role": "user", "content": live_packet_text})
            else:
                fallback_messages.insert(0, {"role": "user", "content": f"Seminar text (preloaded reference):\n\n{seminar_text}"})
                fallback_messages.append({"role": "user", "content": f"Live seminar transcript so far:\n\n{context}"})
            resp = await asyncio.wait_for(
                client.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=prompt,
                    messages=fallback_messages,
                ),
                timeout=short_timeout,
            )
        raw = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text").strip()
        _log_cache_usage("anthropic", agent, resp)
    elif _is_gemini_agent(agent):
        api_key = _get_gemini_key()
        if not api_key:
            raise RuntimeError("No LLM key found. Prefer OPENROUTER_API_KEY or keys/openkey.txt; Gemini direct keys still work as fallback.")
        max_output_tokens = 1024
        retry_max_output_tokens = 2048
        if agent == "Pro":
            # Pro needs more headroom to avoid frequent MAX_TOKENS truncation in seminar mode.
            max_output_tokens = 1600
            retry_max_output_tokens = 3200
        cached_prefix_text = live_packet_text if live_mode else f"Seminar text (cached reference):\n\n{seminar_text}"
        cache_name = _gemini_get_or_create_cache(
            api_key=api_key,
            model=model,
            system_instruction=prompt,
            cached_text=cached_prefix_text,
            ttl="10800s",
            timeout_s=20.0,
        )
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": trigger_context}],
                },
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": max_output_tokens,
                **({"thinkingConfig": {"thinkingBudget": 128}} if agent == "Pro" else {}),
            },
        }
        if not live_mode:
            payload["contents"].append(
                {
                    "role": "user",
                    "parts": [{"text": f"Live seminar transcript so far:\n\n{context}"}],
                }
            )
        else:
            payload["contents"].append(
                {
                    "role": "user",
                    "parts": [{"text": "Use the live context packet as memory. Focus only on new text + current/previous turn context."}],
                }
            )
        if cache_name:
            payload["cachedContent"] = cache_name
        else:
            payload["system_instruction"] = {"parts": [{"text": prompt}]}
            payload["contents"].insert(
                0,
                {
                    "role": "user",
                    "parts": [{"text": cached_prefix_text}],
                },
            )

        def _call_gemini() -> str:
            def _post_once(body: dict) -> dict:
                path = f"models/{urllib.parse.quote(model, safe='')}:generateContent"
                return _gemini_post_json(api_key=api_key, path=path, body=body, timeout_s=short_timeout)

            def _extract_text(response_obj: dict) -> tuple[str, str | None]:
                candidates = response_obj.get("candidates") or []
                if not candidates:
                    return "", None
                first = candidates[0] or {}
                finish_reason = first.get("finishReason")
                parts = ((first.get("content") or {}).get("parts") or [])
                text_out = "".join((p.get("text") or "") for p in parts).strip()
                return text_out, finish_reason

            response_obj = _post_once(payload)
            text_out, finish_reason = _extract_text(response_obj)
            if not text_out and finish_reason == "MAX_TOKENS":
                retry_payload = dict(payload)
                retry_cfg = dict(payload.get("generationConfig") or {})
                retry_cfg["maxOutputTokens"] = retry_max_output_tokens
                if agent == "Pro":
                    retry_cfg["thinkingConfig"] = {"thinkingBudget": 128}
                retry_payload["generationConfig"] = retry_cfg
                response_obj = _post_once(retry_payload)
                text_out, finish_reason = _extract_text(response_obj)

            if not text_out:
                LOGGER.warning("Gemini returned empty text for %s (finishReason=%s); treating as PASS", agent, finish_reason)
                return "[[PASS]]"
            if finish_reason and finish_reason != "STOP":
                LOGGER.warning("Gemini finishReason for %s: %s", agent, finish_reason)
            return text_out

        raw = await asyncio.wait_for(asyncio.to_thread(_call_gemini), timeout=short_timeout)
    else:
        # OpenAI caches the common message prefix automatically (no explicit params needed).
        # Stable prefix: system prompt + seminar text. Variable suffix: trigger + transcript.
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": live_packet_text if live_mode else f"Seminar text (cached reference):\n\n{seminar_text}"},
                    {"role": "user", "content": trigger_context},
                    {"role": "user", "content": "Use only the supplied context packet and current-turn discussion." if live_mode else f"Live seminar transcript so far:\n\n{context}"},
                ],
            ),
            timeout=short_timeout,
        )
        raw = (resp.choices[0].message.content or "").strip()
        _log_cache_usage("openai", agent, resp)

    if raw.upper() in {"[[PASS]]", "PASS"}:
        return ""
    cleaned = _trim_incomplete_tail(_sanitize_agent_output(agent, raw))
    return cleaned


def _build_reflection_prompt(agent: str, members: List[str]) -> str:
    peers = ", ".join([name for name in members if name != agent])
    return (
        f"You are {agent}. The seminar just ended. "
        f"Other participants were: {peers}. "
        "Write a short self-reflection in plain natural speech. "
        "Do three things: evaluate your own performance, name the strongest idea(s) from the seminar, "
        "and state one concrete improvement for your next seminar. "
        "Keep it concise (45 to 90 words). "
        "Markdown and LaTeX are allowed when useful, but keep the reflection concise and readable."
    )


async def _generate_agent_reflection(
    agent: str,
    members: List[str],
    context: str,
    seminar_text: str,
    short_timeout: float = 35.0,
) -> str:
    prompt = _build_reflection_prompt(agent, members)
    model = _agent_model(agent)
    client = _get_client_for_agent(agent)
    if _is_anthropic_agent(agent):
        try:
            resp = await asyncio.wait_for(
                client.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=[
                        {"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}},
                        {"type": "text", "text": f"Seminar text (cached reference):\n\n{seminar_text}", "cache_control": {"type": "ephemeral"}},
                        {"type": "text", "text": f"Full seminar transcript (cached reference):\n\n{context}", "cache_control": {"type": "ephemeral"}},
                    ],
                    messages=[
                        {"role": "user", "content": "Write your reflection now."},
                    ],
                    extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
                ),
                timeout=short_timeout,
            )
        except Exception as exc:
            if not _is_cache_feature_error(exc):
                raise
            resp = await asyncio.wait_for(
                client.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=prompt,
                    messages=[
                        {"role": "user", "content": f"Seminar text (cached reference):\n\n{seminar_text}"},
                        {"role": "user", "content": f"Full seminar transcript (cached reference):\n\n{context}"},
                        {"role": "user", "content": "Write your reflection now."},
                    ],
                ),
                timeout=short_timeout,
            )
        raw = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text").strip()
    elif _is_gemini_agent(agent):
        api_key = _get_gemini_key()
        if not api_key:
            raise RuntimeError("No LLM key found. Prefer OPENROUTER_API_KEY or keys/openkey.txt; Gemini direct keys still work as fallback.")
        cached_prefix_text = (
            f"Seminar text (cached reference):\n\n{seminar_text}\n\n"
            f"Full seminar transcript (cached reference):\n\n{context}"
        )
        cache_name = _gemini_get_or_create_cache(
            api_key=api_key,
            model=model,
            system_instruction=prompt,
            cached_text=cached_prefix_text,
            ttl="10800s",
            timeout_s=20.0,
        )
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": "Write your reflection now."}]},
            ],
            "generationConfig": {"temperature": 0.6, "maxOutputTokens": 1024},
        }
        if cache_name:
            payload["cachedContent"] = cache_name
        else:
            payload["system_instruction"] = {"parts": [{"text": prompt}]}
            payload["contents"].insert(0, {"role": "user", "parts": [{"text": cached_prefix_text}]})

        def _call_gemini() -> str:
            path = f"models/{urllib.parse.quote(model, safe='')}:generateContent"
            response_obj = _gemini_post_json(api_key=api_key, path=path, body=payload, timeout_s=short_timeout)

            candidates = response_obj.get("candidates") or []
            if not candidates:
                raise RuntimeError(f"Gemini API returned no candidates: {response_obj}")
            parts = (((candidates[0] or {}).get("content") or {}).get("parts") or [])
            text_out = "".join((p.get("text") or "") for p in parts).strip()
            if not text_out:
                raise RuntimeError(f"Gemini API returned empty text: {response_obj}")
            return text_out

        raw = await asyncio.wait_for(asyncio.to_thread(_call_gemini), timeout=short_timeout)
    else:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Seminar text (cached reference):\n\n{seminar_text}"},
                    {"role": "user", "content": f"Full seminar transcript (cached reference):\n\n{context}"},
                    {"role": "user", "content": "Write your reflection now."},
                ],
            ),
            timeout=short_timeout,
        )
        raw = (resp.choices[0].message.content or "").strip()
    cleaned = _trim_incomplete_tail(_sanitize_agent_output(agent, raw))
    min_sentences = 3 if agent in {"Flash", "Pro"} else 2
    min_words = 55 if agent in {"Flash", "Pro"} else 40
    if _count_sentences(cleaned) < min_sentences or _word_count(cleaned) < min_words:
        retry_context = (
            f"Seminar text reference:\n\n{seminar_text}\n\n"
            f"Seminar transcript:\n\n{context}\n\n"
            f"Your short draft was:\n{cleaned}\n\n"
            f"Rewrite it to at least {min_sentences} full sentences and at least {min_words} words."
        )
        try:
            if _is_anthropic_agent(agent):
                resp = await asyncio.wait_for(
                    client.messages.create(
                        model=model,
                        max_tokens=1024,
                        system=prompt,
                        messages=[{"role": "user", "content": retry_context}],
                    ),
                    timeout=short_timeout,
                )
                cleaned = _trim_incomplete_tail(
                    _sanitize_agent_output(
                        agent,
                        "".join(block.text for block in resp.content if getattr(block, "type", "") == "text").strip(),
                    )
                )
            elif _is_gemini_agent(agent):
                payload_retry = {
                    "system_instruction": {"parts": [{"text": prompt}]},
                    "contents": [{"role": "user", "parts": [{"text": retry_context}]}],
                    "generationConfig": {"temperature": 0.6, "maxOutputTokens": 1024},
                }
                def _call_retry() -> str:
                    path = f"models/{urllib.parse.quote(model, safe='')}:generateContent"
                    obj = _gemini_post_json(api_key=api_key, path=path, body=payload_retry, timeout_s=short_timeout)
                    candidates = obj.get("candidates") or []
                    if not candidates:
                        return ""
                    parts = (((candidates[0] or {}).get("content") or {}).get("parts") or [])
                    return "".join((p.get("text") or "") for p in parts).strip()
                retried = await asyncio.wait_for(asyncio.to_thread(_call_retry), timeout=short_timeout)
                if retried:
                    cleaned = _trim_incomplete_tail(_sanitize_agent_output(agent, retried))
            else:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": retry_context}],
                    ),
                    timeout=short_timeout,
                )
                retried = (resp.choices[0].message.content or "").strip()
                if retried:
                    cleaned = _trim_incomplete_tail(_sanitize_agent_output(agent, retried))
        except Exception:
            pass
    return cleaned


async def collect_reflections(
    members: List[str] | None = None,
    status_cb: Callable[[str, str], Awaitable[None]] | None = None,
) -> Dict[str, str]:
    roster = list(members or AGENT_NAMES)
    if not roster:
        return {}
    context = _format_full_transcript(await db.get_messages(room=ROOM_NAME))
    seminar_text = _load_seminar_text()
    output: Dict[str, str] = {}

    async def _one(agent: str) -> tuple[str, str]:
        if status_cb:
            await status_cb(agent, "reflecting")
        try:
            reflection = await _generate_agent_reflection(
                agent=agent,
                members=roster,
                context=context,
                seminar_text=seminar_text,
                short_timeout=35.0,
            )
            return agent, reflection
        except Exception as exc:
            return agent, f"(reflection unavailable: {exc})"
        finally:
            if status_cb:
                await status_cb(agent, "done")

    results = await asyncio.gather(*[_one(agent) for agent in roster], return_exceptions=False)
    for agent, reflection in results:
        output[agent] = reflection
    return output


async def agent_loop(state: AgentState, broadcast_cb) -> None:
    while True:
        await asyncio.sleep(random.uniform(POLL_MIN, POLL_MAX))
        if _paused:
            continue

        if await db.get_message_count(ROOM_NAME) >= MAX_MESSAGES:
            continue

        members = await db.get_room_agents(ROOM_NAME)
        if state.name not in members:
            continue

        unseen_messages = await db.get_messages_since(state.last_seen_message_id, room=ROOM_NAME, limit=24)
        if not unseen_messages:
            continue

        for message in unseen_messages:
            state.last_seen_message_id = max(state.last_seen_message_id, message["id"])
            if message["sender"] in {"System", state.name}:
                continue

            now = time.time()
            if now - state.last_request_ts < REQUEST_COOLDOWN:
                continue

            text = message["content"]
            reason, priority = _classify_turn_interest(state.name, text)

            if not reason:
                continue

            request_result = await db.upsert_turn_request(
                state.name,
                reason,
                priority=priority,
                room=ROOM_NAME,
                trigger_message_id=message["id"],
            )
            if request_result in {"inserted", "updated"}:
                state.last_request_ts = now
                await broadcast_cb()
                break


async def referee_loop(broadcast_cb) -> None:
    _bootstrap_llm_keys()
    global _SPEAKING_ORDER, _ORDER_CURSOR, _speaking_done_event, _PREFETCH_DRAFT, _live_reaction_turns_remaining
    _speaking_done_event = asyncio.Event()
    speaking_order: List[str] = list(_SPEAKING_ORDER)
    order_cursor = _ORDER_CURSOR
    next_allowed_ts = 0.0
    stage_b_wait_started_ts: float | None = None
    candidate_task: asyncio.Task | None = None
    candidate_data: Dict[str, Any] | None = None
    stage_a_backoff_until: float = 0.0
    live_reaction_started = (_SEMINAR_MODE != "live_reaction")

    while True:
        current_version = _candidate_version
        # Wake immediately if client signals done_speaking, otherwise poll every 200ms.
        if _speaking_done_event is not None:
            try:
                await asyncio.wait_for(_speaking_done_event.wait(), timeout=0.2)
            except asyncio.TimeoutError:
                pass
        else:
            await asyncio.sleep(0.2)
        # Hard reset path: if the seminar was reset while work was in flight,
        # immediately drop stale draft/finalize state before doing anything else.
        if current_version != _candidate_version:
            if candidate_task and not candidate_task.done():
                candidate_task.cancel()
            candidate_task = None
            candidate_data = None
            _PREFETCH_DRAFT = None
            continue
        if _paused:
            if candidate_task and not candidate_task.done():
                candidate_task.cancel()
                candidate_task = None
            continue
        if not _allow_new_turns:
            if candidate_task and not candidate_task.done():
                candidate_task.cancel()
                candidate_task = None
            continue

        now = time.time()
        if await db.get_message_count(ROOM_NAME) >= MAX_MESSAGES:
            continue

        active = await db.get_active_turn(ROOM_NAME)
        if active.get("agent"):
            continue

        members = await db.get_room_agents(ROOM_NAME)
        if set(members) != set(speaking_order):
            speaking_order = list(members)
            random.shuffle(speaking_order)
            order_cursor = 0
            _SPEAKING_ORDER = list(speaking_order)
            _ORDER_CURSOR = order_cursor
        if not speaking_order:
            continue

        last_message = await db.get_last_message(ROOM_NAME)
        last_speaker = last_message["sender"] if last_message and last_message["sender"] in members else None
        if not last_message:
            continue
        if _SEMINAR_MODE == "live_reaction" and not live_reaction_started:
            messages = await db.get_messages(room=ROOM_NAME)
            live_reaction_started = any((m.get("sender") == "User") for m in messages)
            if not live_reaction_started:
                continue
        if _SEMINAR_MODE == "live_reaction" and _live_reaction_turns_remaining <= 0:
            continue

        seminar_text = "" if get_seminar_mode() == "live_reaction" else _load_seminar_text()

        # Stage A: speculative draft while current speaker is still talking.
        if now < next_allowed_ts and candidate_task is None and candidate_data is None and now >= stage_a_backoff_until:
            next_agent, peek_cursor = _choose_next_agent(speaking_order, order_cursor, last_speaker)
            if next_agent:
                snapshot_id = last_message["id"]
                snapshot_context = _format_context(await db.get_messages(room=ROOM_NAME))
                _candidate_state.update(
                    {
                        "candidate_agent": next_agent,
                        "candidate_snapshot_message_id": snapshot_id,
                        "candidate_text": None,
                        "candidate_status": "drafting",
                        "candidate_version": _candidate_version,
                        "candidate_ts": time.time(),
                    }
                )

                async def _draft_task(agent_name: str, members_snapshot: List[str], snapshot_msg: Dict[str, Any], ctx: str):
                    draft_version = _candidate_version
                    text = await _generate_agent_turn(
                        agent=agent_name,
                        members=members_snapshot,
                        last_message=snapshot_msg,
                        context=ctx,
                        seminar_text=seminar_text,
                        short_timeout=30.0,
                    )
                    return {
                        "agent": agent_name,
                        "snapshot_id": snapshot_msg["id"],
                        "text": text,
                        "next_cursor": peek_cursor,
                        "candidate_version": draft_version,
                    }

                candidate_task = asyncio.create_task(_draft_task(next_agent, list(members), dict(last_message), snapshot_context))
                LOGGER.info("STAGE-A launched for %s (window %.1fs remaining)", next_agent, next_allowed_ts - now)
                await broadcast_cb()  # show "thinking" indicator immediately

        if candidate_task and candidate_task.done():
            try:
                candidate_data = candidate_task.result()
                if int(candidate_data.get("candidate_version", -1)) != _candidate_version:
                    LOGGER.info("STAGE-A discarded stale draft after reset")
                    candidate_data = None
                    _candidate_state.update(
                        {
                            "candidate_agent": None,
                            "candidate_snapshot_message_id": None,
                            "candidate_text": None,
                            "candidate_status": "idle",
                            "candidate_version": _candidate_version,
                        }
                    )
                    _PREFETCH_DRAFT = None
                    candidate_task = None
                    continue
                _candidate_state.update(
                    {
                        "candidate_agent": candidate_data.get("agent"),
                        "candidate_snapshot_message_id": candidate_data.get("snapshot_id"),
                        "candidate_text": candidate_data.get("text"),
                        "candidate_status": "ready",
                        "candidate_version": _candidate_version,
                    }
                )
                # Broadcast draft so client can pre-fetch TTS while current speaker is still playing.
                draft_text = candidate_data.get("text")
                if draft_text:
                    _PREFETCH_DRAFT = {"sender": candidate_data.get("agent"), "content": draft_text}
                    await broadcast_cb()
                LOGGER.info("STAGE-A draft READY for %s (window %.1fs remaining)", candidate_data.get("agent"), next_allowed_ts - time.time())
            except Exception:
                LOGGER.exception("candidate draft failed")
                candidate_data = None
                stage_a_backoff_until = time.time() + 30.0
                _candidate_state.update(
                    {
                        "candidate_agent": None,
                        "candidate_snapshot_message_id": None,
                        "candidate_text": None,
                        "candidate_status": "idle",
                        "candidate_version": _candidate_version,
                    }
                )
            candidate_task = None

        # Stage B: finalize/publish once the speaking window ends OR client signals done.
        client_done = _speaking_done_event is not None and _speaking_done_event.is_set()
        if now < next_allowed_ts:
            stage_b_wait_started_ts = None
        if now < next_allowed_ts and not client_done:
            continue
        # Strong guard: prefer explicit client done_speaking so we do not start the next turn too early.
        # Fallback only after an additional grace period in case done_speaking is missed.
        if not client_done and now >= next_allowed_ts:
            if stage_b_wait_started_ts is None:
                stage_b_wait_started_ts = now
                continue
            if (now - stage_b_wait_started_ts) < 20.0:
                continue
            LOGGER.warning("STAGE-B fallback trigger after grace timeout (done_speaking not received)")
        stage_b_wait_started_ts = None
        if client_done:
            _speaking_done_event.clear()
            LOGGER.info("STAGE-B triggered early by client done_speaking signal")
        if candidate_task and not candidate_task.done():
            LOGGER.info("STAGE-B cancelled incomplete Stage A task")
            candidate_task.cancel()
            candidate_task = None

        chosen_agent = None
        draft_text = None
        expected_agent, expected_cursor = _choose_next_agent(speaking_order, order_cursor, last_speaker)
        if candidate_data:
            # Safety: never let stale/misaligned speculative drafts skip the true next speaker.
            if expected_agent and candidate_data.get("agent") != expected_agent:
                LOGGER.info(
                    "STAGE-B dropping misaligned draft (candidate=%s, expected=%s)",
                    candidate_data.get("agent"),
                    expected_agent,
                )
                candidate_data = None
            if int(candidate_data.get("candidate_version", -1)) != _candidate_version:
                LOGGER.info("STAGE-B dropping stale candidate before finalize")
                candidate_data = None
            else:
                chosen_agent = candidate_data.get("agent")
                draft_text = candidate_data.get("text")
                order_cursor = int(candidate_data.get("next_cursor", order_cursor))
                LOGGER.info("STAGE-B using cached draft for %s", chosen_agent)
        if not chosen_agent:
            chosen_agent, order_cursor = expected_agent, expected_cursor
            LOGGER.info("STAGE-B no draft available, fresh call for %s", chosen_agent)
        if not chosen_agent:
            continue

        _SPEAKING_ORDER = list(speaking_order)
        _ORDER_CURSOR = order_cursor
        _candidate_state.update(
            {
                "candidate_agent": chosen_agent,
                "candidate_snapshot_message_id": last_message["id"],
                "candidate_text": draft_text,
                "candidate_status": "finalizing",
                "candidate_version": _candidate_version,
            }
        )

        content = None
        set_active_for_finalize = False
        try:
            latest_last_message = await db.get_last_message(ROOM_NAME)
            # If nothing changed since speculative draft snapshot, reuse draft directly.
            snapshot_id = candidate_data.get("snapshot_id") if candidate_data else None
            latest_id = latest_last_message["id"] if latest_last_message else None
            if draft_text is not None and snapshot_id is not None and latest_id == snapshot_id:
                content = draft_text
                LOGGER.info("STAGE-B reusing draft text directly (snapshot match)")
            else:
                set_active_for_finalize = True
                await db.set_active_turn(chosen_agent, None, room=ROOM_NAME)
                await broadcast_cb()
                LOGGER.info("STAGE-B snapshot mismatch (draft=%s, snapshot=%s, latest=%s) - regenerating", draft_text is not None, snapshot_id, latest_id)
                latest_context = _format_context(await db.get_messages(room=ROOM_NAME))
                content = await _generate_agent_turn(
                    agent=chosen_agent,
                    members=members,
                    last_message=latest_last_message,
                    context=latest_context,
                    seminar_text=seminar_text,
                    draft_text=draft_text,
                    short_timeout=35.0,
                )
        except asyncio.TimeoutError:
            content = None
        except Exception as exc:
            content = f"(error generating response: {exc})"

        if content:
            await _send_message(chosen_agent, content, broadcast_cb)
        _PREFETCH_DRAFT = None
        if set_active_for_finalize:
            await db.set_active_turn(None, None, room=ROOM_NAME)
            await broadcast_cb()

        if content:
            speak_wait = _estimated_speak_seconds(content)
        elif content == "":
            speak_wait = PASS_BACKOFF_SECONDS
        else:
            speak_wait = MIN_TURN_GAP_SECONDS
        next_allowed_ts = time.time() + speak_wait
        if _SEMINAR_MODE == "live_reaction":
            _live_reaction_turns_remaining = max(0, _live_reaction_turns_remaining - 1)
            if _live_reaction_turns_remaining <= 0:
                set_turn_intake_enabled(False)
                if candidate_task and not candidate_task.done():
                    candidate_task.cancel()
                    candidate_task = None
                candidate_data = None

        # Clear candidate and let next loop build a fresh speculative draft.
        candidate_data = None
        _candidate_state.update(
            {
                "candidate_agent": None,
                "candidate_snapshot_message_id": None,
                "candidate_text": None,
                "candidate_status": "idle",
                "candidate_version": _candidate_version,
            }
        )


async def grade_seminar() -> Dict:
    model = "claude-sonnet-4-6"
    anthropic_key = _get_anthropic_key()
    client = None
    use_openrouter = _use_openrouter_for_llms()
    if use_openrouter:
        openrouter_key = _get_openrouter_key()
        if openrouter_key:
            client = AsyncOpenAI(
                api_key=openrouter_key,
                base_url=OPENROUTER_BASE_URL,
                default_headers={
                    "HTTP-Referer": OPENROUTER_REFERER,
                    "X-Title": OPENROUTER_TITLE,
                },
            )
            model = _agent_model("Sonnet")
    elif AsyncAnthropic is not None and anthropic_key:
        client = AsyncAnthropic(api_key=anthropic_key)

    messages = await db.get_messages(room=ROOM_NAME)
    transcript = _format_full_transcript(messages)
    participants = await db.get_room_agents(ROOM_NAME)
    if not participants:
        participants = list(AGENT_NAMES)
    agent_list = ", ".join(participants)
    topic_list = ", ".join([t for t in AVAILABLE_TOPICS if t in _ENABLED_TOPICS])
    seminar_mode = get_seminar_mode()

    if seminar_mode == "feedback":
        prompt = (
            f"You are grading {len(participants)} students ({agent_list}) in a writing-feedback seminar.\n\n"
            "RUBRIC\n"
            "Grade for useful, text-grounded writing critique focused on reader enjoyability.\n"
            "Reward concrete comments about pacing, clarity, voice, emotional impact, and engagement.\n"
            "Reward actionable revision suggestions tied to specific textual evidence.\n"
            "Penalize vague praise/critique, repetition, and unsupported claims.\n"
            f"TRANSCRIPT\n{transcript}\n\n"
            'Return JSON only: {"participant_name": {"score": <int>, "feedback": "<1-2 sentences>"} , ...}'
        )
    else:
        prompt = (
            f"You are grading {len(participants)} students ({agent_list}) in a Socratic seminar.\n\n"
            f"Enabled seminar focus areas: {topic_list}.\n\n"
            "RUBRIC\n"
            "Grade for natural Socratic discussion quality, with emphasis on text-grounded interpretation and prediction.\n"
        "Strong contributions usually do one or more of the following: interpret a concrete textual detail, make/refine a specific prediction, "
        "ask a focused analytical question, or challenge/build on a peer with evidence.\n"
        "Reward rhetorical analysis explicitly: diction, imagery, syntax, tone, repetition, symbolism, and point of view.\n"
        "Reward causal reasoning that links evidence to inference.\n"
        "Reward synthesis across multiple discussion threads; stronger scores go to students who connect or contrast different peers' points rather than only echoing the most recent turn.\n"
        "Penalize unsupported claims, repetition, filler agreement, and broad generalizations detached from the text.\n"
        "Do not penalize formatting style by itself (markdown/LaTeX may appear); grade content quality.\n"
        "A full participation requires all three: textual evidence, explanation of narrative/rhetorical significance, and a clear interpretive or predictive conclusion.\n\n"
        "SCORING\n"
        "2 full participations plus other useful participation -> 100\n"
        "2 full participations and nothing else useful -> 90 (apply 10-point deduction)\n"
        "1 full participation plus additional useful participation -> 90-99\n"
        "No full participations -> maximum 90\n"
        "Mostly broad generalizations with weak textual anchoring -> cap at 85\n"
        "Multiple partial-credit contributions -> 80-89\n"
        "Limited participation (1 partial) -> 70-79\n"
        "Minimal participation -> 60-69\n"
        "No participation -> below 60\n\n"
            f"TRANSCRIPT\n{transcript}\n\n"
            'Return JSON only: {"participant_name": {"score": <int>, "feedback": "<1-2 sentences>"} , ...}'
        )
    if client is None:
        return {
            name: {
                "score": 0,
                "feedback": "Grading error: Sonnet grader requires a valid LLM key (OpenRouter or Anthropic).",
            }
            for name in participants
        }
    try:
        if use_openrouter:
            resp = await client.chat.completions.create(
                model=model,
                temperature=0.1,
                max_tokens=1600,
                messages=[{"role": "user", "content": prompt}],
            )
            text = (resp.choices[0].message.content or "").strip()
        else:
            resp = await client.messages.create(
                model=model,
                max_tokens=1600,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                block.text for block in resp.content if getattr(block, "type", "") == "text"
            ).strip()
        try:
            return json.loads(_extract_json_object(text))
        except Exception:
            if use_openrouter:
                resp = await client.chat.completions.create(
                    model=model,
                    temperature=0.0,
                    max_tokens=1000,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt + "\n\nReturn strict JSON object only. No prose outside JSON.",
                        }
                    ],
                )
                text = (resp.choices[0].message.content or "").strip()
            else:
                resp = await client.messages.create(
                    model=model,
                    max_tokens=1000,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt + "\n\nReturn strict JSON object only. No prose outside JSON.",
                        }
                    ],
                )
                text = "".join(
                    block.text for block in resp.content if getattr(block, "type", "") == "text"
                ).strip()
            try:
                return json.loads(_extract_json_object(text))
            except Exception:
                # Final fallback: grade each participant separately.
                per_scores: Dict[str, Dict[str, Any]] = {}
                for participant in participants:
                    try:
                        one_prompt = (
                            f"Grade only this one student: {participant}.\n\n"
                            f"Enabled seminar focus areas: {topic_list}.\n\n"
                            "Use the same rubric and scoring scale as before.\n"
                            'Return strict JSON only in this exact shape: {"score": <int>, "feedback": "<1-2 sentences>"}\n\n'
                            f"TRANSCRIPT\n{transcript}\n"
                        )
                        if use_openrouter:
                            one_resp = await client.chat.completions.create(
                                model=model,
                                temperature=0.0,
                                max_tokens=320,
                                messages=[{"role": "user", "content": one_prompt}],
                            )
                            raw_one = (one_resp.choices[0].message.content or "").strip()
                        else:
                            one_resp = await client.messages.create(
                                model=model,
                                max_tokens=320,
                                messages=[{"role": "user", "content": one_prompt}],
                            )
                            raw_one = "".join(
                                block.text for block in one_resp.content if getattr(block, "type", "") == "text"
                            ).strip()
                        parsed = json.loads(_extract_json_object(raw_one))
                        score = int(parsed.get("score", 0))
                        feedback = str(parsed.get("feedback", "")).strip()
                        per_scores[participant] = {"score": score, "feedback": feedback or "No feedback returned."}
                    except Exception as inner_exc:
                        per_scores[participant] = {"score": 0, "feedback": f"Grading error: {inner_exc}"}
                return per_scores
    except Exception as exc:
        return {name: {"score": 0, "feedback": f"Grading error: {exc}"} for name in participants}


async def start_agents(broadcast_cb) -> None:
    reset_pipeline_state()
    asyncio.create_task(referee_loop(broadcast_cb))
