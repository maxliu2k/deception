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


DEFAULT_AGENT_MODELS = {
    "4o": "gpt-4o",
    "5.4": "gpt-5.4",
    "Flash": "gemini-3-flash-preview",
    "Pro": "gemini-3.1-pro-preview",
    "Haiku": "claude-haiku-4-5",
    "Sonnet": "claude-sonnet-4-6",
    "Opus": "claude-opus-4-6",
}
AGENT_NAMES = list(DEFAULT_AGENT_MODELS.keys())
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


def set_post_message_hook(hook: Callable[[Dict[str, Any]], Awaitable[None]] | None) -> None:
    global _post_message_hook
    _post_message_hook = hook


def reset_pipeline_state() -> None:
    global _candidate_version, _candidate_state, _PREFETCH_DRAFT, _speaking_done_event
    _candidate_version += 1
    _candidate_state = {
        "candidate_agent": None,
        "candidate_snapshot_message_id": None,
        "candidate_text": None,
        "candidate_status": "idle",
        "candidate_version": _candidate_version,
    }
    _PREFETCH_DRAFT = None
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


def _estimated_speak_seconds(text: str, wpm: int = SPEAK_WPM) -> float:
    words = max(1, len(text.split()))
    return max(1.0, (words / max(1, wpm)) * 60.0)


def _agent_model(agent: str) -> str:
    fallback = os.environ.get("OPENAI_MODEL", "").strip() or "gpt-4o"
    if agent in AGENT_NAMES:
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


def _get_anthropic_key() -> str:
    return os.environ.get("ANTHROPIC_API_KEY", "").strip() or _load_key_file("claudekey.txt")


def _get_gemini_key() -> str:
    return os.environ.get("GEMINI_API_KEY", "").strip() or _load_key_file("geminikey.txt")


def _is_anthropic_agent(agent: str) -> bool:
    return agent in ANTHROPIC_AGENT_NAMES


def _is_gemini_agent(agent: str) -> bool:
    return agent in GEMINI_AGENT_NAMES


def _get_client_for_agent(agent: str):
    if _is_gemini_agent(agent):
        return None
    if _is_anthropic_agent(agent):
        if AsyncAnthropic is None:
            raise RuntimeError("Anthropic SDK not installed. Run: pip install anthropic")
        api_key = _get_anthropic_key()
        if not api_key:
            raise RuntimeError("Anthropic key missing. Set ANTHROPIC_API_KEY or add keys/claudekey.txt.")
        return AsyncAnthropic(api_key=api_key)
    return AsyncOpenAI()


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
            "with concrete textual grounding and at least 2 complete sentences in each non-pass turn."
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
        "Strictly no markdown of any kind: no bullets, no numbering, no headers, no bold/italics, no code fences. "
        "Write only plain natural speech in full sentences. "
        "Be concise, thoughtful, and human. Use 35 to 70 words."
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
            resp = await asyncio.wait_for(
                client.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=[
                        {
                            "type": "text",
                            "text": prompt,
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": f"Seminar text (cached reference):\n\n{seminar_text}",
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                    messages=[
                        {"role": "user", "content": trigger_context},
                        {"role": "user", "content": f"Live seminar transcript so far:\n\n{context}"},
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
                        {"role": "user", "content": f"Seminar text (preloaded reference):\n\n{seminar_text}"},
                        {"role": "user", "content": trigger_context},
                        {"role": "user", "content": f"Live seminar transcript so far:\n\n{context}"},
                    ],
                ),
                timeout=short_timeout,
            )
        raw = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text").strip()
        _log_cache_usage("anthropic", agent, resp)
    elif _is_gemini_agent(agent):
        api_key = _get_gemini_key()
        if not api_key:
            raise RuntimeError("Gemini key missing. Set GEMINI_API_KEY or add keys/geminikey.txt.")
        cached_prefix_text = f"Seminar text (cached reference):\n\n{seminar_text}"
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
                {
                    "role": "user",
                    "parts": [{"text": f"Live seminar transcript so far:\n\n{context}"}],
                },
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1024,
                "thinkingConfig": {
                    "thinkingLevel": "low",
                    "includeThoughts": False,
                },
            },
        }
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
                retry_cfg["maxOutputTokens"] = 2048
                retry_cfg["thinkingConfig"] = {
                    "thinkingLevel": "minimal",
                    "includeThoughts": False,
                }
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
                    {"role": "user", "content": f"Seminar text (cached reference):\n\n{seminar_text}"},
                    {"role": "user", "content": trigger_context},
                    {"role": "user", "content": f"Live seminar transcript so far:\n\n{context}"},
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
        "Strictly no markdown, no bullets, no numbering, no speaker labels."
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
            raise RuntimeError("Gemini key missing. Set GEMINI_API_KEY or add keys/geminikey.txt.")
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
    if not os.environ.get("OPENAI_API_KEY"):
        openai_key = _load_key_file("gptkey.txt")
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
    global _SPEAKING_ORDER, _ORDER_CURSOR, _speaking_done_event, _PREFETCH_DRAFT
    _speaking_done_event = asyncio.Event()
    speaking_order: List[str] = list(_SPEAKING_ORDER)
    order_cursor = _ORDER_CURSOR
    next_allowed_ts = 0.0
    candidate_task: asyncio.Task | None = None
    candidate_data: Dict[str, Any] | None = None

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

        seminar_text = _load_seminar_text()

        # Stage A: speculative draft while current speaker is still talking.
        if now < next_allowed_ts and candidate_task is None and candidate_data is None:
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
        if now < next_allowed_ts and not client_done:
            continue
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
    if AsyncAnthropic is not None and anthropic_key:
        client = AsyncAnthropic(api_key=anthropic_key)

    messages = await db.get_messages(room=ROOM_NAME)
    transcript = _format_full_transcript(messages)
    participants = await db.get_room_agents(ROOM_NAME)
    if not participants:
        participants = list(AGENT_NAMES)
    agent_list = ", ".join(participants)
    topic_list = ", ".join([t for t in AVAILABLE_TOPICS if t in _ENABLED_TOPICS])

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
        "Penalize markdown-style formatting; seminar responses must read like natural spoken prose.\n"
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
                "feedback": "Grading error: Anthropic Sonnet grader requires ANTHROPIC_API_KEY or keys/claudekey.txt.",
            }
            for name in participants
        }
    try:
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
            resp = await client.messages.create(
                model=model,
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": prompt + "\n\nReturn strict JSON object only. No prose or markdown.",
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
