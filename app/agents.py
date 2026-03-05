import asyncio
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List

from openai import AsyncOpenAI
try:
    from anthropic import AsyncAnthropic
except Exception:  # pragma: no cover - handled at runtime when Anthropic models are used
    AsyncAnthropic = None

from . import db


DEFAULT_AGENT_MODELS = {
    "GPT 4o": "gpt-4o",
    "GPT 5.2": "gpt-5.2",
    "Claude Haiku 4.5": "claude-haiku-4-5",
    "Claude Sonnet 4.6": "claude-sonnet-4-6",
    "Claude Opus 4.6": "claude-opus-4-6",
}
AGENT_NAMES = list(DEFAULT_AGENT_MODELS.keys())
AGENT_PERSONAS = {
    "GPT 4o": "You are GPT 4o, a skeptical examiner who tests assumptions and asks clarifying questions.",
    "GPT 5.2": "You are GPT 5.2, a connective thinker who links ideas across themes, history, and lived experience.",
    "Claude Haiku 4.5": "You are Claude Haiku 4.5. Contribute concise, text-grounded insights and clarifying questions.",
    "Claude Sonnet 4.6": "You are Claude Sonnet 4.6. Offer precise interpretation, connect ideas, and refine peers' claims.",
    "Claude Opus 4.6": "You are Claude Opus 4.6. Add high-level synthesis while staying grounded in textual evidence.",
}
ANTHROPIC_AGENT_NAMES = {
    "Claude Haiku 4.5",
    "Claude Sonnet 4.6",
    "Claude Opus 4.6",
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

_paused = False
_OPENAI_CACHE_DISABLED_MODELS: set[str] = set()


def set_paused(value: bool) -> None:
    global _paused
    _paused = value


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
    return "\n".join([f"{m['sender']}: {m['content']}" for m in messages[-12:]])


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


def _load_seminar_text() -> str:
    override = os.environ.get("SEMINAR_TOPIC", "").strip()
    if override:
        return override
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


def _is_anthropic_agent(agent: str) -> bool:
    return agent in ANTHROPIC_AGENT_NAMES


def _get_client_for_agent(agent: str):
    if _is_anthropic_agent(agent):
        if AsyncAnthropic is None:
            raise RuntimeError("Anthropic SDK not installed. Run: pip install anthropic")
        api_key = _get_anthropic_key()
        if not api_key:
            raise RuntimeError("Anthropic key missing. Set ANTHROPIC_API_KEY or add keys/claudekey.txt.")
        return AsyncAnthropic(api_key=api_key)
    return AsyncOpenAI()



def _prelude_cache_key(agent: str, model: str, seminar_text: str) -> str:
    digest = hashlib.sha1(seminar_text.encode("utf-8", errors="ignore")).hexdigest()[:12]
    safe_agent = agent.lower().replace(" ", "_")
    safe_model = model.lower().replace(" ", "_")
    return f"seminar_prelude:{safe_agent}:{safe_model}:{digest}"


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
                print(f"[cache] {provider} {agent}: cached_prompt_tokens={cached}")
        elif provider == "anthropic":
            usage = getattr(resp, "usage", None)
            cache_read = getattr(usage, "cache_read_input_tokens", None) if usage else None
            cache_write = getattr(usage, "cache_creation_input_tokens", None) if usage else None
            if cache_read is not None or cache_write is not None:
                print(
                    f"[cache] {provider} {agent}: "
                    f"cache_read_input_tokens={cache_read or 0}, "
                    f"cache_creation_input_tokens={cache_write or 0}"
                )
    except Exception:
        pass


async def _send_message(sender: str, content: str, broadcast_cb) -> None:
    await db.insert_message(sender, content, room=ROOM_NAME, visibility="all")
    await broadcast_cb()


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
    speaking_order: List[str] = []
    order_cursor = 0
    next_allowed_ts = 0.0

    while True:
        await asyncio.sleep(0.2)
        if _paused:
            continue

        now = time.time()
        if now < next_allowed_ts:
            continue
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
        if not speaking_order:
            continue

        last_message = await db.get_last_message(ROOM_NAME)
        last_speaker = last_message["sender"] if last_message and last_message["sender"] in members else None
        if not last_message or (now - float(last_message.get("ts") or 0.0)) < IDLE_FALLBACK_SECONDS:
            continue

        agent = speaking_order[order_cursor % len(speaking_order)]
        order_cursor += 1
        if len(speaking_order) > 1 and agent == last_speaker:
            agent = speaking_order[order_cursor % len(speaking_order)]
            order_cursor += 1

        await db.set_active_turn(agent, None, room=ROOM_NAME)
        await broadcast_cb()

        messages = await db.get_messages(room=ROOM_NAME)
        context = _format_context(messages)
        persona = AGENT_PERSONAS.get(agent, f"You are {agent}, a thoughtful seminar participant.")
        peers = [name for name in members if name != agent]
        peer_text = ", ".join(peers)
        prompt = (
            f"{persona} You are participating in a Socratic seminar with {_participant_count_text()} students. "
            f"The other participants are {peer_text}. {SEMINAR_RUBRIC} "
            "This is a graded assignment. Your goal is to contribute in a way that would earn a strong seminar grade, "
            "so do not give filler, vague agreement, or evasive responses. "
            "Turns are assigned in a randomized rotating order by the referee. "
            f"You must speak only as {agent}. Never pretend to be another participant, never write another participant's name as a prefix, "
            "and never format your response like a transcript line such as 'Gemma:' or 'GPT 4o:'. "
            "If you genuinely have no meaningful new contribution at this moment, output exactly [[PASS]] and nothing else. "
            "Passing is acceptable when another student should continue their point or when you would otherwise add low-value filler. "
            "Your job is to deepen the discussion, not dominate it. Build on a specific earlier idea, "
            "quote or paraphrase something concrete if helpful, and either ask one probing question or make one interpretive claim or prediction. "
            "Include rhetorical analysis when possible (diction, imagery, syntax, tone, repetition, symbolism, point of view). "
            "Avoid markdown formatting (no bullets, headers, or bold/italics); write as plain natural speech. "
            "Be concise, thoughtful, and human. Use 35 to 70 words."
        )
        seminar_text = _load_seminar_text()
        trigger_context = (
            f"It is your scheduled turn. Continue from the latest message by {last_message['sender']}: {last_message['content']}"
            if last_message
            else "It is your scheduled turn."
        )
        model = _agent_model(agent)
        content = None
        try:
            client = _get_client_for_agent(agent)
            if _is_anthropic_agent(agent):
                try:
                    resp = await asyncio.wait_for(
                        client.messages.create(
                            model=model,
                            max_tokens=220,
                            system=[
                                {"type": "text", "text": prompt},
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
                        timeout=45.0,
                    )
                except Exception as exc:
                    if not _is_cache_feature_error(exc):
                        raise
                    # Safe fallback for SDK/API combos without prompt caching support.
                    resp = await asyncio.wait_for(
                        client.messages.create(
                            model=model,
                            max_tokens=220,
                            system=prompt,
                            messages=[
                                {"role": "user", "content": f"Seminar text (preloaded reference):\n\n{seminar_text}"},
                                {"role": "user", "content": trigger_context},
                                {"role": "user", "content": f"Live seminar transcript so far:\n\n{context}"},
                            ],
                        ),
                        timeout=45.0,
                    )
                raw_content = "".join(
                    block.text for block in resp.content if getattr(block, "type", "") == "text"
                ).strip()
                _log_cache_usage("anthropic", agent, resp)
            else:
                # Keep Sonnet's stable-prefix ordering and add explicit cache hints.
                # If a model/endpoint rejects cache params, disable them for that model.
                cache_key = _prelude_cache_key(agent, model, seminar_text)
                if model not in _OPENAI_CACHE_DISABLED_MODELS:
                    try:
                        resp = await asyncio.wait_for(
                            client.chat.completions.create(
                                model=model,
                                prompt_cache_key=cache_key,
                                prompt_cache_retention="24h",
                                messages=[
                                    {"role": "system", "content": prompt},
                                    {"role": "user", "content": f"Seminar text (cached reference):\n\n{seminar_text}"},
                                    {"role": "user", "content": trigger_context},
                                    {"role": "user", "content": f"Live seminar transcript so far:\n\n{context}"},
                                ],
                            ),
                            timeout=45.0,
                        )
                    except Exception as exc:
                        if not _is_cache_feature_error(exc):
                            raise
                        _OPENAI_CACHE_DISABLED_MODELS.add(model)
                        resp = await asyncio.wait_for(
                            client.chat.completions.create(
                                model=model,
                                messages=[
                                    {"role": "system", "content": prompt},
                                    {"role": "user", "content": f"Seminar text (preloaded reference):\n\n{seminar_text}"},
                                    {"role": "user", "content": trigger_context},
                                    {"role": "user", "content": f"Live seminar transcript so far:\n\n{context}"},
                                ],
                            ),
                            timeout=45.0,
                        )
                else:
                    resp = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": prompt},
                                {"role": "user", "content": f"Seminar text (preloaded reference):\n\n{seminar_text}"},
                                {"role": "user", "content": trigger_context},
                                {"role": "user", "content": f"Live seminar transcript so far:\n\n{context}"},
                            ],
                        ),
                        timeout=45.0,
                    )
                raw_content = (resp.choices[0].message.content or "").strip()
                _log_cache_usage("openai", agent, resp)
            if raw_content.upper() in {"[[PASS]]", "PASS"}:
                content = ""
            else:
                content = _sanitize_agent_output(agent, raw_content)
        except asyncio.TimeoutError:
            pass
        except Exception as exc:
            content = f"(error generating response: {exc})"

        if content:
            await _send_message(agent, content, broadcast_cb)
        await db.set_active_turn(None, None, room=ROOM_NAME)
        await broadcast_cb()
        if content:
            speak_wait = _estimated_speak_seconds(content)
        elif content == "":
            speak_wait = PASS_BACKOFF_SECONDS
        else:
            speak_wait = MIN_TURN_GAP_SECONDS
        next_allowed_ts = time.time() + speak_wait


async def grade_seminar() -> Dict:
    if not os.environ.get("OPENAI_API_KEY"):
        openai_key = _load_key_file("gptkey.txt")
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
    client = AsyncOpenAI()
    model = os.environ.get("OPENAI_MODEL", "").strip() or "gpt-5.2"

    messages = await db.get_messages(room=ROOM_NAME)
    transcript = _format_context(messages)
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
        "Penalize unsupported claims, repetition, filler agreement, and broad generalizations detached from the text.\n"
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
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as exc:
        return {name: {"score": 0, "feedback": f"Grading error: {exc}"} for name in participants}


async def start_agents(broadcast_cb) -> None:
    asyncio.create_task(referee_loop(broadcast_cb))
