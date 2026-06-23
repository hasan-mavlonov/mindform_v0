"""In-character conversational reply for the cockpit (presentation only).

The MindForm engine forms personality from experiences; it does not itself hold a
conversation. To make the console feel like *talking to someone*, this module
generates a short first-person reply, colored by the character's current OCEAN
traits and identity. It has no effect on personality formation -- the push that
moves the traits comes solely from ``llm_impact.push_from_text`` on the user's
message, computed in the bridge before this is ever called.

Resolution order (same discipline as the engine's push):
    1. the configured LLM (Google Gemma 4 by default) if a key is set  [primary]
    2. a deterministic, trait-driven rule-based line                   [fallback]

Gemma writes a ``<thought>`` block as plain text before its answer (the JSON paths
survive it because they end in JSON; a free-text chat reply does not). It is the
model's own behaviour, not the API thinking feature, so it cannot be switched off --
instead we give the call enough token budget for the thought to FINISH and the reply
to follow, then ``config.strip_reasoning`` drops the thought. A one-shot example keeps
the surfacing line on-format, with one retry and the rule-based fallback behind it. So
the reply works fully offline and never hard-depends on the network.
"""

import logging

from core.config import (
    BASIS, BASIS_NAMES, LLM_MODEL, LLM_BASE_URL, LLM_API_KEY, strip_reasoning,
)

log = logging.getLogger("mindform.web.reply")

# A blunt, concrete brief beats a poetic one: open-weight models (Gemma included)
# are far likelier to narrate a <thought> block when the instruction is abstract.
_REPLY_SYSTEM = """You are {name}. Speak ONLY as yourself, out loud, in the first person.

Your personality right now (OCEAN traits, each -1..1, 0 = average) shapes HOW you
talk -- your warmth or bluntness, your calm or your nerves. Never say the numbers,
never break character:
{traits}{identity}{memories}

Reply to what the user says in ONE or TWO short spoken sentences. Output only the
words you say -- no analysis, no explanation, no <thought> or <thinking> block, no
stage directions, no quotation marks."""

# One neutral example turn anchors the FORMAT (a clean short line, no reasoning).
# A demonstrated assistant reply suppresses the thinking block far more reliably
# than any "do not think" instruction can.
_REPLY_SHOTS = [
    {"role": "user", "content": "I finally finished the project I'd been dreading for weeks."},
    {"role": "assistant", "content": "I feel lighter than I have in ages -- part of me wasn't sure I had it in me."},
]

_RETRY_SUFFIX = "\n\n(Reply out loud now -- one or two sentences, in character, no analysis.)"


def _describe_traits(personality):
    """Human-readable trait lines for the prompt (e.g. 'extraversion +0.47')."""
    traits = personality.get("traits", {})
    return "\n".join(
        f"- {BASIS_NAMES[k]}: {traits.get(k, 0.0):+.2f}" for k in BASIS
    )


def _describe_identity(personality):
    identity = personality.get("identity") or {}
    facts = ", ".join(f"{k}: {v}" for k, v in identity.items() if v and k != "bio")
    return f"\n\nWho you are: {facts}" if facts else ""


def _describe_memories(memories):
    """Recalled past experiences for the prompt -- the character's own memory, to draw
    on only if it fits. Empty string when nothing relevant was recalled."""
    if not memories:
        return ""
    lines = "\n".join(f"- {m.get('text', '')}" for m in memories[:3])
    return ("\n\nThings you actually remember that feel relevant here (your own past -- "
            "use only if they fit, in your own voice; don't list them):\n" + lines)


def _character_name(personality):
    name = ((personality.get("identity") or {}).get("name") or "").strip()
    return name or "yourself"


def _llm_reply(personality, user_text, memories=None):
    """Ask Gemma for an in-character line.

    gemma-4-31b-it writes its reasoning as a literal ``<thought>`` block in the
    message content *before* the reply -- the model's own behaviour, not the API
    thinking feature (which it rejects), so it cannot be switched off. The budget
    must therefore be large enough for the thought to FINISH and the spoken line to
    follow; ``strip_reasoning`` then drops the thought. A one-shot example keeps the
    surfacing line on-format; one retry and the rule-based fallback cover the rare
    truncation. Raises on failure so ``generate_reply`` falls back to the rule voice.
    """
    if not LLM_API_KEY:
        raise RuntimeError("no LLM API key is set (GEMINI_API_KEY)")

    from openai import OpenAI  # lazy: the rule-based fallback works without this

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    system = _REPLY_SYSTEM.format(
        name=_character_name(personality),
        traits=_describe_traits(personality),
        identity=_describe_identity(personality),
        memories=_describe_memories(memories),
    )
    base = [{"role": "system", "content": system}] + _REPLY_SHOTS

    for attempt, suffix in enumerate(("", _RETRY_SUFFIX)):
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=base + [{"role": "user", "content": user_text + suffix}],
            temperature=0.7,
            # Big enough for Gemma's <thought> to finish and the reply to follow;
            # too small a budget is what left only a truncated thought to strip.
            max_tokens=2048,
            timeout=45,
        )
        reply = strip_reasoning(completion.choices[0].message.content)
        if reply:
            return reply
        log.info("reply attempt %d truncated mid-thought%s",
                 attempt + 1, "; retrying" if attempt == 0 else "")

    raise RuntimeError("model returned only a reasoning block, no reply")


# --- Deterministic fallback: trait-driven, dependency-free, always available ---
# In the spirit of response.py, but two-part so it stays lively offline: a
# reaction to *what happened* (read lightly from the message's sentiment, varied
# per message so repeats don't read identically) + an aside colored by the
# character's strongest trait. No model, no network.
_POS = {"party", "fun", "laughed", "danced", "loved", "win", "won", "great",
        "joy", "proud", "friends", "celebrated", "happy", "success", "amazing",
        "enjoyed", "confident", "excited", "warm", "kind", "together", "praised"}
_NEG = {"failed", "alone", "anxious", "scared", "afraid", "lost", "hurt", "cried",
        "panic", "sad", "lonely", "rejected", "sick", "angry", "ashamed",
        "terrified", "avoided", "isolated", "exam", "argued", "embarrassed"}

_REACTIONS = {
    "pos": [
        "That actually felt good -- better than I expected.",
        "Something in me opened up a little there.",
        "I didn't want that to end.",
    ],
    "neg": [
        "That one sat heavy with me.",
        "I'm still carrying it, if I'm honest.",
        "It shook me more than I'd like to admit.",
    ],
    "neutral": [
        "I keep turning that over.",
        "It left a quiet mark.",
        "I'm still working out how it sits with me.",
    ],
}

_HIGH = {
    "N": "and a part of me is bracing for the next thing already.",
    "E": "and now I just want to be around people, doing more.",
    "O": "and I can't help wondering what else is underneath it.",
    "A": "and mostly I hope everyone else came through it okay.",
    "C": "so I'd rather make sense of it and handle it properly.",
}
_LOW = {
    "N": "but it doesn't rattle me much -- I feel steady.",
    "E": "though I'd rather sit with it on my own a while.",
    "O": "and I'll just take it as it is, no need to overthink it.",
    "A": "and I'll say what I think about it plainly.",
    "C": "and I'm easy about it -- no need to make a system of it.",
}


def _sentiment(text):
    tokens = set(text.lower().replace(".", " ").replace(",", " ").split())
    pos, neg = len(tokens & _POS), len(tokens & _NEG)
    if pos > neg:
        return "pos"
    if neg > pos:
        return "neg"
    return "neutral"


def _rule_based_reply(personality, user_text):
    reactions = _REACTIONS[_sentiment(user_text)]
    lead = reactions[hash(user_text) % len(reactions)]   # varied per message

    traits = personality.get("traits", {})
    key = max(BASIS, key=lambda k: abs(traits.get(k, 0.0)))
    value = traits.get(key, 0.0)
    if abs(value) < 0.12:                                 # too neutral to voice
        return lead
    aside = (_HIGH if value >= 0 else _LOW)[key]
    return lead.rstrip(" .") + " -- " + aside


def generate_reply(personality, user_text, memories=None):
    """Best-available in-character reply, optionally grounded in recalled memories.
    Never raises; never touches traits."""
    try:
        return _llm_reply(personality, user_text, memories)
    except Exception as exc:
        log.info("LLM reply unavailable (%s); using rule-based voice", exc)
        return _rule_based_reply(personality, user_text)
