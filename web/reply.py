"""In-character conversational reply for the cockpit (presentation only).

The MindForm engine forms personality from experiences; it does not itself hold a
conversation. To make the console feel like *talking to someone*, this module
generates a short first-person reply. The reply's TEXT never re-enters formation --
the push that moves the traits comes solely from ``llm_impact.push_from_text`` on
the user's message, computed in the bridge before this is ever called. (What does
feed forward is engine state: the BEHAVIOR stance this reply enacts, recorded
before it is generated and resolved against the next user message as its outcome.)

The voice is the SOCIAL EXPRESSION node made audible (``nodes/expression.py``):
  * the character speaks from its SELF-VIEW (``self.image``), not its raw traits --
    we perform who we think we are -- with a one-line "leak" naming the largest
    image-vs-actual gap so the real trait can flash through;
  * a "voice" brief (style dims + active need + dominant value + this turn's mood,
    read from the INTERPRETED appraisal) tells the model HOW to carry itself;
  * offline, ``expression.plain_reply`` shapes a deterministic line with the same
    dims, so the fallback voice sounds like the character too.

Resolution order (same discipline as the engine's push):
    1. the configured LLM (Google Gemma 4 by default) if a key is set  [primary]
    2. the deterministic, style-driven ``expression.plain_reply``      [fallback]

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
    BASIS, BASIS_NAMES, LLM_MODEL, LLM_BASE_URL, LLM_API_KEY, LLM_LABEL,
    SELF_INCONGRUENCE_THRESH, strip_reasoning,
)
from nodes.expression import voice as expression_voice, plain_reply

log = logging.getLogger("mindform.web.reply")

# A blunt, concrete brief beats a poetic one: open-weight models (Gemma included)
# are far likelier to narrate a <thought> block when the instruction is abstract.
_REPLY_SYSTEM = """You are {name}. Speak ONLY as yourself, out loud, in the first person.

How you see yourself right now (the self you speak from; each -1..1, 0 = average)
shapes HOW you talk -- your warmth or bluntness, your calm or your nerves. Never say
the numbers, never break character:
{self_view}{voice}{identity}{memories}

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


def _describe_self_view(personality):
    """The self they speak from: the self-image lines (falling back to the actual traits
    when no self exists yet), plus a one-line 'leak' when the self-image has drifted from
    reality -- the real trait is allowed to flash through."""
    self_state = personality.get("self") or {}
    image = self_state.get("image") if isinstance(self_state.get("image"), dict) else None
    source = image or personality.get("traits", {})
    lines = "\n".join(f"- {BASIS_NAMES[k]}: {source.get(k, 0.0):+.2f}" for k in BASIS)
    if image:
        traits = personality.get("traits", {})
        gaps = {k: float(traits.get(k, 0.0)) - float(image.get(k, 0.0)) for k in BASIS}
        mean_gap = sum(abs(g) for g in gaps.values()) / len(BASIS)
        if mean_gap > SELF_INCONGRUENCE_THRESH:
            worst = max(BASIS, key=lambda k: abs(gaps[k]))
            direction = "higher" if gaps[worst] > 0 else "lower"
            lines += (f"\n(In truth your {BASIS_NAMES[worst]} runs {direction} than you "
                      "believe -- let it leak through only in flashes.)")
    return lines


def _describe_voice(personality, appraisal):
    """The expression node's brief: how they carry themselves when they speak."""
    lines = expression_voice(personality, appraisal)
    if not lines:
        return ""
    body = "\n".join(f"- {line}" for line in lines)
    return "\n\nHow you carry yourself when you speak (follow it, never mention it):\n" + body


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


def _llm_reply(personality, user_text, memories=None, appraisal=None):
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
        self_view=_describe_self_view(personality),
        voice=_describe_voice(personality, appraisal),
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


def generate_reply(personality, user_text, memories=None, appraisal=None):
    """Best-available in-character reply, optionally grounded in recalled memories and this
    turn's INTERPRETED appraisal (how the event landed through their lens). Returns
    ``(text, source)`` with source in {LLM label, "rule"}. Never raises; never touches traits."""
    try:
        return _llm_reply(personality, user_text, memories, appraisal), LLM_LABEL
    except Exception as exc:
        log.info("LLM reply unavailable (%s); using the offline voice", exc)
        return plain_reply(personality, user_text, appraisal), "rule"
