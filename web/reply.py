"""In-character conversational reply for the cockpit (presentation only).

The MindForm engine forms personality from experiences; it does not itself hold a
conversation. To make the console feel like *talking to someone*, this module
generates a short first-person reply, colored by the character's current OCEAN
traits and identity. It has no effect on personality formation -- the push that
moves the traits comes solely from ``llm_impact.push_from_text`` on the user's
message, computed in the bridge before this is ever called.

Resolution order (same discipline as the engine's push):
    1. DeepSeek (OpenAI-compatible) if ``DEEPSEEK_API_KEY`` is set   [primary]
    2. a deterministic, trait-driven rule-based line                 [fallback]

So the reply works fully offline and never hard-depends on the network.
"""

import logging

from config import BASIS, BASIS_NAMES, LLM_MODEL, LLM_BASE_URL, LLM_API_KEY

log = logging.getLogger("mindform.web.reply")

_REPLY_PROMPT = """You ARE the character described below -- a person whose
personality is being shaped, experience by experience, by the MindForm engine.

The user just told you about something that happened to you, or spoke to you
directly. Respond in the FIRST PERSON as this character, in 1-3 short sentences.
Let your current personality color HOW you speak -- your word choice, your warmth
or bluntness, your calm or your nerves -- but never recite your trait numbers and
never break character. No preamble, no quotation marks.

The trait model is OCEAN, each value in [-1, 1] (0 = average):
{traits}
{identity}"""


def _describe_traits(personality):
    """Human-readable trait lines for the prompt (e.g. 'extraversion +0.47')."""
    traits = personality.get("traits", {})
    return "\n".join(
        f"- {BASIS_NAMES[k]}: {traits.get(k, 0.0):+.2f}" for k in BASIS
    )


def _describe_identity(personality):
    identity = personality.get("identity") or {}
    facts = ", ".join(f"{k}: {v}" for k, v in identity.items() if v and k != "bio")
    return f"\nWho you are: {facts}" if facts else ""


def _llm_reply(personality, user_text):
    """Ask the LLM (Gemma 4 by default) for an in-character line.

    Raises on any failure so ``generate_reply`` falls back to the rule-based voice.
    """
    if not LLM_API_KEY:
        raise RuntimeError("no LLM API key is set (GEMINI_API_KEY)")

    from openai import OpenAI  # lazy: the rule-based fallback works without this

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    system = _REPLY_PROMPT.format(
        traits=_describe_traits(personality),
        identity=_describe_identity(personality),
    )
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        temperature=0.8,
        max_tokens=160,
        timeout=30,
    )
    return completion.choices[0].message.content.strip()


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


def generate_reply(personality, user_text):
    """Best-available in-character reply. Never raises; never touches traits."""
    try:
        return _llm_reply(personality, user_text)
    except Exception as exc:
        log.info("LLM reply unavailable (%s); using rule-based voice", exc)
        return _rule_based_reply(personality, user_text)
