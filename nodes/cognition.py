"""COGNITIVE PATTERNS: the interpretation lens between perception and formation.

How a person *thinks* colours what they perceive. The engine appraises an experience the
same way for everyone (``appraisal.appraise``); this node bends that reading by who the
character currently is AND by what they remember, so the SAME event forms two different
people differently -- and lands differently on the same person depending on their past.

  * ``interpret(appraisal, personality, recalled)`` -- tilts the appraisal vector twice:
      - by current OCEAN traits (Slice 1): anxious -> more perceived threat and a darker
        valence; open -> more novelty; agreeable -> warmer on social events.
      - by memory (Slice 2): recalled similar past episodes pull the valence and threat
        *toward how those felt* (a learned expectation / schema), and the more this
        resembles lived experience, the less novel it reads (familiarity damps novelty).
    Deterministic; it colours the heuristic push (traits / values / moral all run through
    ``appraise -> impact``).
  * ``lens(personality, recalled)`` -- the same two tilts as an interpretive brief injected
    into the LLM push prompt, so the primary (Gemma) path is coloured too (it reads the raw
    text, not the appraisal). ``read_lens`` is the short version for display.

Both biases are small (``config.COGNITION_GAIN`` / ``config.MEMORY_GAIN``) and clamped, so
perception is tinted, not overwhelmed. The memory pull is *toward a bounded target* (it
asymptotes once the reading matches the remembered tone) and temperament's pull-back keeps
the perceive -> form -> perceive loop stable, so the schema reinforces without running away.
"""

from core.config import COGNITION_GAIN, MEMORY_GAIN

_THRESH = 0.25       # how far from neutral a trait must be to colour the lens
_TONE_THRESH = 0.20  # how clearly the remembered tone must lean to surface a brief / tag


def _clamp(value, low, high):
    return max(low, min(high, value))


# --- the trait lens (Slice 1): who they are bends the reading -----------------
def _trait_tilt(out, personality):
    """Tilt ``out`` (an appraisal dict, mutated in place) by current OCEAN traits."""
    traits = (personality or {}).get("traits") or {}
    N = traits.get("N", 0.0)
    O = traits.get("O", 0.0)
    A = traits.get("A", 0.0)
    g = COGNITION_GAIN

    social = out.get("social", 0.0)
    # neuroticism: read more threat, and a darker valence
    out["threat_challenge"] = _clamp(out.get("threat_challenge", 0.0) + g * N, -1.0, 1.0)
    out["valence"] = _clamp(
        out.get("valence", 0.0) - g * N + g * A * max(0.0, social), -1.0, 1.0)
    # openness: more perceived novelty / interest in the new
    out["novelty"] = _clamp(out.get("novelty", 0.0) + g * O, 0.0, 1.0)
    return out


# --- the memory lens (Slice 2): what they remember bends the reading ----------
def _appraisal_dim(memory, dim):
    """How a recalled past episode was *perceived* on one appraisal dimension."""
    return float(((memory or {}).get("appraisal") or {}).get(dim, 0.0))


def _memory_expectation(recalled):
    """Similarity-weighted expectation from the recalled episodes, or ``None``.

    Returns ``{familiarity, valence, threat}`` where ``familiarity`` is the mean cosine
    similarity (how strongly this situation is recognised) and ``valence`` / ``threat`` are
    the score-weighted average of how those past episodes *felt* -- closer memories dominate.
    """
    if not recalled:
        return None
    scores = [max(0.0, float(m.get("score", 0.0))) for m in recalled]
    total = sum(scores)
    if total <= 0.0:
        return None
    weighted = lambda dim: sum(s * _appraisal_dim(m, dim)
                               for s, m in zip(scores, recalled)) / total
    return {
        "familiarity": total / len(scores),
        "valence": weighted("valence"),
        "threat": weighted("threat_challenge"),
    }


def _memory_tilt(out, recalled):
    """Pull ``out`` toward the remembered tone (bounded) and damp novelty by familiarity."""
    exp = _memory_expectation(recalled)
    if exp is None:
        return out
    g = MEMORY_GAIN
    f = exp["familiarity"]
    # learned expectation: a partial blend toward how similar past events felt. Toward a
    # bounded target, so it asymptotes -- it can colour, never overshoot the memory.
    for dim, target in (("valence", exp["valence"]), ("threat_challenge", exp["threat"])):
        cur = out.get(dim, 0.0)
        out[dim] = _clamp(cur + g * f * (target - cur), -1.0, 1.0)
    # familiarity: the more this resembles lived experience, the less novel it reads
    out["novelty"] = _clamp(out.get("novelty", 0.0) * (1.0 - g * f), 0.0, 1.0)
    return out


def interpret(appraisal, personality, recalled=None):
    """Return a copy of ``appraisal`` bent by who the character is and what they remember.

    ``recalled`` is the list of similar past episodes (from ``memory.recall``) for this
    experience; pass ``None`` / ``[]`` for the trait-only lens (offline-safe, Slice 1).
    """
    out = _trait_tilt(dict(appraisal), personality)
    return _memory_tilt(out, recalled)


# --- briefs: the same tilts as words, for the LLM prompt and the UI -----------
def _lens_bits(personality):
    traits = (personality or {}).get("traits") or {}
    N = traits.get("N", 0.0)
    O = traits.get("O", 0.0)
    A = traits.get("A", 0.0)
    bits = []
    if N > _THRESH:
        bits.append("reads situations as threatening and dwells on what could go wrong")
    elif N < -_THRESH:
        bits.append("stays calm and reads situations as manageable")
    if O > _THRESH:
        bits.append("is curious and drawn to the new")
    elif O < -_THRESH:
        bits.append("prefers the familiar and is wary of novelty")
    if A > _THRESH:
        bits.append("reads people warmly and gives the benefit of the doubt")
    elif A < -_THRESH:
        bits.append("reads people warily and is quick to sense a slight")
    return bits


def _trait_brief(bits):
    """The trait lens as a sentence for the LLM push prompt (empty when near-neutral)."""
    if not bits:
        return ""
    return ("How this person tends to perceive events (judge how the experience lands FOR "
            "THEM, not in the abstract): they " + "; ".join(bits) + ".")


def _tone_phrase(exp):
    """A short verb phrase for the remembered tone, or '' when memory is neutral."""
    if exp["threat"] > _TONE_THRESH:
        return "tended to feel threatening"
    if exp["valence"] < -_TONE_THRESH:
        return "tended to go badly"
    if exp["valence"] > _TONE_THRESH:
        return "tended to go well"
    return ""


def _memory_brief(recalled):
    """The memory lens as a sentence for the LLM push prompt (empty when no clear tone)."""
    exp = _memory_expectation(recalled)
    if exp is None:
        return ""
    tone = _tone_phrase(exp)
    if not tone:
        return ""
    return ("This experience resembles things they have lived through before, which for "
            f"them {tone}; weigh how it lands the way someone carrying that memory would.")


def _memory_tag(recalled):
    """The memory lens as a short display phrase (empty when nothing notable surfaced)."""
    exp = _memory_expectation(recalled)
    if exp is None:
        return ""
    if exp["threat"] > _TONE_THRESH:
        return "braced by similar past experiences"
    if exp["valence"] < -_TONE_THRESH:
        return "expecting trouble, from memory"
    if exp["valence"] > _TONE_THRESH:
        return "expecting it to go well, from memory"
    if exp["familiarity"] >= 0.5:
        return "this feels familiar"
    return ""


def lens(personality, recalled=None):
    """A short interpretive brief for the LLM push prompt -- how this person tends to read
    events, from their strong traits and from how similar past experiences felt. Empty when
    nothing tilts the reading (near-neutral character, no relevant memory)."""
    parts = [_trait_brief(_lens_bits(personality)), _memory_brief(recalled)]
    return " ".join(p for p in parts if p)


def read_lens(personality, recalled=None):
    """The cognitive lens as a short display phrase (for the UI). Empty when nothing tilts."""
    bits = _lens_bits(personality)
    tag = _memory_tag(recalled)
    return " · ".join(bits + ([tag] if tag else []))
