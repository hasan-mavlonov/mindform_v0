"""COGNITIVE PATTERNS: the interpretation lens between perception and formation.

How a person *thinks* colours what they perceive. The engine appraises an experience the
same way for everyone (``appraisal.appraise``); this node bends that reading by who the
character currently is, so the SAME event forms two different people differently:

  * ``interpret(appraisal, personality)`` -- gently tilts the appraisal vector by the
    current OCEAN traits (anxious -> more perceived threat and a darker valence; open ->
    more novelty; agreeable -> warmer on social events). Deterministic; it colours the
    heuristic push (traits / values / moral all run through ``appraise -> impact``).
  * ``lens(personality)`` -- the same tilt as a one-line interpretive brief injected into
    the LLM push prompt, so the primary (Gemma) path is personality-coloured too (it reads
    the raw text, not the appraisal). ``read_lens`` is the short version for display.

The bias is small (``config.COGNITION_GAIN``) and clamped, so it tints perception without
overwhelming the event's real meaning; temperament's pull-back keeps the perceive ->
form -> perceive loop stable. (Memory- and schema-driven interpretation are a later slice.)
"""

from core.config import COGNITION_GAIN

_THRESH = 0.25   # how far from neutral a trait must be to colour the lens


def _clamp(value, low, high):
    return max(low, min(high, value))


def interpret(appraisal, personality):
    """Return a copy of ``appraisal`` tilted by the character's current OCEAN traits."""
    traits = (personality or {}).get("traits") or {}
    N = traits.get("N", 0.0)
    O = traits.get("O", 0.0)
    A = traits.get("A", 0.0)
    g = COGNITION_GAIN

    out = dict(appraisal)
    social = out.get("social", 0.0)
    # neuroticism: read more threat, and a darker valence
    out["threat_challenge"] = _clamp(out.get("threat_challenge", 0.0) + g * N, -1.0, 1.0)
    out["valence"] = _clamp(
        out.get("valence", 0.0) - g * N + g * A * max(0.0, social), -1.0, 1.0)
    # openness: more perceived novelty / interest in the new
    out["novelty"] = _clamp(out.get("novelty", 0.0) + g * O, 0.0, 1.0)
    return out


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


def lens(personality):
    """A one-line interpretive brief for the LLM push prompt -- how this person tends to
    perceive events, from their strong traits. Empty string when near-neutral (no tilt)."""
    bits = _lens_bits(personality)
    if not bits:
        return ""
    return ("How this person tends to perceive events (judge how the experience lands FOR "
            "THEM, not in the abstract): they " + "; ".join(bits) + ".")


def read_lens(personality):
    """The cognitive lens as a short display phrase (for the UI). Empty when near-neutral."""
    return " · ".join(_lens_bits(personality))
