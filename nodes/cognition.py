"""COGNITIVE PATTERNS: the interpretation lens between perception and formation.

How a person *thinks* colours what they perceive. The engine appraises an experience the
same way for everyone (``appraisal.appraise``); this node bends that reading by who the
character currently is AND by what they remember, so the SAME event forms two different
people differently -- and lands differently on the same person depending on their past.

  * ``interpret(appraisal, personality, recalled, recalled_beliefs)`` -- tilts the
    appraisal vector:
      - by current OCEAN traits (Slice 1): anxious -> more perceived threat and a darker
        valence; open -> more novelty; agreeable -> warmer on social events.
      - by EPISODIC memory (Slice 2): recalled similar past episodes pull the valence and
        threat *toward how those felt* (a learned expectation / schema), and the more this
        resembles lived experience, the less novel it reads (familiarity damps novelty).
      - by SEMANTIC memory: recalled beliefs damp novelty and raise self-relevance by how
        firmly they're held -- "I already have a theory about this" reads as less
        surprising and more personally relevant, regardless of whether the belief itself
        is optimistic or grim (deliberately not a valence pull -- see ``_belief_tilt``).
    Deterministic; it colours the heuristic push (traits / values / moral all run through
    ``appraise -> impact``).
  * ``lens(personality, recalled, recalled_beliefs)`` -- the same tilts as an interpretive
    brief injected into the LLM push prompt, so the primary (Gemma) path is coloured too
    (it reads the raw text, not the appraisal). ``read_lens`` is the short version for
    display.

All biases are small (``config.COGNITION_GAIN`` / ``config.MEMORY_GAIN`` /
``config.BELIEF_TILT_GAIN``) and clamped, so perception is tinted, not overwhelmed. The
memory and belief pulls are *toward a bounded target* (each asymptotes once the reading
matches it) and temperament's pull-back keeps the perceive -> form -> perceive loop
stable, so the schema reinforces without running away.
"""

from core.config import (
    COGNITION_GAIN, MEMORY_GAIN, BELIEF_TILT_GAIN, DRIVE_GAIN, DRIVES, DRIVE_SAT,
    DRIVE_ACTIVE_THRESH, BASIS, M, SELF_GAIN, SELF_ESTEEM_GAIN, SELF_ACTIVE_THRESH,
    SELF_INCONGRUENCE_THRESH, SELF_OUGHT_GAIN, SELF_GAP_THRESH, BEHAV_EXPOSURE,
)
from core.impact import rule_pull
from nodes.self_concept import ought_gap, aspiration_gap

_THRESH = 0.25       # how far from neutral a trait must be to colour the lens
_TONE_THRESH = 0.20  # how clearly the remembered tone must lean to surface a brief / tag
_CONVICTION_THRESH = 0.30  # how firmly-held x relevant a recalled belief must be to surface


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
    # neuroticism: read more threat, and a darker valence. NOTE the sign: config defines
    # threat_challenge as -1 = threat/loss, +1 = challenge/growth, so "more threat" means
    # pushing it *negative* (matching M's N row and the SE/security value).
    out["threat_challenge"] = _clamp(out.get("threat_challenge", 0.0) - g * N, -1.0, 1.0)
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


# --- the belief lens (SEMANTIC memory): what they believe bends the reading ---
def _belief_expectation(recalled_beliefs):
    """Similarity-weighted expectation from the recalled beliefs, or ``None``.

    Returns ``{familiarity, conviction}`` -- ``familiarity`` is the mean recall relevance
    (how strongly this situation matches a belief the character holds) and ``conviction``
    is the score-weighted mean of how firmly those beliefs are held (``|confidence|``).
    Deliberately NO valence/direction here -- see ``_belief_tilt``.
    """
    if not recalled_beliefs:
        return None
    scores = [max(0.0, float(b.get("score", 0.0))) for b in recalled_beliefs]
    total = sum(scores)
    if total <= 0.0:
        return None
    conviction = sum(s * abs(float(b.get("confidence", 0.0)))
                     for s, b in zip(scores, recalled_beliefs)) / total
    return {"familiarity": total / len(scores), "conviction": conviction}


def _belief_tilt(out, recalled_beliefs):
    """A relevant, firmly-held belief damps novelty and raises self-relevance -- bounded,
    multiplicative, the direction-agnostic counterpart to ``_memory_tilt``'s familiarity
    damp. NOT a valence pull: ``confidence`` measures how strongly a proposition is held,
    not whether its content is good or bad news -- "the world is dangerous" held at
    confidence +0.9 is a strongly CONFIRMED belief with THREATENING content; reading its
    sign as brightness would tilt exactly backwards. "This is already categorized" reads
    as less surprising and more personally about you either way, which is the effect
    modelled here; a directional version would need a second, separate signal (the belief
    statement's own content polarity) and is deliberately out of scope."""
    exp = _belief_expectation(recalled_beliefs)
    if exp is None:
        return out
    g = BELIEF_TILT_GAIN
    f, c = exp["familiarity"], exp["conviction"]
    out["novelty"] = _clamp(out.get("novelty", 0.0) * (1.0 - g * f * c), 0.0, 1.0)
    out["self_relevance"] = _clamp(out.get("self_relevance", 0.0) + g * f * c, 0.0, 1.0)
    return out


# --- the drive lens (Motivation): what they want bends the reading -------------
def _drive_tilt(out, personality):
    """Tilt ``out`` by the character's currently-active needs (``personality["drives"]``).

    An event that bears on a loud need reads as more *self-relevant* (real goal-relevance,
    replacing the crude proxy -- so it forms the character harder via salience); one that
    MEETS an active need reads warmer, one that DENIES it reads as more threat. The tilt is
    the tension-weighted engagement of the needs, small and clamped (``DRIVE_GAIN``).
    """
    drives = (personality or {}).get("drives") or {}
    tension = {d: float((drives.get(d) or {}).get("tension", 0.0)) for d in DRIVES}
    if sum(tension.values()) <= 0.0:
        return out                                   # no active need -> no goal-relevance
    signal = rule_pull(out, DRIVES, DRIVE_SAT)       # signed per-need engagement of this reading
    # tension-weighted SUM (not mean): a LOUDER need bends the reading harder -- a starved
    # need makes the event land harder. Clamped so it saturates rather than runs away.
    signed = _clamp(sum(tension[d] * signal[d] for d in DRIVES), -1.0, 1.0)     # + meets / - denies
    relevance = _clamp(sum(tension[d] * abs(signal[d]) for d in DRIVES), 0.0, 1.0)
    g = DRIVE_GAIN
    out["self_relevance"] = _clamp(out.get("self_relevance", 0.0) + g * relevance, 0.0, 1.0)
    out["valence"] = _clamp(out.get("valence", 0.0) + g * signed, -1.0, 1.0)
    # denying an active need adds threat (threat_challenge is signed, negative = threat)
    out["threat_challenge"] = _clamp(
        out.get("threat_challenge", 0.0) + g * min(0.0, signed), -1.0, 1.0)
    return out


# --- the self lens (Self-Concept): who they think they are bends the reading ---
def _self_tilt(out, personality):
    """Tilt ``out`` by the character's self-concept (``personality["self"]``).

    Two effects, small and clamped: (1) SELF-CONSISTENCY -- project the reading onto the traits
    it implies and compare to the self-image; an event that CONTRADICTS who they think they are
    reads as more threat and more self-relevant (dissonance), an affirming one warms. (2) ESTEEM
    BUFFER -- high self-regard reads events as surmountable challenges, low regard as threats.
    """
    # agitation (Higgins): falling short of who they OUGHT to be keeps them vigilant -- the
    # reading tilts toward threat in proportion to the ought-gap. Before the no-self guard
    # on purpose: a blank self-image against a formed ought is the maximal falling-short.
    og = ought_gap(personality)
    if og > 0.0:
        out["threat_challenge"] = _clamp(
            out.get("threat_challenge", 0.0) - SELF_OUGHT_GAIN * og, -1.0, 1.0)

    self_state = (personality or {}).get("self") or {}
    image = self_state.get("image") or {}
    esteem = float(self_state.get("esteem", 0.0))
    img = {k: float(image.get(k, 0.0)) for k in BASIS}
    if sum(abs(v) for v in img.values()) + abs(esteem) <= 0.0:
        return out                                    # no self-view yet -> no image/esteem lens
    # self-consistency
    implied = rule_pull(out, BASIS, M)                # the trait direction this event implies
    align = _clamp(sum(img[k] * implied[k] for k in BASIS), -1.0, 1.0)      # + affirms / - contradicts
    relevance = _clamp(sum(abs(img[k]) * abs(implied[k]) for k in BASIS), 0.0, 1.0)
    g = SELF_GAIN
    out["self_relevance"] = _clamp(out.get("self_relevance", 0.0) + g * relevance, 0.0, 1.0)
    out["valence"] = _clamp(out.get("valence", 0.0) + g * align, -1.0, 1.0)
    out["threat_challenge"] = _clamp(out.get("threat_challenge", 0.0) + g * min(0.0, align), -1.0, 1.0)
    # esteem buffer (threat_challenge is signed, +challenge/-threat)
    ge = SELF_ESTEEM_GAIN
    out["threat_challenge"] = _clamp(out.get("threat_challenge", 0.0) + ge * esteem, -1.0, 1.0)
    out["valence"] = _clamp(out.get("valence", 0.0) + ge * esteem, -1.0, 1.0)
    return out


# --- the behavior gate (Behavior): the stance they carry bends how hard life lands ---
def _behavior_tilt(out, personality):
    """The intake gate -- the door before the tint. The action stance carried from last
    turn multiplies this experience's INTENSITY: leaning in exposes the character to life
    (events land harder and form more, via salience); holding back muffles it (the shy
    spiral). Intensity is behavior's private channel -- no other tilt touches it -- so the
    gate stays exactly attributable. Memoryless and bounded in [1-EXPOSURE, 1+EXPOSURE].
    """
    behavior = (personality or {}).get("behavior") or {}
    tendency = float((behavior.get("set") or {}).get("tendency", 0.0))
    if tendency == 0.0:
        return out
    factor = 1.0 + BEHAV_EXPOSURE * _clamp(tendency, -1.0, 1.0)
    out["intensity"] = _clamp(out.get("intensity", 0.0) * factor, 0.0, 1.0)
    return out


def interpret(appraisal, personality, recalled=None, recalled_beliefs=None):
    """Return a copy of ``appraisal`` bent by how engaged they are with the world, who they
    are, what they remember (lived and believed), what they currently want, and who they
    think they are.

    ``recalled`` is the list of similar past EPISODES (from ``memory.recall``);
    ``recalled_beliefs`` is the list of relevant held BELIEFS (from
    ``belief_memory.recall_beliefs``) -- pass ``None`` / ``[]`` for either to skip that
    lens (offline-safe, and exactly today's behavior when belief recall isn't wired up).
    The behavior, drive, and self tilts read their ``personality`` keys and are no-ops
    when those are blank.
    """
    out = _behavior_tilt(dict(appraisal), personality)   # the door: how much gets in
    out = _trait_tilt(out, personality)
    out = _memory_tilt(out, recalled)
    out = _belief_tilt(out, recalled_beliefs)
    out = _drive_tilt(out, personality)
    return _self_tilt(out, personality)


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
    """A short verb phrase for the remembered tone, or '' when memory is neutral.
    ``exp["threat"]`` is the threat_challenge dim: negative = threat/loss (config convention)."""
    if exp["threat"] < -_TONE_THRESH:
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
    if exp["threat"] < -_TONE_THRESH:        # negative threat_challenge = threat/loss
        return "braced by similar past experiences"
    if exp["valence"] < -_TONE_THRESH:
        return "expecting trouble, from memory"
    if exp["valence"] > _TONE_THRESH:
        return "expecting it to go well, from memory"
    if exp["familiarity"] >= 0.5:
        return "this feels familiar"
    return ""


def _belief_brief(recalled_beliefs):
    """The belief lens as a sentence for the LLM push prompt (empty when nothing firmly
    held and relevant surfaced). Deliberately doesn't say WHAT the belief is or which way
    it leans -- only that the situation is one they already have a settled view on."""
    exp = _belief_expectation(recalled_beliefs)
    if exp is None or exp["familiarity"] * exp["conviction"] < _CONVICTION_THRESH:
        return ""
    return ("This is the kind of thing they already have a settled view on; it reads as "
            "less of a surprise and more clearly about them because of it.")


def _belief_tag(recalled_beliefs):
    """The belief lens as a short display phrase (empty when nothing notable surfaced)."""
    exp = _belief_expectation(recalled_beliefs)
    if exp is None or exp["familiarity"] * exp["conviction"] < _CONVICTION_THRESH:
        return ""
    return "this matches something they believe"


# the loudest active need, as words for the LLM prompt and the UI
_DRIVE_NEED_PHRASE = {
    "autonomy": "freedom to act on their own terms",
    "competence": "a sense of mastery and getting it right",
    "relatedness": "connection and belonging",
}
_DRIVE_TAG = {
    "autonomy": "needs autonomy",
    "competence": "needs to prove themselves",
    "relatedness": "needs connection",
}


def _active_need(personality):
    """The loudest need, but only if it is loud enough to matter (``DRIVE_ACTIVE_THRESH``)."""
    drives = (personality or {}).get("drives") or {}
    if not drives:
        return None
    top = max(DRIVES, key=lambda d: float((drives.get(d) or {}).get("tension", 0.0)))
    if float((drives.get(top) or {}).get("tension", 0.0)) < DRIVE_ACTIVE_THRESH:
        return None
    return top


def _drive_brief(personality):
    """The active need as a sentence for the LLM push prompt (empty when nothing is pressing)."""
    need = _active_need(personality)
    if not need:
        return ""
    return (f"Right now they most need {_DRIVE_NEED_PHRASE[need]}; weigh how much this "
            "experience meets or denies that.")


def _self_incongruent(personality):
    """True when the self-image has drifted meaningfully from the actual traits (self-deception)."""
    self_state = (personality or {}).get("self") or {}
    img = self_state.get("image") or {}
    if not img:
        return False
    traits = (personality or {}).get("traits") or {}
    incong = sum(abs(float(img.get(k, 0.0)) - float(traits.get(k, 0.0))) for k in BASIS) / len(BASIS)
    return incong > SELF_INCONGRUENCE_THRESH


def _self_bits(personality):
    """Short self-view phrases for the UI tag (self-regard level + self-image incongruence)."""
    self_state = (personality or {}).get("self") or {}
    if not self_state:
        return []
    bits = []
    esteem = float(self_state.get("esteem", 0.0))
    if esteem > SELF_ACTIVE_THRESH:
        bits.append("secure in themselves")
    elif esteem < -SELF_ACTIVE_THRESH:
        bits.append("shaky sense of self")
    if _self_incongruent(personality):
        bits.append("self-image lagging who they've become")
    if aspiration_gap(personality) > SELF_GAP_THRESH:
        bits.append("falling short of who they want to be")
    if ought_gap(personality) > SELF_GAP_THRESH:
        bits.append("weighed by who they ought to be")
    return bits


def _self_brief(personality):
    """The self-view as a sentence for the LLM push prompt (empty when nothing stands out)."""
    self_state = (personality or {}).get("self") or {}
    if not self_state:
        return ""
    esteem = float(self_state.get("esteem", 0.0))
    parts = []
    if esteem > SELF_ACTIVE_THRESH:
        parts.append("they carry a steady sense of their own worth and meet setbacks as challenges")
    elif esteem < -SELF_ACTIVE_THRESH:
        parts.append("they doubt their own worth and take setbacks personally")
    if _self_incongruent(personality):
        parts.append("their self-image lags who they have become, and they resist revising it")
    if aspiration_gap(personality) > SELF_GAP_THRESH:
        parts.append("they feel they are falling short of who they want to be")
    if ought_gap(personality) > SELF_GAP_THRESH:
        parts.append("they carry the weight of who they think they ought to be")
    return ("How they see themselves: " + "; ".join(parts) + ".") if parts else ""


_BEHAVIOR_BRIEF = {
    "approach": ("Lately they lean into things; judge how hard this lands for someone "
                 "reaching toward the world."),
    "withdraw": ("Lately they have been holding back; judge how hard this lands for someone "
                 "keeping the world at arm's length."),
    "conflicted": "They are pulled two ways right now -- drawn in and braced at once.",
}


def _behavior_brief(personality):
    """The carried stance as a sentence for the LLM push prompt (empty when steady)."""
    behavior = (personality or {}).get("behavior") or {}
    mode = (behavior.get("set") or {}).get("mode", "steady")
    return _BEHAVIOR_BRIEF.get(mode, "")


def lens(personality, recalled=None, recalled_beliefs=None):
    """A short interpretive brief for the LLM push prompt -- how this person tends to read events,
    from their carried stance, their strong traits, how similar past experiences felt, what they
    already believe about situations like this, what they currently want, and who they think they
    are. Empty when nothing tilts the reading."""
    parts = [_behavior_brief(personality),
             _trait_brief(_lens_bits(personality)),
             _memory_brief(recalled),
             _belief_brief(recalled_beliefs),
             _drive_brief(personality),
             _self_brief(personality)]
    return " ".join(p for p in parts if p)


def read_lens(personality, recalled=None, recalled_beliefs=None):
    """The cognitive lens as a short display phrase (for the UI). Empty when nothing tilts."""
    bits = _lens_bits(personality)
    tag = _memory_tag(recalled)
    belief_tag = _belief_tag(recalled_beliefs)
    need = _active_need(personality)
    extras = (([tag] if tag else []) + ([belief_tag] if belief_tag else [])
              + ([_DRIVE_TAG[need]] if need else []) + _self_bits(personality))
    return " · ".join(bits + extras)
