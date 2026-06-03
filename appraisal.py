"""Meaning extraction: experience text -> appraisal vector.

The appraisal vector is the experience representation that drives personality
formation -- the causal ingredients of change (valence, intensity, control,
outcome, ...), NOT a trait-expression reading. See ``config.APPRAISAL_SCHEMA``.

Raw extraction (text -> appraisal), in resolution order:
    1. learned affect/appraisal head (appraisal_head.pth) if it has been trained
    2. heuristic lexicon extractor                       (always available)

Two tints are then applied on top, so the same text is read differently depending
on context and on the agent (both deterministic, no LLM):
    - blend_appraisal : "this reminds me of ..."  (memory-conditioned)
    - bias_appraisal  : "and I'm the kind of agent who ..."  (personality-conditioned)

There is deliberately no LLM/API dependency. The learned head is trained on
existing cross-sectional affect/appraisal corpora (see ``bootstrap/``), using a
MiniLM-embedding + small-regression-head recipe.
"""


from config import RETRIEVAL_ALPHA, APPRAISAL_BIAS


def _clamp(value, minimum=-1.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def appraise(text):
    head = _try_head()
    if head is not None:
        return head(text)
    return _heuristic_appraise(text)


def blend_appraisal(raw, neighbors, alpha=RETRIEVAL_ALPHA):
    """Read an experience partly through the appraisals of similar remembered ones.

    Returns a convex blend of the ``raw`` appraisal and the mean of ``neighbors``
    (past appraisal dicts). With no neighbors it is the raw appraisal unchanged, so
    the first time something happens it is appraised on its own terms; only once a
    history exists does memory tint the reading ("this reminds me of ..."). The blend
    is a convex combination, so every value stays within the existing [-1, 1] range.
    """
    if not neighbors:
        return dict(raw)
    blended = {}
    for key, value in raw.items():
        mean = sum(n.get(key, 0.0) for n in neighbors) / len(neighbors)
        blended[key] = (1 - alpha) * value + alpha * mean
    return blended


def bias_appraisal(appraisal, personality):
    """Color the raw appraisal by WHO THE AGENT IS (slow traits) and HOW IT FEELS NOW
    (fast mood). The same text becomes a different *experience* for different agents,
    so the resulting nudge -- and even its direction -- depends on who he was.

    Three legible couplings (strengths in config.APPRAISAL_BIAS):
      1. high Neuroticism reads a negative event as more threatening (lowers
         threat_challenge), so adversity raises N more for an already-anxious agent;
      2. high Openness reads novelty as more positive (raises valence);
      3. mood-congruence: the agent's current mood tints valence -- in a good mood it
         reads the next event a little more positively, in a bad mood more negatively.

    Returns a new appraisal dict; every value stays within its [-1, 1] / [0, 1] range.
    With a blank-slate agent (all traits and mood 0) this is the identity, so a fresh
    agent appraises events on their own terms.
    """
    traits = personality["traits"]
    o = traits["O"]["trait"]
    n = traits["N"]["trait"]
    # mood proxy: positive affect = extraverted-up and neurotic-down right now
    mood_valence = traits["E"]["state"] - traits["N"]["state"]

    a = dict(appraisal)
    negativity = max(0.0, -a.get("valence", 0.0))
    # 1. neuroticism -> threat bias on negative events
    a["threat_challenge"] = _clamp(
        a.get("threat_challenge", 0.0)
        - APPRAISAL_BIAS["neuroticism_threat"] * max(0.0, n) * negativity
    )
    # 2. openness -> novelty positivity
    valence = a.get("valence", 0.0) + APPRAISAL_BIAS["openness_novelty"] * max(0.0, o) * a.get("novelty", 0.0)
    # 3. mood-congruence
    valence += APPRAISAL_BIAS["mood_congruence"] * mood_valence
    a["valence"] = _clamp(valence)
    return a


def _try_head():
    """Return the learned head's predict fn, or None if unavailable/untrained."""
    try:
        from appraisal_head import predict_appraisal, load_head
        return predict_appraisal if load_head() is not None else None
    except Exception:
        return None


# --- Zero-dependency heuristic extractor: runs today, no model, no network. ---
# Each appraisal dimension is read from its OWN lexicon so the dimensions stay
# INDEPENDENT. The previous version hardcoded ``outcome = 0.5*valence``,
# ``threat_challenge = valence`` and a constant ``novelty = 0.3``, which silently
# collapsed the 8-D schema onto ~2 axes (valence + social) and turned Openness
# into a valence sensor instead of a novelty sensor. Keeping the dimensions
# genuinely distinct is what lets the appraisal -> trait matrix M do its job, and
# it also gives the future learned head non-degenerate weak labels to imitate.
_POS = {"love", "great", "happy", "fun", "calm", "proud", "safe", "enjoy",
        "enjoyed", "good", "fine", "joy", "excited", "glad", "relieved",
        "grateful", "content", "peaceful", "wonderful"}
_NEG = {"terrified", "afraid", "scared", "anxious", "hate", "sad", "panic",
        "bad", "hurt", "lonely", "angry", "ashamed", "miserable", "upset",
        "worried", "stressed", "hopeless"}
_SOCIAL = {"people", "friend", "friends", "party", "social", "event", "events",
           "together", "crowd", "meet", "met", "group", "team", "everyone",
           "colleagues", "family", "guests"}
_ISOLATION = {"alone", "lonely", "isolated", "solitude", "withdrew", "secluded"}
_AGENCY = {"chose", "decided", "started", "tried", "faced", "made", "built",
           "took", "pushed", "confronted", "handled", "planned", "led",
           "organized", "initiated"}
_HELPLESS = {"helpless", "trapped", "powerless", "forced", "stuck", "unable",
             "overwhelmed", "couldn't", "cannot"}
_NOVELTY = {"new", "first", "never", "unfamiliar", "strange", "unknown",
            "discovered", "explore", "explored", "fascinating", "curious",
            "surprising", "unexpected", "novel", "different", "wonder",
            "experiment", "experimented", "quantum"}
_OUTCOME_POS = {"won", "passed", "finished", "completed", "succeeded",
                "achieved", "accomplished", "solved", "aced", "nailed", "built"}
_OUTCOME_NEG = {"failed", "lost", "quit", "broke", "missed", "rejected",
                "flunked", "abandoned"}
_THREAT = {"terrified", "afraid", "scared", "fear", "danger", "dangerous",
           "threat", "threatening", "panic", "trapped", "helpless", "unsafe",
           "dread", "attacked", "threatened"}
_CHALLENGE = {"faced", "challenge", "challenging", "overcame", "confronted",
              "tackled", "persevered", "mastered", "tried", "pushed"}
_INTENSIFIERS = {"very", "extremely", "incredibly", "so", "really", "utterly",
                 "completely", "absolutely", "deeply", "totally"}
_SELF = {"i", "me", "my", "myself", "mine", "i'm", "i've", "we", "our"}


def _clamp(value, minimum=-1.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def _heuristic_appraise(text):
    lower = text.lower()
    tokens = lower.replace(".", " ").replace(",", " ").split()

    def n(lexicon):
        return sum(1 for t in tokens if t in lexicon)

    pos, neg = n(_POS), n(_NEG)
    valence = (pos - neg) / (pos + neg) if (pos + neg) else 0.0

    soc, iso = n(_SOCIAL), n(_ISOLATION)
    social = _clamp(0.9 * soc - 0.9 * iso) if (soc or iso) else -0.1

    return {
        # emotional sign of the event
        "valence": valence,
        # arousal: more sentiment words / intensifiers / "!" -> more intense
        "intensity": _clamp(
            0.35 + 0.15 * (pos + neg) + 0.2 * n(_INTENSIFIERS) + 0.2 * lower.count("!"),
            0.2, 1.0,
        ),
        # novelty is now its OWN signal (was a constant) -> drives Openness
        "novelty": _clamp(0.15 + 0.35 * n(_NOVELTY), 0.0, 1.0),
        # sense of control: agentic verbs raise it, helplessness lowers it
        "agency": _clamp(0.4 * n(_AGENCY) - 0.5 * n(_HELPLESS)),
        # social engagement (+) vs isolation (-)
        "social": social,
        # achievement/failure, INDEPENDENT of overall valence
        "outcome": _clamp(0.6 * n(_OUTCOME_POS) - 0.6 * n(_OUTCOME_NEG)),
        # autobiographical self-reference
        "self_relevance": _clamp(0.3 + 0.2 * n(_SELF), 0.0, 1.0),
        # threat (-) vs challenge/growth (+), INDEPENDENT of overall valence
        "threat_challenge": _clamp(0.6 * n(_CHALLENGE) - 0.6 * n(_THREAT)),
    }
