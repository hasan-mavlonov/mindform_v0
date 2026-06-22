"""Meaning extraction: experience text -> appraisal vector.

The appraisal vector is the experience representation that drives personality
formation -- the causal ingredients of change (valence, intensity, control,
outcome, ...), NOT a trait-expression reading. See ``config.APPRAISAL_SCHEMA``.

Resolution order:
    1. learned affect/appraisal head (appraisal_head.pth) if it has been trained
    2. heuristic lexicon extractor                       (always available)

There is deliberately no LLM/API dependency. The learned head is trained on
existing cross-sectional affect/appraisal corpora (see ``bootstrap/``), using a
frozen MiniLM-embedding + small regression-head recipe.
"""


def appraise(text):
    head = _try_head()
    if head is not None:
        return head(text)
    return _heuristic_appraise(text)


def _try_head():
    """Return the learned head's predict fn, or None if unavailable/untrained."""
    try:
        from core.appraisal_head import predict_appraisal, load_head
        return predict_appraisal if load_head() is not None else None
    except Exception:
        return None


# --- Zero-dependency heuristic extractor: runs today, no model, no network. ---
_POS = {"love", "great", "happy", "fun", "calm", "proud", "safe", "enjoy",
        "good", "win", "fine", "joy", "excited", "glad", "relieved"}
_NEG = {"terrified", "afraid", "scared", "anxious", "hate", "sad", "fail",
        "alone", "panic", "bad", "hurt", "lonely", "angry", "ashamed"}
_SOCIAL = {"people", "friend", "friends", "party", "social", "event", "events",
           "together", "crowd", "meet", "group", "team", "everyone"}
_AGENCY = {"chose", "decided", "started", "tried", "faced", "made", "built",
           "took", "pushed", "confronted", "handled"}


def _heuristic_appraise(text):
    tokens = set(text.lower().replace(".", " ").replace(",", " ").split())
    pos = len(tokens & _POS)
    neg = len(tokens & _NEG)
    soc = len(tokens & _SOCIAL)
    valence = (pos - neg) / max(1, pos + neg)

    return {
        "valence": valence,
        "intensity": 0.5 if (pos + neg) else 0.3,
        "novelty": 0.3,
        "agency": 0.4 if tokens & _AGENCY else 0.0,
        "social": 1.0 if soc else -0.2,
        "outcome": 0.5 * valence,
        "self_relevance": 0.6 if "i" in tokens else 0.3,
        "threat_challenge": valence,
    }
