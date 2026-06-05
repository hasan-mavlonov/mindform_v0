"""Meaning extraction: experience text -> appraisal vector (deterministic fallback).

The appraisal vector is the experience representation behind the fallback push
(impact.py) -- the causal ingredients of change (valence, intensity, control,
outcome, ...), NOT a trait-expression reading. It is used only when the DeepSeek
push (llm_impact.py) is unavailable, so this module stays a zero-dependency lexicon
extractor: no model, no network.
"""

# --- Word lists for the zero-dependency heuristic. ---
_POS = {"love", "great", "happy", "fun", "calm", "proud", "safe", "enjoy",
        "good", "win", "fine", "joy", "excited", "glad", "relieved"}
_NEG = {"terrified", "afraid", "scared", "anxious", "hate", "sad", "fail",
        "alone", "panic", "bad", "hurt", "lonely", "angry", "ashamed"}
_SOCIAL = {"people", "friend", "friends", "party", "social", "event", "events",
           "together", "crowd", "meet", "group", "team", "everyone"}
_AGENCY = {"chose", "decided", "started", "tried", "faced", "made", "built",
           "took", "pushed", "confronted", "handled"}


def appraise(text):
    """Heuristic appraisal of an experience -> the appraisal dims (no model/network)."""
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
