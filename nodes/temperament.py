"""Temperament: the biological baseline a character is born with (genesis).

``genesis(bio)`` turns a short biography into a seeded character:

    identity      -- immutable facts (name, origin, religion-raised, ...), the lens
    temperament   -- per-trait OCEAN baseline ``mu`` in [-1, 1] and per-trait
                     stickiness ``tau`` in [0, 1] (how hard biology anchors a trait)
    traits        -- the current OCEAN, born AT the baseline (x = mu)

This is Slice 1: it only *seeds* the baseline. The baseline-as-attractor dynamics
-- the current trait pulled back toward ``mu``, and ``mu`` drifting slowly after a
sustained ``x`` -- are applied in updater.py (Slice 2).

The primary seed comes from an OpenAI-compatible LLM (Google's Gemma 4 by default).
If no API key is set, the ``openai`` package is missing, the network fails, or the
reply is unparseable, ``seed_from_bio`` falls back to a deterministic lexical
heuristic, so genesis never hard-depends on the network -- the same discipline as
llm_impact.py.
"""

import logging

from core.config import BASIS, DEFAULT_TAU, LLM_LABEL
from core.llm import complete_json
from nodes.character import default_character

log = logging.getLogger("mindform.genesis")


GENESIS_PROMPT = """You are MindForm's Genesis Engine.

You convert a short biography into a person's TEMPERAMENT -- the biologically
influenced baseline they are born with (their "default settings"), visible early
in life and tied to brain and nervous-system differences. This is NOT their
current mood, and NOT their learned character or opinions; it is the stable
set-point their personality forms around.

The trait model is OCEAN: O=Openness, C=Conscientiousness, E=Extraversion,
A=Agreeableness, N=Neuroticism.

Return JSON with three parts:

1. "identity": immutable biographical facts, inferred or stated -- name, age,
   gender, origin, culture, language, religion_raised, family, and a one-line
   "summary". Use null for anything genuinely unknown; never invent specifics
   that contradict the biography.

2. "mu": the OCEAN baseline, each a float in [-1.0, 1.0] (0 = average) -- where
   each trait naturally sits in early life.

3. "tau": per-trait stickiness, each a float in [0.0, 1.0] -- how strongly
   biology anchors that trait. High (~0.8) = resilient, snaps back to baseline
   after experience; low (~0.15) = easily reshaped by experience.

Guidance:
* Read for innate temperament -- reactivity, energy, sensitivity -- not
  achievements or beliefs.
* "anxious / sensitive / reactive" -> higher N; "calm / easygoing" -> lower N.
* "outgoing / energetic" -> higher E; "reserved / shy" -> lower E.
* If the biography is sparse, keep baselines near 0 and tau moderate (~0.3);
  do not over-commit.

Return ONLY valid JSON, no markdown, exactly:
{"identity": {...}, "mu": {"O": float, "C": float, "E": float, "A": float, "N": float},
 "tau": {"O": float, "C": float, "E": float, "A": float, "N": float}, "reasoning": "brief"}
"""


def _clamp(value, low, high):
    return max(low, min(high, value))


# --- Deterministic heuristic seed: dependency-free, runs with no network. ---
_TEMP_LEX = {
    "anxious": {"N": 0.5}, "nervous": {"N": 0.4}, "worried": {"N": 0.4},
    "sensitive": {"N": 0.3, "A": 0.1}, "fearful": {"N": 0.4}, "moody": {"N": 0.4},
    "insecure": {"N": 0.4}, "overwhelmed": {"N": 0.3},
    "calm": {"N": -0.4}, "easygoing": {"N": -0.4, "A": 0.2}, "relaxed": {"N": -0.3},
    "stable": {"N": -0.4}, "resilient": {"N": -0.3}, "secure": {"N": -0.3},
    "outgoing": {"E": 0.5}, "social": {"E": 0.4}, "sociable": {"E": 0.4},
    "energetic": {"E": 0.4}, "bold": {"E": 0.3, "N": -0.1}, "talkative": {"E": 0.4},
    "shy": {"E": -0.5}, "reserved": {"E": -0.4}, "quiet": {"E": -0.3},
    "introverted": {"E": -0.5}, "withdrawn": {"E": -0.4, "N": 0.1},
    "sheltered": {"E": -0.2, "O": -0.1},
    "curious": {"O": 0.5}, "creative": {"O": 0.5}, "imaginative": {"O": 0.4},
    "artistic": {"O": 0.4}, "poet": {"O": 0.4}, "adventurous": {"O": 0.3, "E": 0.2},
    "conventional": {"O": -0.3}, "traditional": {"O": -0.3, "C": 0.1},
    "disciplined": {"C": 0.5}, "organized": {"C": 0.5}, "strict": {"C": 0.4},
    "diligent": {"C": 0.4}, "reliable": {"C": 0.3}, "careless": {"C": -0.4},
    "impulsive": {"C": -0.4, "N": 0.1}, "lazy": {"C": -0.4},
    "kind": {"A": 0.4}, "caring": {"A": 0.5}, "warm": {"A": 0.4, "E": 0.1},
    "compassionate": {"A": 0.5}, "gentle": {"A": 0.4}, "cold": {"A": -0.4},
    "harsh": {"A": -0.4}, "aggressive": {"A": -0.4, "N": 0.1},
}


def _guess_name(bio):
    for token in bio.replace(",", " ").split():
        if token[:1].isupper() and token.isalpha():
            return token
    return None


def _heuristic_seed(bio):
    tokens = bio.lower().replace(".", " ").replace(",", " ").split()
    mu = {d: 0.0 for d in BASIS}
    for token in tokens:
        for trait, delta in _TEMP_LEX.get(token, {}).items():
            mu[trait] = _clamp(mu[trait] + delta, -1.0, 1.0)
    return {
        "identity": {"name": _guess_name(bio), "bio": bio.strip()},
        "mu": mu,
        "tau": {d: DEFAULT_TAU for d in BASIS},
        "reasoning": "",
    }


def _llm_seed(bio):
    """Ask the LLM (Gemma 4 by default) to seed identity + mu + tau from the bio.

    Raises on any failure (missing key/package, network, malformed JSON, missing
    or non-numeric trait) so ``seed_from_bio`` can fall back to the heuristic.
    """
    data = complete_json(GENESIS_PROMPT, f"Biography:\n{bio}", temperature=0.3)
    mu = {d: float(data["mu"][d]) for d in BASIS}      # KeyError / ValueError -> fallback
    tau = {d: float(data["tau"][d]) for d in BASIS}
    return {
        "identity": dict(data.get("identity") or {}),
        "mu": mu,
        "tau": tau,
        "reasoning": str(data.get("reasoning", "")),
    }


def seed_from_bio(bio):
    """Best-available seed for a biography. Returns ``(seed, source)``."""
    try:
        return _llm_seed(bio), LLM_LABEL
    except Exception as exc:  # any failure -> deterministic heuristic
        log.info("LLM genesis unavailable (%s); using heuristic seed", exc)
        return _heuristic_seed(bio), "heuristic"


def _finalize(seed, overrides=None):
    """Clamp a seed and build the character dict (traits born at baseline x = mu)."""
    if overrides:
        for key in ("identity", "mu", "tau"):
            if key in overrides:
                seed[key] = {**seed.get(key, {}), **overrides[key]}

    mu = {d: _clamp(float(seed["mu"].get(d, 0.0)), -1.0, 1.0) for d in BASIS}
    tau = {d: _clamp(float(seed["tau"].get(d, DEFAULT_TAU)), 0.0, 1.0) for d in BASIS}
    return {
        "identity": dict(seed.get("identity") or {}),
        "temperament": {"mu": mu, "tau": tau},
        "traits": dict(mu),       # born at baseline: x = mu
        "character": default_character(),   # values start neutral -- earned, not innate
        "experience_count": 0,
    }


def genesis(bio, overrides=None):
    """Birth a character from a free-text biography.

    Returns ``(personality, source, reasoning)``; the current traits are born at
    the baseline (``x = mu``). Pass ``overrides={"mu": {...}, "tau": {...},
    "identity": {...}}`` to hand-edit the seed before it commits (hybrid path).
    """
    seed, source = seed_from_bio(bio)
    return _finalize(seed, overrides), source, seed.get("reasoning", "")


def _compose_bio(identity, background):
    """Assemble a seeding bio from explicit identity fields + a background blurb.

    The identity facts give the model context; the free-text background carries
    the temperament cues (the heuristic lexicon scans it for trait words).
    """
    facts = ", ".join(f"{key}: {value}" for key, value in identity.items() if value)
    if facts and background:
        return f"{facts}. {background}"
    return background or facts


def create_character(fields, overrides=None):
    """Create a character from explicit identity fields + an optional background.

    ``fields`` is a dict of immutable identity values (name, age, origin, religion,
    ...) plus an optional ``"background"`` free-text blurb. The identity is stored
    verbatim and is authoritative; the OCEAN baseline ``mu`` and stickiness ``tau``
    are seeded from the background (LLM, heuristic fallback) with the identity facts
    as context. Returns ``(personality, source, reasoning)``.
    """
    identity = {key: value for key, value in fields.items()
                if key != "background" and value not in (None, "")}
    background = (fields.get("background") or "").strip()

    seed, source = seed_from_bio(_compose_bio(identity, background))
    seed["identity"] = identity      # explicit fields win over anything guessed
    return _finalize(seed, overrides), source, seed.get("reasoning", "")


def build_character(identity, mu, tau=None):
    """Build a character from an explicit identity + an explicitly chosen OCEAN baseline.

    No seeding / LLM -- the caller (e.g. the creation questionnaire) chose ``mu``
    directly. ``tau`` defaults to ``DEFAULT_TAU`` per trait. Traits are born at the
    baseline (``x = mu``). Returns ``(personality, "manual", "")``.
    """
    seed = {
        "identity": dict(identity),
        "mu": {d: float(mu.get(d, 0.0)) for d in BASIS},
        "tau": dict(tau) if tau else {d: DEFAULT_TAU for d in BASIS},
        "reasoning": "",
    }
    return _finalize(seed), "manual", ""
