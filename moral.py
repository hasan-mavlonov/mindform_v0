"""Experience text -> signed Moral Foundations delta via an LLM, heuristic fallback.

The MORAL-OUTLOOK half of CHARACTER. Where ``values.py`` reads an experience for how it
pushes the Schwartz *values*, this reads the SAME experience for how it pushes the six
Moral Foundations (Haidt) -- which moral concerns one occurrence teaches the person to
hold more, or less. The contract mirrors ``values.py`` exactly: an OpenAI-compatible
model (Google's Gemma 4 by default) returns a signed delta in [-1, 1] per foundation for
a SINGLE occurrence; ``character.update_moral`` then applies it with the same diminishing
returns. The engine accumulates repetition itself, so the prompt asks about one occurrence.

If the LLM is unavailable -- no key (``GEMINI_API_KEY``), no ``openai`` package, a network
error, or an unparseable reply -- it falls back to the deterministic
``impact(appraise(text), MORAL, MORAL_M)`` heuristic, so moral formation, like trait and
values formation, never hard-depends on the network.
"""

import logging

from config import (
    MORAL, MORAL_M, LLM_FORMATION_RATE, LLM_LABEL, LLM_MODEL, LLM_BASE_URL,
    LLM_API_KEY, parse_json_object,
)
from appraisal import appraise
from impact import impact, clamp

log = logging.getLogger("mindform.moral")

MORAL_SYSTEM_PROMPT = """You are MindForm's Moral-Formation Engine.

Your task is to estimate the directional pressure a SINGLE occurrence of the described
experience exerts on each of a person's MORAL FOUNDATIONS -- which moral concerns it
teaches them to hold more strongly, and which less. Judge ONE occurrence; the engine
accumulates repetition and applies diminishing returns over time.

This is moral FORMATION, not detection: read the experience's meaning, not the current
personality of whoever wrote it.

The model is Haidt's six Moral Foundations:

CARE  = Care / Harm (compassion; protecting others from suffering)
FAIR  = Fairness / Cheating (justice, reciprocity, proportionality)
LOYAL = Loyalty / Betrayal (standing with one's group, family, team, nation)
AUTH  = Authority / Subversion (respect for hierarchy, tradition, duty, order)
SANC  = Sanctity / Degradation (purity, the sacred, self-discipline, disgust)
LIB   = Liberty / Oppression (resisting domination and coercion; valuing freedom)

For each foundation, estimate a delta in the range [-1.0, 1.0]:

+1.0 = one occurrence most strongly teaches the person to prize this concern
 0.0 = little or no effect
-1.0 = one occurrence most strongly pushes the person away from this concern

Important:
* Judge the EXPERIENCE, not the author.
* One experience may move several foundations at once, some up and some down (punishing a
  rule-breaker can raise Authority and Fairness while lowering Care).
* Neutral experiences should produce values near zero.
* Do not think out loud or emit any <thought>/<thinking> block before the JSON --
  put any explanation in the "reasoning" field, nowhere else.

Examples:

Experience:
"I broke a pointless rule to protect a friend from an unfair punishment."

Output:
{"CARE": 0.5, "FAIR": 0.4, "LOYAL": 0.4, "AUTH": -0.5, "SANC": 0.0, "LIB": 0.4, "reasoning": "put a person and fairness above rule-following"}

Experience:
"I kept the fast and every ritual exactly as my faith prescribes."

Output:
{"CARE": 0.0, "FAIR": 0.0, "LOYAL": 0.2, "AUTH": 0.4, "SANC": 0.7, "LIB": -0.1, "reasoning": "honored the sacred and religious order"}

Return ONLY valid JSON, with no markdown and no extra text, in exactly this format:

{"CARE": float, "FAIR": float, "LOYAL": float, "AUTH": float, "SANC": float, "LIB": float, "reasoning": "brief explanation"}
"""


def _llm_moral_delta(text):
    """Ask the LLM for the signed Moral Foundations delta of one occurrence of ``text``.

    Returns ``{CARE..LIB: float, "reasoning": str}``. Raises on any failure (missing
    key/package, network error, malformed JSON, missing/non-numeric foundation) so
    ``moral_push_from_text`` can fall back to the heuristic.
    """
    if not LLM_API_KEY:
        raise RuntimeError("no LLM API key is set (GEMINI_API_KEY)")

    from openai import OpenAI  # lazy: the heuristic fallback works without this package

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": MORAL_SYSTEM_PROMPT},
            {"role": "user", "content": f"Experience:\n{text}"},
        ],
        temperature=0.2,
        max_tokens=500,
        timeout=30,
    )
    data = parse_json_object(completion.choices[0].message.content)
    delta = {dim: float(data[dim]) for dim in MORAL}  # KeyError / ValueError -> fallback
    delta["reasoning"] = str(data.get("reasoning", ""))
    return delta


def moral_push_from_text(text, appraisal=None):
    """Best-available signed per-foundation push for an experience.

    Mirrors ``values.values_push_from_text``: tries the LLM (``text -> moral delta ->
    push = clamp(rate * delta)``); on any failure falls back to the deterministic
    heuristic ``impact(appraise(text), MORAL, MORAL_M)``. Returns
    ``(push, source, reasoning)`` where ``source`` is the provider label or
    ``"heuristic"``, and ``reasoning`` is the model's note (empty on fallback).
    """
    try:
        delta = _llm_moral_delta(text)
        push = {dim: clamp(LLM_FORMATION_RATE * delta[dim]) for dim in MORAL}
        return push, LLM_LABEL, delta["reasoning"]
    except Exception as exc:  # any failure -> graceful deterministic fallback
        log.info("LLM moral push unavailable (%s); using heuristic fallback", exc)
        if appraisal is None:
            appraisal = appraise(text)
        return impact(appraisal, MORAL, MORAL_M), "heuristic", ""
