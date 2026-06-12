"""Experience text -> signed Schwartz values delta via an LLM, heuristic fallback.

The CHARACTER half of formation. Where ``llm_impact.push_from_text`` reads an
experience for how it pushes the OCEAN *traits*, this reads the SAME experience for
how it pushes the ten Schwartz *values* -- what one occurrence teaches the person to
prize or dismiss. The contract mirrors ``llm_impact`` exactly: an OpenAI-compatible
model (Google's Gemma 4 by default) returns a signed delta in [-1, 1] per value for a
SINGLE occurrence; we scale it by ``LLM_FORMATION_RATE`` into a push that
``character.update_values`` then applies with the same diminishing returns. The engine
accumulates repetition itself, so the prompt asks only about one occurrence.

If the LLM is unavailable -- no key (``GEMINI_API_KEY``), no ``openai`` package, a
network error, or an unparseable reply -- it falls back to the deterministic
``impact(appraise(text), VALUES, VALUES_M)`` heuristic, so values formation, like
trait formation, never hard-depends on the network.

Note: with an API key set, a turn now makes a *separate* values call in addition to
the trait push and the reply. Folding traits + values into one formation call is a
natural future optimization; they are kept apart here so CHARACTER stays its own
clean, independently swappable node (mirroring the diagram).
"""

import logging

from config import (
    VALUES, VALUES_M, LLM_FORMATION_RATE, LLM_LABEL, LLM_MODEL, LLM_BASE_URL,
    LLM_API_KEY, parse_json_object,
)
from appraisal import appraise
from impact import impact, clamp

log = logging.getLogger("mindform.values")

VALUES_SYSTEM_PROMPT = """You are MindForm's Character Formation Engine.

Your task is to estimate the directional pressure a SINGLE occurrence of the described
experience exerts on each of a person's BASIC HUMAN VALUES -- which values it teaches
them to prize more, and which less. Judge ONE occurrence; the engine accumulates
repetition and applies diminishing returns over time.

This is values FORMATION, not values detection: read the experience's meaning, not the
current personality of whoever wrote it.

The model is Schwartz's ten basic values:

SD = Self-Direction (independent thought and action; autonomy, curiosity)
ST = Stimulation (excitement, novelty, and challenge)
HE = Hedonism (pleasure and sensuous gratification)
AC = Achievement (personal success through demonstrating competence)
PO = Power (social status and prestige, control or dominance over people/resources)
SE = Security (safety, harmony, and stability of self, relationships, and society)
CO = Conformity (restraint of actions/impulses likely to upset others or violate norms)
TR = Tradition (respect for and commitment to the customs and ideas of one's culture/religion)
BE = Benevolence (preserving and enhancing the welfare of people one is close to)
UN = Universalism (understanding, tolerance, justice, and protection for all people and nature)

For each value, estimate a delta in the range [-1.0, 1.0]:

+1.0 = one occurrence most strongly teaches the person to prize this value
 0.0 = little or no effect
-1.0 = one occurrence most strongly pushes the person away from this value

Important:
* Judge the EXPERIENCE, not the author.
* One experience may move several values at once, some up and some down -- the values
  trade off against each other (e.g. embracing risk raises Stimulation while lowering
  Security; serving the wider world can lower Power).
* Neutral experiences should produce values near zero.
* Do not think out loud or emit any <thought>/<thinking> block before the JSON --
  put any explanation in the "reasoning" field, nowhere else.

Examples:

Experience:
"I spoke up against my whole team to defend someone who was being treated unfairly."

Output:
{"SD": 0.4, "ST": 0.1, "HE": 0.0, "AC": 0.0, "PO": 0.0, "SE": -0.2, "CO": -0.4, "TR": 0.0, "BE": 0.5, "UN": 0.6, "reasoning": "stood on principle for another's welfare against the group"}

Experience:
"I quit my stable job to travel alone with no plan."

Output:
{"SD": 0.7, "ST": 0.8, "HE": 0.3, "AC": -0.2, "PO": -0.1, "SE": -0.6, "CO": -0.3, "TR": -0.2, "BE": 0.0, "UN": 0.1, "reasoning": "chose autonomy and novelty over safety and convention"}

Experience:
"I spent the holiday keeping every family ritual exactly as my grandmother taught me."

Output:
{"SD": -0.2, "ST": -0.2, "HE": 0.1, "AC": 0.0, "PO": 0.0, "SE": 0.3, "CO": 0.3, "TR": 0.8, "BE": 0.4, "UN": 0.0, "reasoning": "honored custom and family continuity"}

Return ONLY valid JSON, with no markdown and no extra text, in exactly this format:

{"SD": float, "ST": float, "HE": float, "AC": float, "PO": float, "SE": float, "CO": float, "TR": float, "BE": float, "UN": float, "reasoning": "brief explanation"}
"""


def _llm_values_delta(text):
    """Ask the LLM for the signed Schwartz delta of one occurrence of ``text``.

    Returns ``{SD..UN: float, "reasoning": str}``. Raises on any failure (missing
    key/package, network error, malformed JSON, missing/non-numeric value) so
    ``values_push_from_text`` can fall back to the heuristic.
    """
    if not LLM_API_KEY:
        raise RuntimeError("no LLM API key is set (GEMINI_API_KEY)")

    from openai import OpenAI  # lazy: the heuristic fallback works without this package

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": VALUES_SYSTEM_PROMPT},
            {"role": "user", "content": f"Experience:\n{text}"},
        ],
        temperature=0.2,
        max_tokens=600,
        timeout=30,
    )
    data = parse_json_object(completion.choices[0].message.content)
    delta = {dim: float(data[dim]) for dim in VALUES}  # KeyError / ValueError -> fallback
    delta["reasoning"] = str(data.get("reasoning", ""))
    return delta


def values_push_from_text(text, appraisal=None):
    """Best-available signed per-value push for an experience.

    Mirrors ``llm_impact.push_from_text``: tries the LLM (``text -> Schwartz delta ->
    push = clamp(rate * delta)``); on any failure falls back to the deterministic
    heuristic ``impact(appraise(text), VALUES, VALUES_M)``. Returns
    ``(push, source, reasoning)`` where ``source`` is the provider label (e.g.
    ``"gemma"``) or ``"heuristic"``, and ``reasoning`` is the model's note (empty on
    fallback).
    """
    try:
        delta = _llm_values_delta(text)
        push = {dim: clamp(LLM_FORMATION_RATE * delta[dim]) for dim in VALUES}
        return push, LLM_LABEL, delta["reasoning"]
    except Exception as exc:  # any failure -> graceful deterministic fallback
        log.info("LLM values push unavailable (%s); using heuristic fallback", exc)
        if appraisal is None:
            appraisal = appraise(text)
        return impact(appraisal, VALUES, VALUES_M), "heuristic", ""
