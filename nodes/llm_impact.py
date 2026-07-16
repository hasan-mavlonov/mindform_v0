"""Experience text -> signed per-trait push via an LLM, with heuristic fallback.

This is the LLM-primary path for personality formation. An OpenAI-compatible chat
model -- Google's **Gemma 4** through the Gemini API by default -- reads the
experience and returns a signed OCEAN *delta* in [-1, 1] per trait: the directional
pressure of a SINGLE occurrence of the experience. We scale that delta by
``LLM_FORMATION_RATE`` to get the push that ``updater.py`` then applies with
diminishing returns ``(1 - |trait|)``. The model judges only the experience's
meaning, never a trait-expression reading of the author (formation, not detection).

Repetition is NOT baked into the delta: the engine accumulates repeated experiences
itself (the user logs many events) and the diminishing-returns update handles the
asymptote, so the prompt asks for the effect of one occurrence.

If the LLM is unavailable -- no API key (``GEMINI_API_KEY``), the ``openai`` package
is not installed, a network error, or an unparseable reply -- ``push_from_text``
falls back to the deterministic heuristic ``impact(appraise(text))``. The engine
therefore always produces a push and never hard-depends on the network.

Configure by copying ``.env.example`` to ``.env`` and setting ``GEMINI_API_KEY``
(or point ``LLM_BASE_URL`` / ``LLM_MODEL`` at any OpenAI-compatible provider).
"""

import logging

from core.config import BASIS, LLM_FORMATION_RATE, LLM_LABEL
from core.llm import complete_json
from core.appraisal import appraise
from core.impact import impact, clamp

log = logging.getLogger("mindform.llm")

SYSTEM_PROMPT = """You are MindForm's Personality Formation Engine.

Your task is NOT to identify the current personality of the person who wrote the text.

Your task is to estimate the directional pressure a SINGLE occurrence of the described
experience exerts on each personality trait -- which way it pushes the trait, and how
strongly. The engine itself accumulates repeated experiences and applies diminishing
returns over time, so judge ONE occurrence, not a lifetime of it.

Think in terms of personality formation, not personality detection.

The personality model is OCEAN:

O = Openness
C = Conscientiousness
E = Extraversion
A = Agreeableness
N = Neuroticism

For each trait, estimate a delta in the range [-1.0, 1.0].

Interpretation:

+1.0 = one occurrence of this experience exerts the strongest upward pressure on the trait
0.0  = little or no effect
-1.0 = one occurrence exerts the strongest downward pressure on the trait

Important:

* Judge the EXPERIENCE, not the author.
* Use common-sense developmental psychology.
* Do not rely on keyword matching; consider context and meaning.
* One experience may affect multiple traits at once.
* Neutral experiences should produce values near zero.

Examples:

Experience:
"I spent the evening laughing and talking with close friends."

Output:
{"O": 0.0, "C": 0.0, "E": 0.8, "A": 0.4, "N": -0.2}

Experience:
"I stayed isolated in my room for several days and avoided everyone."

Output:
{"O": -0.1, "C": -0.2, "E": -0.8, "A": -0.1, "N": 0.4}

Experience:
"I started learning a completely new language and practiced every day."

Output:
{"O": 0.8, "C": 0.6, "E": 0.0, "A": 0.0, "N": -0.1}

Experience:
"I failed an important exam despite studying for months."

Output:
{"O": 0.0, "C": 0.1, "E": -0.1, "A": 0.0, "N": 0.5}

Experience:
"I volunteered every weekend helping elderly people."

Output:
{"O": 0.1, "C": 0.3, "E": 0.2, "A": 0.8, "N": -0.1}

Do not think out loud or emit any <thought>/<thinking> block before the JSON --
put any explanation in the "reasoning" field, nowhere else.

Return ONLY valid JSON, with no markdown and no extra text, in exactly this format:

{"O": float, "C": float, "E": float, "A": float, "N": float, "reasoning": "brief explanation"}
"""


def _llm_delta(text, lens=""):
    """Ask the LLM for the signed OCEAN delta of one occurrence of ``text``.

    Uses the configured OpenAI-compatible provider (Gemini 3.5 Flash by default).
    Returns ``{O, C, E, A, N: float, "reasoning": str}``. Raises on any failure
    (missing key/package, network error, malformed JSON, missing/non-numeric
    trait) so ``push_from_text`` can fall back to the heuristic.
    """
    user = f"Experience:\n{text}" + (f"\n\n{lens}" if lens else "")
    data = complete_json(SYSTEM_PROMPT, user)
    delta = {dim: float(data[dim]) for dim in BASIS}  # KeyError / ValueError -> fallback
    delta["reasoning"] = str(data.get("reasoning", ""))
    return delta


def push_from_text(text, appraisal=None, lens=""):
    """Best-available signed per-trait push for an experience.

    Tries the LLM first (``text -> OCEAN delta -> push = clamp(rate * delta)``);
    on any failure falls back to the deterministic heuristic. Returns
    ``(push, source, reasoning)`` where ``source`` is the provider label (e.g.
    ``"gemma"``) or ``"heuristic"``, and ``reasoning`` is the model's note
    (empty on fallback).
    """
    try:
        delta = _llm_delta(text, lens)
        push = {dim: clamp(LLM_FORMATION_RATE * delta[dim]) for dim in BASIS}
        return push, LLM_LABEL, delta["reasoning"]
    except Exception as exc:  # any failure -> graceful deterministic fallback
        log.info("LLM push unavailable (%s); using heuristic fallback", exc)
        if appraisal is None:
            appraisal = appraise(text)
        return impact(appraisal), "heuristic", ""
