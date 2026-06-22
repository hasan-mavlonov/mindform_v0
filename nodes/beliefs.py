"""Experience text -> the beliefs it forms (an LLM extraction; no heuristic fallback).

The BELIEF half of CHARACTER. Where ``values.py`` / ``moral.py`` push fixed vectors,
beliefs are open-ended PROPOSITIONS the character comes to hold (``"hard work pays off"``,
``"people can be trusted"``). The LLM reads one experience and returns 0-2 general beliefs
it expresses, strengthens, or undermines, each with a signed confidence delta in [-1, 1]
for THIS occurrence. ``character.update_beliefs`` accumulates the conviction with the same
diminishing returns; ``character.form_beliefs`` walks the memory backlog so experiences
logged while the model was offline still become beliefs later.

There is deliberately NO heuristic fallback: open-vocabulary propositions can't be read
from a lexical appraisal. With no LLM, ``extract_beliefs`` raises, and belief formation
simply waits -- the experiences are safe in memory until the model is back.
"""

import logging

from core.config import LLM_LABEL
from core.llm import complete_json

log = logging.getLogger("mindform.beliefs")

BELIEF_SYSTEM_PROMPT = """You are MindForm's Belief-Formation Engine.

From a single described experience, infer the BELIEFS it forms in the person -- general
propositions about themselves, other people, or the world that this experience teaches,
strengthens, or undermines. Judge ONE occurrence; the engine accumulates conviction over
repetition.

Rules:
* Return 0 to 2 beliefs. Most ordinary experiences form 0 or 1. Return an empty list if none.
* Each belief is a short, general statement (e.g. "hard work pays off", "people can be
  trusted", "the world is dangerous") -- NOT a description of the event itself.
* "confidence" is the directional pressure of THIS one occurrence, in [-1.0, 1.0]:
  +1 strongly supports holding the belief, -1 strongly undermines it.
* Prefer canonical phrasings so the same belief recurs identically across experiences.
* Judge the experience's meaning, not the author's current personality.
* Output ONLY JSON. No markdown, no <thought>/<thinking> block.

Examples:

Experience: "I studied for months and still failed the exam."
Output: {"beliefs": [{"statement": "hard work does not guarantee success", "confidence": 0.6}]}

Experience: "A stranger returned my lost wallet with all the cash inside."
Output: {"beliefs": [{"statement": "people are basically good", "confidence": 0.7}]}

Experience: "I had a quiet lunch."
Output: {"beliefs": []}

Return ONLY valid JSON, in exactly this format:
{"beliefs": [{"statement": str, "confidence": float}]}
"""


def extract_beliefs(text):
    """Ask the LLM for the beliefs one occurrence of ``text`` forms.

    Returns ``(beliefs, source)`` where ``beliefs`` is a (possibly empty) list of
    ``{statement, confidence}``. Raises on any failure (no key/package, network, bad
    JSON) so ``character.form_beliefs`` can leave the experience for a later pass.
    """
    data = complete_json(BELIEF_SYSTEM_PROMPT, f"Experience:\n{text}")
    out = []
    for item in (data.get("beliefs") or [])[:2]:
        statement = str(item.get("statement", "")).strip()
        if statement:
            out.append({"statement": statement, "confidence": float(item.get("confidence", 0.0))})
    return out, LLM_LABEL
