"""Experience text -> appraisal vector via an LLM, offline fallback -- and the distillation source.

PERCEPTION, LLM-primary. ``core.appraisal.appraise`` (trained head -> lexicon) deliberately
never touches the network; this node adds the same LLM-primary/offline-fallback discipline
every formation node uses to the appraisal itself. On the online path Gemma reads the raw
text and returns the eight appraisal dims -- so the reading that feeds the needs, esteem,
behavior cues, mood panel, and memory records is no longer the thin lexicon whenever a key
is set.

Every successful LLM appraisal is also the TEACHER SIGNAL for the offline head: the bridge
logs ``(text, appraisal)`` pairs to ``data/appraisal_dataset.jsonl`` (``core.appraisal_log``),
the exact file ``bootstrap/train_appraisal_head.py`` trains from -- so the offline appraiser
improves with use (distillation), and ``appraise()`` automatically prefers the trained head
once it exists. Only LLM-labelled rows are ever logged: the head must never train on its own
(or the lexicon's) outputs.

If the LLM is unavailable -- no key (``GEMINI_API_KEY``), no ``openai`` package, a network
error, or an unparseable reply -- it falls back to ``appraise()`` (head -> lexicon), so
perception, like everything else, never hard-depends on the network.
"""

import logging

from core.config import APPRAISAL_DIMS, APPRAISAL_SCHEMA, LLM_LABEL
from core.llm import complete_json
from core.appraisal import appraise

log = logging.getLogger("mindform.llm_appraisal")

APPRAISAL_SYSTEM_PROMPT = """You are MindForm's Perception Engine.

Your task is to appraise a SINGLE described experience -- to rate the causal ingredients
of how it lands on the person living it. This is meaning extraction, NOT trait detection:
rate the experience itself, not the personality of whoever wrote it.

Rate these eight appraisal dimensions:

valence          -1..1  : how bad or good the experience feels (-1 awful, +1 wonderful)
intensity         0..1  : emotional strength / arousal (0 barely registers, 1 overwhelming)
novelty           0..1  : how new or unfamiliar this kind of experience is
agency           -1..1  : their own doing and control (+1 chose/acted/mastered,
                          -1 helpless / done TO them, 0 neither)
social           -1..1  : social presence (+1 people are central to the moment,
                          -1 marked isolation or exclusion, 0 nonsocial)
outcome          -1..1  : success/gain (+1) vs failure/loss (-1), 0 no clear outcome
self_relevance    0..1  : how much it bears on the person themselves and their life
threat_challenge -1..1  : the framing of demand -- CAREFUL WITH THE SIGN:
                          -1 = threat or loss looming, +1 = challenge / growth / opportunity,
                          0 = neither

Judge the single occurrence as an average person would live it.

Return ONLY valid JSON, no markdown, exactly:
{"valence": float, "intensity": float, "novelty": float, "agency": float, "social": float,
 "outcome": float, "self_relevance": float, "threat_challenge": float}"""


def _clamp_dim(key, value):
    lo = 0.0 if APPRAISAL_SCHEMA.get(key) == "unit" else -1.0
    return max(lo, min(1.0, float(value)))


def _llm_appraise(text):
    """Ask the LLM for the eight dims. Raises on any failure (missing key/package, network,
    malformed JSON, missing or non-numeric dim) so the caller can fall back."""
    data = complete_json(APPRAISAL_SYSTEM_PROMPT, f"Experience:\n{text}", temperature=0.1)
    return {k: _clamp_dim(k, data[k]) for k in APPRAISAL_DIMS}   # KeyError/ValueError -> fallback


def appraise_from_text(text):
    """Best-available appraisal for an experience. Returns ``(appraisal, source)`` with
    source in {LLM label, "offline"} -- only LLM-sourced readings may enter the
    distillation log."""
    try:
        return _llm_appraise(text), LLM_LABEL
    except Exception as exc:
        log.info("LLM appraisal unavailable (%s); using offline appraiser", exc)
        return appraise(text), "offline"
