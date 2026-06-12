"""Central configuration: trait basis, appraisal schema, prior matrix, constants.

Everything tunable about MindForm lives here so the moving parts stay in one
place and each can later be swapped for a learned component.
"""

import json
import os
import re


def _load_dotenv(path=".env"):
    """Minimal .env loader (stdlib only): KEY=VALUE lines -> os.environ.

    Real shell environment variables win (we only fill in defaults), and a missing
    file is fine -- this keeps the no-network/no-secret path dependency-free. Copy
    .env.example to .env to configure. (For richer parsing, ``pip install
    python-dotenv`` and this stays compatible.)
    """
    if not os.path.exists(path):
        return
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

# --- Trait basis (a swappable coordinate system; default Big Five / OCEAN) ---
BASIS = ["O", "C", "E", "A", "N"]
BASIS_NAMES = {
    "O": "openness",
    "C": "conscientiousness",
    "E": "extraversion",
    "A": "agreeableness",
    "N": "neuroticism",
}

# --- Appraisal schema: dim -> range.  "signed" = [-1, 1], "unit" = [0, 1]. ---
APPRAISAL_SCHEMA = {
    "valence": "signed",
    "intensity": "unit",
    "novelty": "unit",
    "agency": "signed",
    "social": "signed",
    "outcome": "signed",
    "self_relevance": "unit",
    "threat_challenge": "signed",   # -1 = threat/loss, +1 = challenge/growth
}
APPRAISAL_DIMS = list(APPRAISAL_SCHEMA)

# --- Appraisal -> trait prior matrix M (rows = BASIS, cols = appraisal dims). ---
# Hand-authored from the personality-change literature. This is the single
# basis-specific component; swap M to change the trait basis, learn M later.
# Outputs are clamped to [-1, 1], so rows need not be normalized.
M = {
    "O": {"novelty": 0.6, "valence": 0.2, "threat_challenge": 0.3, "self_relevance": 0.1},
    "C": {"outcome": 0.5, "agency": 0.3, "self_relevance": 0.2},
    "E": {"social": 0.6, "valence": 0.2, "outcome": 0.2},
    "A": {"social": 0.3, "valence": 0.4},
    # bad + helpless + failed + threatening -> +N ; good + agentic + mastery -> -N
    "N": {"valence": -0.4, "agency": -0.3, "outcome": -0.3,
          "threat_challenge": -0.3, "intensity": 0.2},
}

# --- Formation rate ---
# push = FORMATION_RATE * salience * (M . appraisal); updater.py then applies it
# with diminishing returns (1 - |trait|). Tuned so a vivid social experience moves
# extraversion by ~0.3 on the first occurrence (0.00 -> 0.30 -> 0.51 -> ...).
# Lower it for slower, more gradual personality formation.
FORMATION_RATE = 1.3

# --- Character: Schwartz basic values (the values substrate formed by experience) -
# Where TEMPERAMENT is the innate OCEAN baseline a character is *born* with, CHARACTER
# is what they come to *prize* -- laid down by experience, not biology. We model it
# with Schwartz's ten basic human values. Each is signed [-1, 1]: 0 = neutral,
# +1 = central / strongly held, -1 = strongly rejected. Values start at 0 (earned,
# not innate) and form by the SAME push + diminishing-returns dynamics as the traits.
VALUES = ["SD", "ST", "HE", "AC", "PO", "SE", "CO", "TR", "BE", "UN"]
VALUES_NAMES = {
    "SD": "self-direction",   # independent thought and action; autonomy, curiosity
    "ST": "stimulation",      # excitement, novelty, challenge
    "HE": "hedonism",         # pleasure and enjoyment
    "AC": "achievement",      # success through demonstrating competence
    "PO": "power",            # status, prestige, control or dominance
    "SE": "security",         # safety, harmony, stability
    "CO": "conformity",       # restraint, fitting in, following norms
    "TR": "tradition",        # custom, culture, religion
    "BE": "benevolence",      # caring for people one is close to
    "UN": "universalism",     # tolerance, justice, welfare of all and nature
}
# Schwartz's two higher-order axes (four poles), for plain-language roll-up readouts.
# Hedonism bridges openness-to-change and self-enhancement, so its weight is split.
VALUES_HIGHER_ORDER = {
    "openness_to_change": {"SD": 1.0, "ST": 1.0, "HE": 0.5},
    "self_enhancement":   {"AC": 1.0, "PO": 1.0, "HE": 0.5},
    "conservation":       {"SE": 1.0, "CO": 1.0, "TR": 1.0},
    "self_transcendence": {"BE": 1.0, "UN": 1.0},
}
# Appraisal -> values prior (heuristic fallback only; the LLM is primary, exactly as
# with traits). Same shape and role as M: rows = VALUES, cols = appraisal dims. Hand-
# authored and deliberately conservative -- the values that are hard to read from a
# lexical appraisal (power, tradition, universalism) lean on the LLM and stay light.
VALUES_M = {
    "SD": {"agency": 0.5, "novelty": 0.3},
    "ST": {"novelty": 0.6, "intensity": 0.3, "threat_challenge": 0.2},
    "HE": {"valence": 0.5, "intensity": 0.2},
    "AC": {"outcome": 0.6, "agency": 0.3},
    "PO": {"agency": 0.4, "outcome": 0.2, "social": 0.2},
    "SE": {"threat_challenge": -0.4, "valence": 0.1},   # threat teaches the worth of safety
    "CO": {"social": 0.3, "agency": -0.2},
    "TR": {"social": 0.2},
    "BE": {"social": 0.5, "valence": 0.2},
    "UN": {"social": 0.1, "valence": 0.1, "novelty": 0.1},
}
# How many similar past experiences (memory recurrence, RECURRENCE_THRESHOLD) it takes
# for a recurring experience to count as a habit.
HABIT_MIN_RECURRENCE = 3

# --- LLM push (OpenAI-compatible): default Google Gemma 4 via the Gemini API ---
# llm_impact.py asks an OpenAI-compatible chat model for a signed OCEAN delta in
# [-1, 1] per trait, then push = clamp(LLM_FORMATION_RATE * delta); updater.py
# applies it with diminishing returns. A max-strength delta (1.0) thus nudges a
# neutral trait by the rate (~0.3), never all the way in one experience --
# formation builds over many experiences. Falls back to the heuristic impact()
# whenever no key/model is reachable.
#
# The default provider is Google's Gemma 4 through the Gemini API's
# OpenAI-compatible endpoint. Put your Google AI Studio key in GEMINI_API_KEY
# (copy .env.example). Any OpenAI-compatible endpoint still works -- point
# LLM_BASE_URL + LLM_MODEL at it. The legacy DEEPSEEK_* names remain honored.
def _env(*names, default=None):
    """First non-empty environment variable among ``names`` (real env wins)."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


LLM_API_KEY = _env("LLM_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY")
LLM_BASE_URL = _env("LLM_BASE_URL", "DEEPSEEK_BASE_URL",
                    default="https://generativelanguage.googleapis.com/v1beta/openai/")
LLM_MODEL = _env("LLM_MODEL", "DEEPSEEK_MODEL", default="gemma-4-31b-it")
LLM_FORMATION_RATE = float(_env("LLM_FORMATION_RATE", default="0.3"))

# Backward-compatible aliases (older imports referenced these names).
DEEPSEEK_MODEL = LLM_MODEL
DEEPSEEK_BASE_URL = LLM_BASE_URL


def _llm_label(model):
    """Short provider tag for the UI/CLI (e.g. 'gemma', 'deepseek')."""
    name = (model or "").lower()
    if "gemma" in name:
        return "gemma"
    if "deepseek" in name:
        return "deepseek"
    if "gpt" in name or "openai" in name:
        return "openai"
    return "llm"


LLM_LABEL = _llm_label(LLM_MODEL)


# Tag names open-weight models use to wrap their private chain-of-thought.
_REASONING_TAGS = ("thought", "thinking", "think", "reasoning", "reason",
                   "reflection", "scratchpad")


def strip_reasoning(text):
    """Remove any chain-of-thought block a model emitted before its real answer.

    Open-weight instruction models -- Gemma among them -- sometimes narrate their
    reasoning in a ``<thought>...</thought>`` block *before*, or *instead of*, the
    answer. That block must never reach the user or the JSON parser, so we drop:

    * well-formed ``<thought>...</thought>`` blocks (any of several tag names);
    * a trailing block left unterminated when the model hit ``max_tokens``
      mid-thought (``<thought>...`` to end of text); and
    * any stray lone tag left over.

    Matching is case-insensitive and tolerant of attributes/whitespace in the tag.
    Returns the cleaned text (possibly empty, if the model produced only thinking).
    """
    if not text:
        return ""
    names = "|".join(_REASONING_TAGS)
    # Closed blocks anywhere (backref keeps the open/close tag names matched).
    text = re.sub(rf"<\s*({names})\b[^>]*>.*?<\s*/\s*\1\s*>", "", text,
                  flags=re.IGNORECASE | re.DOTALL)
    # A block left open by truncation: drop from the opening tag to the end.
    text = re.sub(rf"<\s*({names})\b[^>]*>.*$", "", text,
                  flags=re.IGNORECASE | re.DOTALL)
    # Any orphaned lone tag.
    text = re.sub(rf"<\s*/?\s*({names})\b[^>]*>", "", text, flags=re.IGNORECASE)
    return text.strip()


def parse_json_object(text):
    """Parse a JSON object from a model reply, tolerating ```fences``` and prose.

    Open-weight models (Gemma included) sometimes wrap JSON in a markdown code
    fence, prepend a ``<thought>`` reasoning block, or add a stray sentence. We
    strip reasoning, then a leading/trailing fence, and -- failing a direct parse
    -- fall back to the outermost ``{...}`` span. Raises ValueError if no JSON
    object can be found, so the callers fall back to the heuristic.
    """
    cleaned = strip_reasoning(text)
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except ValueError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))

# --- Memory / recurrence ---
RECURRENCE_THRESHOLD = 0.80   # cosine similarity to count as the "same" experience

# --- Temperament (genesis baseline) ---
# A character is born (temperament.genesis) with a per-trait OCEAN baseline `mu`
# in [-1, 1] and a per-trait stickiness `tau` in [0, 1]. The current traits start
# AT the baseline (x = mu). DEFAULT_TAU is used when a seed leaves stickiness
# unspecified (and for blank / legacy characters). High tau = resilient (snaps
# back to baseline); low tau = easily reshaped. The baseline-as-attractor and slow
# baseline-drift dynamics that consume tau are added in updater.py (Slice 2).
DEFAULT_TAU = 0.30

# --- Identity (immutable facts collected when a character is created) ---
# (field_key, prompt_label), in the order the creation form asks for them. These
# are stored verbatim in personality["identity"] and never drift. The separate
# free-text "background" blurb (not listed here) is what seeds temperament mu/tau.
IDENTITY_FIELDS = [
    ("name", "Name"),
    ("age", "Age"),
    ("gender", "Gender"),
    ("origin", "Where from (city / country)"),
    ("culture", "Culture / ethnicity"),
    ("language", "Native language"),
    ("religion", "Religion raised in"),
    ("family", "Family background"),
]

# --- Trait questionnaire (manual character creation) ---
# One plain-language question per OCEAN trait: (key, name, low-pole, high-pole).
# The answer is a level 1-5 that maps to a baseline mu via TRAIT_LEVELS, so the
# author sets the temperament directly instead of having it inferred from text.
TRAIT_QUESTIONS = [
    ("O", "Openness",          "practical, conventional", "curious, imaginative"),
    ("C", "Conscientiousness", "spontaneous, easygoing",  "disciplined, organized"),
    ("E", "Extraversion",      "reserved, private",       "outgoing, energetic"),
    ("A", "Agreeableness",     "blunt, competitive",      "warm, cooperative"),
    ("N", "Neuroticism",       "calm, resilient",         "sensitive, easily stressed"),
]
TRAIT_LEVELS = {1: -0.8, 2: -0.4, 3: 0.0, 4: 0.4, 5: 0.8}
