"""Central configuration: trait basis, appraisal schema, prior matrix, constants.

Everything tunable about MindForm lives here so the moving parts stay in one
place and each can later be swapped for a learned component.
"""

import os


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

# --- LLM push (DeepSeek, OpenAI-compatible) ---
# llm_impact.py asks DeepSeek for a signed OCEAN delta in [-1, 1] per trait, then
# push = clamp(LLM_FORMATION_RATE * delta); updater.py applies it with diminishing
# returns. A max-strength delta (1.0) thus nudges a neutral trait by the rate (~0.3),
# never all the way in one experience -- formation builds over many experiences.
# Falls back to the heuristic impact() when DeepSeek is unavailable. Secrets
# (DEEPSEEK_API_KEY) live in .env -- copy .env.example. Env vars override these.
LLM_FORMATION_RATE = float(os.environ.get("LLM_FORMATION_RATE", "0.3"))
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

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
