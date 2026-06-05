"""Central configuration: trait basis, prior matrix, constants, shared helpers.

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


def clamp(value, minimum=-1.0, maximum=1.0):
    """Clamp into [minimum, maximum] -- defaults to the trait/push range [-1, 1]."""
    return max(minimum, min(maximum, value))


# --- Trait basis (a swappable coordinate system; default Big Five / OCEAN) ---
BASIS = ["O", "C", "E", "A", "N"]
BASIS_NAMES = {
    "O": "openness",
    "C": "conscientiousness",
    "E": "extraversion",
    "A": "agreeableness",
    "N": "neuroticism",
}

# --- Appraisal -> trait prior matrix M (rows = BASIS, cols = appraisal dims). ---
# Hand-authored from the personality-change literature. Drives the deterministic
# fallback push (impact.py) when DeepSeek is unavailable; learn M later.
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

# --- Formation rate (fallback heuristic path) ---
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
