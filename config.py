"""Central configuration: trait basis, appraisal schema, prior matrix, constants.

Everything tunable about MindForm lives here so the moving parts stay in one
place and each can later be swapped for a learned component.
"""

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

# --- Memory / recurrence ---
RECURRENCE_THRESHOLD = 0.80   # cosine similarity to count as the "same" experience
