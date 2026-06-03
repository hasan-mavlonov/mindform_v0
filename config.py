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
# push = FORMATION_RATE * salience * (M . appraisal). In the two-timescale model
# (see below) the push lands on the fast STATE layer (mood), so FORMATION_RATE is
# how strongly a single experience moves mood -- ~0.2-0.3 for a vivid experience.
FORMATION_RATE = 1.3

# --- Two-timescale dynamics: fast STATE (mood) vs slow TRAIT (disposition) ---
# A persistent personality is not the same as a momentary reaction, so each trait
# axis carries two values (see personality.py / updater.py):
#   state  -- fast: every experience pushes it; it decays back toward 0 (mood fades)
#   trait  -- slow: integrates SUSTAINED state, with diminishing returns, and drifts
#             back toward a set-point absent reinforcement (set-point theory)
# This is what makes "formation over time" mean something: one vivid event moves
# mood a lot and disposition barely; only a repeated/sustained pattern of mood
# graduates into a lasting trait, and an unreinforced trait partially relaxes back.
STATE_DECAY = 0.30          # fraction of mood that fades each experience
CONSOLIDATION_RATE = 0.05   # how much sustained state graduates into disposition
HOMEOSTASIS = 0.01          # set-point return rate of the slow trait toward SETPOINT
SETPOINT = 0.0              # dispositional baseline a trait relaxes toward

# --- Memory / recurrence ---
RECURRENCE_THRESHOLD = 0.80   # cosine similarity to count as the "same" experience
