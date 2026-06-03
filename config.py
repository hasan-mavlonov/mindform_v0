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
HOMEOSTASIS = 0.01          # return rate of the slow trait toward its core baseline
SETPOINT = 0.0              # default core baseline when an agent is given no temperament

# --- State-dependent formation: SAME experience, DIFFERENT effect per agent --------
# "A single text nudges the personality, then the personality is changed based on who
# he was." Three deterministic levers make the nudge depend on the current agent, so
# two agents diverge under identical input -- with NO LLM in the update path:
#   (1) APPRAISAL BIAS   -- current traits + mood color how the event is READ, so the
#       same text becomes a different *experience* (can even flip the direction of
#       change).                                    [appraisal.bias_appraisal]
#   (2) REACTIVITY GAIN  -- emotionally reactive (high-N) agents are moved more by
#       adverse events.                             [impact.reactivity]
#   (3) PLASTICITY DECAY -- the disposition crystallizes as experience accumulates
#       (rank-order stability rises with age -- the one robust law).   [updater]
APPRAISAL_BIAS = {
    "neuroticism_threat": 0.5,   # high N reads a negative event as more threatening
    "openness_novelty":   0.4,   # high O reads novelty as more positive (valence)
    "mood_congruence":    0.3,   # current mood tints valence (good mood -> reads good)
}
REACTIVITY_N = 0.6               # push *= 1 + REACTIVITY_N * max(0, N) * threat_load
PLASTICITY_BASE = 1.0            # young agents form fast ...
PLASTICITY_HALFLIFE = 80.0       # ... then crystallize: rho = BASE / (1 + count/HALFLIFE)

# --- Density-distribution layer (Whole Trait Theory): per-axis dispersion ---
# A trait is not a point but a DISTRIBUTION of momentary states (Fleeson): a steady
# extravert and a volatile one can share the same mean. We track the within-person
# variability of expression as a slow EWMA of squared successive swings in mood
# (MSSD, a standard within-person instability index); its sqrt is reported as the
# trait's "dispersion" (see evaluation.py). It is drift-invariant -- a strong but
# steady stream stays low-dispersion. Within-person variability is itself a stable
# individual difference, so the rate is slow.
VARIABILITY_RATE = 0.10     # EWMA rate for the per-axis state-density variance

# --- Memory / recurrence ---
RECURRENCE_THRESHOLD = 0.80   # cosine similarity to count as the "same" experience

# --- Memory feedback (recurrence shapes formation) ---
# A recurring experience stirs LESS mood each time (habituation) but graduates into
# DISPOSITION more strongly (chronicity / the corresponsive principle): the patterns
# that recur across a life shape character even as they stop feeling novel.
HABITUATION = 0.3      # mood damping:        push *= 1 / (1 + HABITUATION * recurrence)
CHRONICITY = 0.5       # consolidation gain:  *= (1 + CHRONICITY * recurrence)
# Retrieval-conditioned appraisal: a new experience is read partly through the
# appraisals of similar remembered ones ("this reminds me of ...").
RETRIEVAL_K = 5        # how many similar memories to retrieve
RETRIEVAL_ALPHA = 0.3  # weight on the retrieved appraisal prior (0 = ignore memory)
