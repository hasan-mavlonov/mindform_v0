"""Apply an experience's push on two timescales, modulated by recurrence.

Per experience, for each trait axis:

    1. mood fades, then this experience hits it (fast); a recurring experience
       hits mood less hard (habituation):
           push'  = push / (1 + HABITUATION * recurrence)
           state <- clamp(state * (1 - STATE_DECAY) + push')
    2. sustained mood consolidates into disposition, with diminishing returns; a
       recurring experience consolidates more (chronicity), and the agent's
       PLASTICITY decays as experience accumulates (crystallization with age):
           plasticity = PLASTICITY_BASE / (1 + experience_count / PLASTICITY_HALFLIFE)
           trait <- clamp(trait + CONSOLIDATION_RATE * plasticity
                                 * (1 + CHRONICITY*recurrence) * state * (1 - |trait|))
    3. set-point return: disposition drifts back toward THIS AGENT'S CORE (homeostasis),
       so identity has a home it returns to absent reinforcement:
           trait <- clamp(trait - HOMEOSTASIS * (trait - core))
    4. density layer (Whole Trait Theory): the within-person variability of
       expression is tracked as a slow EWMA of squared successive swings in mood
       (MSSD), drift-invariant so a strong-but-steady life stays low-dispersion:
           trait_var <- (1-VARIABILITY_RATE)*trait_var + VARIABILITY_RATE*(state-prev_state)^2

So a single vivid experience moves mood a lot and disposition barely; a repeated or
sustained pattern of mood graduates into a lasting trait (and, via chronicity, the
more familiar it is the more efficiently it does so, even as it stirs less mood).
With ``recurrence = 0`` this reduces exactly to the plain two-timescale update.
"""

from config import (
    STATE_DECAY, CONSOLIDATION_RATE, HOMEOSTASIS, SETPOINT,
    HABITUATION, CHRONICITY, VARIABILITY_RATE,
    PLASTICITY_BASE, PLASTICITY_HALFLIFE,
)


def clamp(value, minimum=-1.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def update_personality(personality, push, recurrence=0):
    """Return a new personality with mood and disposition advanced (input unchanged).

    ``recurrence`` is how many similar experiences are already remembered; it damps
    the mood response (habituation) and amplifies consolidation (chronicity). The
    agent's accumulated ``experience_count`` decays its plasticity (crystallization),
    and each trait relaxes toward this agent's ``core`` baseline (temperament).
    """
    habituation = 1.0 / (1.0 + HABITUATION * recurrence)
    chronicity = 1.0 + CHRONICITY * recurrence
    count = personality.get("experience_count", 0)
    plasticity = PLASTICITY_BASE / (1.0 + count / PLASTICITY_HALFLIFE)
    core = personality.get("core", {})

    traits = {}
    for dim, layers in personality["traits"].items():
        delta = push.get(dim, 0.0) * habituation
        home = float(core.get(dim, SETPOINT))

        prev_state = layers["state"]
        state = clamp(prev_state * (1 - STATE_DECAY) + delta)
        trait = clamp(layers["trait"]
                      + CONSOLIDATION_RATE * plasticity * chronicity
                      * state * (1 - abs(layers["trait"])))
        trait = clamp(trait - HOMEOSTASIS * (trait - home))

        # density layer (Whole Trait Theory): track the within-person variability of
        # momentary expression as a slow EWMA of squared successive swings in mood
        # (MSSD -- a standard within-person instability index). It is drift-invariant:
        # a consistent life settles to ~0 swing (low dispersion) however strong the
        # experience, while a volatile life keeps swinging (high dispersion) -- so two
        # people can share a mean disposition yet differ in how steadily they express it.
        swing_sq = (state - prev_state) ** 2
        trait_var = ((1 - VARIABILITY_RATE) * layers.get("trait_var", 0.0)
                     + VARIABILITY_RATE * swing_sq)

        traits[dim] = {"state": state, "trait": trait, "trait_var": trait_var}

    return {
        "traits": traits,
        "core": dict(core),
        "experience_count": count + 1,
    }
