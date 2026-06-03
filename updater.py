"""Apply an experience's push on two timescales: fast mood, slow disposition.

Per experience, for each trait axis:

    1. mood fades, then this experience hits it (fast):
           state <- clamp(state * (1 - STATE_DECAY) + push)
    2. sustained mood consolidates into disposition, with diminishing returns (slow):
           trait <- clamp(trait + CONSOLIDATION_RATE * state * (1 - |trait|))
    3. set-point return: disposition drifts back toward baseline (homeostasis):
           trait <- clamp(trait - HOMEOSTASIS * (trait - SETPOINT))

So a single vivid experience moves mood a lot and disposition barely; only a
repeated or sustained pattern of mood graduates into a lasting trait, and a trait
that stops being reinforced partially relaxes back toward SETPOINT. The push is
signed, so experiences can lower a trait as well as raise it.
"""

from config import STATE_DECAY, CONSOLIDATION_RATE, HOMEOSTASIS, SETPOINT


def clamp(value, minimum=-1.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def update_personality(personality, push):
    """Return a new personality with mood and disposition advanced (input unchanged)."""
    traits = {}
    for dim, layers in personality["traits"].items():
        delta = push.get(dim, 0.0)

        state = clamp(layers["state"] * (1 - STATE_DECAY) + delta)
        trait = clamp(layers["trait"] + CONSOLIDATION_RATE * state * (1 - abs(layers["trait"])))
        trait = clamp(trait - HOMEOSTASIS * (trait - SETPOINT))

        traits[dim] = {"state": state, "trait": trait}

    return {
        "traits": traits,
        "experience_count": personality.get("experience_count", 0) + 1,
    }
