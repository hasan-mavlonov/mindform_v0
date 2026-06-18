"""Apply an experience's signed push to the five traits, then the temperament dynamics.

Each experience moves a trait in three steps:

    1. FAST   -- the push forms the trait, with diminishing returns:
                   x[k] <- clamp(x[k] + push[k] * (1 - |x[k]|))
    2. MEDIUM -- temperament pulls the current trait back toward its baseline:
                   x[k] <- clamp(x[k] + tau[k] * (mu[k] - x[k]))
    3. SLOW   -- the baseline itself drifts toward the lived state:
                   mu[k] <- mu[k] + TEMPERAMENT_DRIFT * (x[k] - mu[k])

``tau`` is the fraction of the gap back to baseline a trait closes per experience
(high = resilient, low = easily reshaped); the drift rate is far smaller, so only a
*sustained* shift moves the baseline. With the pull on, a repeated experience settles a
trait at a fixed point *between* its baseline and the extreme (not at the +/-1 wall), so
two characters with the same experiences but different temperaments diverge.

``apply_diminishing`` (the FAST step) is shared with the Schwartz values
(``character.update_values``); values have no innate baseline, so only the traits get the
temperament pull/drift.
"""

from config import TEMPERAMENT_DRIFT


def clamp(value, minimum=-1.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def apply_diminishing(state, push, keys=None):
    """Move each dimension by its push with diminishing returns.

        state[k] <- clamp(state[k] + push[k] * (1 - |state[k]|))

    Shared by the OCEAN traits and the Schwartz values (``character``): same formation
    dynamics, different substrate. ``keys`` selects which dimensions to update and
    defaults to every key already present in ``state`` (pass the full basis to ensure
    missing dims are seeded). Returns a new dict; ``state`` is left unchanged.
    """
    keys = list(state) if keys is None else keys
    return {
        k: clamp(state.get(k, 0.0) + push.get(k, 0.0) * (1 - abs(state.get(k, 0.0))))
        for k in keys
    }


def relax_to_temperament(traits, temperament):
    """Pull each trait toward its baseline, then drift the baseline toward the trait.

        x[k]  <- clamp(x[k] + tau[k] * (mu[k] - x[k]))          # MEDIUM
        mu[k] <- mu[k] + TEMPERAMENT_DRIFT * (x[k] - mu[k])     # SLOW

    Returns ``(new_traits, new_mu)``. With no baseline/stickiness present the traits pass
    through unchanged and ``mu`` is returned as-is.
    """
    mu = temperament.get("mu") or {}
    tau = temperament.get("tau") or {}
    if not mu and not tau:
        return dict(traits), dict(mu)

    new_traits, new_mu = {}, dict(mu)
    for k, value in traits.items():
        x = clamp(value + tau.get(k, 0.0) * (mu.get(k, 0.0) - value))
        new_traits[k] = x
        if k in new_mu:
            new_mu[k] = new_mu[k] + TEMPERAMENT_DRIFT * (x - new_mu[k])
    return new_traits, new_mu


def update_personality(personality, push):
    """Return a new personality with the experience push + temperament dynamics applied.

    Input is left unchanged. Identity and the character layer ride through untouched;
    the baseline ``temperament.mu`` is updated by the slow drift.
    """
    formed = apply_diminishing(personality["traits"], push)            # FAST
    temperament = personality.get("temperament") or {}
    traits, mu = relax_to_temperament(formed, temperament)             # MEDIUM + SLOW

    result = {
        **personality,
        "traits": traits,
        "experience_count": personality.get("experience_count", 0) + 1,
    }
    if temperament:
        result["temperament"] = {**temperament, "mu": mu}
    return result
