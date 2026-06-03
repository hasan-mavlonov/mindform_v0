"""Experience -> signed push on the five traits.

    salience = intensity · (0.5 + 0.5·self_relevance) · (0.5 + 0.5·novelty)
    pull     = M · appraisal                       # which traits move, signed
    gain     = reactivity(appraisal, personality)  # who-he-was: adverse events land
                                                   # harder on an emotionally reactive agent
    push[k]  = clamp( FORMATION_RATE · gain · salience · pull[k] )

The push is how much the experience *would* move each trait; updater.py then
applies it with diminishing returns. Impact reads only the appraisal (what the
experience means), never a trait-expression reading of the text. ``M`` (config.M)
is the theory-authored appraisal->trait projection, replaceable by a learned map.
"""

from config import BASIS, M, FORMATION_RATE, REACTIVITY_N


def clamp(value, minimum=-1.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def rule_pull(appraisal):
    """Project the appraisal vector onto the trait basis via the matrix M."""
    return {
        dim: clamp(sum(weight * appraisal.get(col, 0.0)
                       for col, weight in M.get(dim, {}).items()))
        for dim in BASIS
    }


def reactivity(appraisal, personality):
    """How hard this event lands, given who the agent is. Emotionally reactive
    (high-N) agents are moved more by adverse (negative / threatening) events; for a
    calm agent (N <= 0) or a non-adverse event this is 1.0 (no amplification)."""
    if personality is None:
        return 1.0
    n = personality["traits"]["N"]["trait"]
    threat_load = (max(0.0, -appraisal.get("valence", 0.0))
                   + max(0.0, -appraisal.get("threat_challenge", 0.0)))
    return 1.0 + REACTIVITY_N * max(0.0, n) * threat_load


def impact(appraisal, personality=None):
    """Return the signed per-trait push for an experience's appraisal.

    ``personality`` (optional) makes the push state-dependent via ``reactivity``; with
    ``None`` it reduces to the plain, agent-independent push.
    """
    salience = (
        appraisal.get("intensity", 0.0)
        * (0.5 + 0.5 * appraisal.get("self_relevance", 0.0))
        * (0.5 + 0.5 * appraisal.get("novelty", 0.0))
    )
    gain = reactivity(appraisal, personality)
    pull = rule_pull(appraisal)
    return {dim: clamp(FORMATION_RATE * gain * salience * pull[dim]) for dim in BASIS}
