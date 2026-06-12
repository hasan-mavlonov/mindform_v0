"""Experience -> signed push on the five traits.

    salience = intensity · (0.5 + 0.5·self_relevance) · (0.5 + 0.5·novelty)
    pull     = M · appraisal                       # which traits move, signed
    push[k]  = clamp( FORMATION_RATE · salience · pull[k] )

The push is how much the experience *would* move each trait; updater.py then
applies it with diminishing returns. Impact reads only the appraisal (what the
experience means), never a trait-expression reading of the text. ``M`` (config.M)
is the theory-authored appraisal->trait projection, replaceable by a learned map.
"""

from config import BASIS, M, FORMATION_RATE


def clamp(value, minimum=-1.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def rule_pull(appraisal, basis=BASIS, matrix=M):
    """Project the appraisal vector onto a basis via a prior matrix (default: traits)."""
    return {
        dim: clamp(sum(weight * appraisal.get(col, 0.0)
                       for col, weight in matrix.get(dim, {}).items()))
        for dim in basis
    }


def impact(appraisal, basis=BASIS, matrix=M, rate=FORMATION_RATE):
    """Return the signed per-dimension push for an experience's appraisal.

    Defaults project onto the OCEAN traits (``config.M``). Pass ``basis`` / ``matrix``
    / ``rate`` to reuse the very same salience-scaled, clamped projection for another
    substrate -- e.g. the Schwartz values (``config.VALUES`` / ``config.VALUES_M``),
    which is the heuristic fallback for character formation.
    """
    salience = (
        appraisal.get("intensity", 0.0)
        * (0.5 + 0.5 * appraisal.get("self_relevance", 0.0))
        * (0.5 + 0.5 * appraisal.get("novelty", 0.0))
    )
    pull = rule_pull(appraisal, basis, matrix)
    return {dim: clamp(rate * salience * pull[dim]) for dim in basis}
