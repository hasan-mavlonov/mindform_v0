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


def rule_pull(appraisal):
    """Project the appraisal vector onto the trait basis via the matrix M."""
    return {
        dim: clamp(sum(weight * appraisal.get(col, 0.0)
                       for col, weight in M.get(dim, {}).items()))
        for dim in BASIS
    }


def impact(appraisal):
    """Return the signed per-trait push for an experience's appraisal."""
    salience = (
        appraisal.get("intensity", 0.0)
        * (0.5 + 0.5 * appraisal.get("self_relevance", 0.0))
        * (0.5 + 0.5 * appraisal.get("novelty", 0.0))
    )
    pull = rule_pull(appraisal)
    return {dim: clamp(FORMATION_RATE * salience * pull[dim]) for dim in BASIS}
