"""SELF-CONCEPT & IDENTITY: the character's reflective model OF ITSELF.

Where every prior layer describes what the character *is*, this describes who it *thinks it
is* -- a model that can DIVERGE from reality. Two stored fields in ``personality["self"]``:

  * ``image`` -- a self-schema on the OCEAN basis (the perceived self). It forms by
    self-perception (Bem: it drifts toward the actual traits the character has formed) WITH
    self-verification resistance (Swann: drift that opposes the current self-view is damped),
    so the self-image LAGS reality -- the character still thinks it is the shy one long after
    it has become confident. The gap between ``image`` and the real ``traits`` is self-deception.
  * ``esteem`` -- a scalar self-regard in [-1, 1]. A sociometer: it rides the competence +
    relatedness needs (``drives.satisfaction``), so success/acceptance lift it and
    failure/rejection lower it, and it relaxes toward a dispositional baseline set by
    temperament + achievement (``seed_base``) -- a calm high-achiever rests secure, an anxious
    one rests low; the gap between esteem and its base is impostor / inflated self-regard.

Per turn (``engine_bridge.run_turn``), mirroring drives:
  * ``refresh``     -- relax esteem toward its baseline BEFORE interpret, so the self you carry
                       colours the read.
  * ``cognition._self_tilt`` -- reads image + esteem: events that CONTRADICT the self-view read
                       as threat + more self-relevant (dissonance); high esteem buffers threat.
  * ``apply_event`` -- AFTER formation: esteem responds to the interpreted experience and the
                       self-image drifts toward the just-formed traits (Bem + Swann).

Only ``image`` and ``esteem`` are persisted; ``base`` is derived each turn. Pure arithmetic
(no LLM, no numpy) -- it degrades exactly like every other node.
"""

from core.config import (
    BASIS, BASIS_NAMES, M, SELF_BASE, SCHEMA_LEARN, SCHEMA_RESIST, ESTEEM_GAIN, ESTEEM_RELAX,
)
from core.impact import rule_pull
from nodes.drives import satisfaction as drive_satisfaction


def _clamp(value, lo=-1.0, hi=1.0):
    return max(lo, min(hi, value))


def seed_base(personality):
    """The dispositional self-esteem set-point, from temperament (mu) + the achievement value.

    Low neuroticism, higher extraversion, and a formed achievement value lift chronic self-regard.
    """
    mu = ((personality.get("temperament") or {}).get("mu")) or {}
    values = ((personality.get("character") or {}).get("values")) or {}
    base = sum(w * mu.get(k, 0.0) for k, w in SELF_BASE["mu"].items())
    base += sum(w * values.get(k, 0.0) for k, w in SELF_BASE["values"].items())
    return _clamp(base)


def default_self(personality):
    """A newborn self: the self-image is an accurate mirror of the birth traits, and self-regard
    sits at its dispositional baseline -- divergence is then EARNED by living."""
    traits = personality.get("traits") or {}
    return {"image": {k: float(traits.get(k, 0.0)) for k in BASIS}, "esteem": seed_base(personality)}


def _image(self_state):
    img = (self_state or {}).get("image") or {}
    return {k: float(img.get(k, 0.0)) for k in BASIS}


def refresh(personality):
    """Relax self-esteem toward its dispositional baseline (the 'chronic regard re-asserts' step,
    mirroring drives.refresh / updater.relax_to_temperament). Returns a new personality."""
    self_state = personality.get("self") or default_self(personality)
    base = seed_base(personality)
    e = float(self_state.get("esteem", 0.0))
    relaxed = _clamp(e + ESTEEM_RELAX * (base - e))
    return {**personality, "self": {**self_state, "esteem": relaxed}}


def _self_evaluation(appraisal):
    """The sociometer signal: a self-relevance-weighted mean of competence + relatedness
    satisfaction. Reuses drives.satisfaction, so self-worth rides the needs already built --
    mastery and acceptance raise it, failure and rejection lower it."""
    sat = drive_satisfaction(appraisal)
    raw = 0.5 * (sat.get("competence", 0.0) + sat.get("relatedness", 0.0))
    gate = 0.3 + 0.7 * appraisal.get("self_relevance", 0.0)     # only self-relevant events touch worth
    return _clamp(raw * gate)


def apply_event(personality, appraisal, traits):
    """AFTER formation: move self-esteem by the sociometer signal (diminishing returns), and
    drift the self-image toward the just-formed ``traits`` (Bem), resisting moves that oppose the
    current self-view (Swann). ``traits`` are the POST-update traits. Returns a new personality."""
    self_state = personality.get("self") or default_self(personality)
    e = float(self_state.get("esteem", 0.0))
    e = _clamp(e + ESTEEM_GAIN * _self_evaluation(appraisal) * (1.0 - abs(e)))

    img = _image(self_state)
    new_img = {}
    for k in BASIS:
        gap = float(traits.get(k, 0.0)) - img[k]
        # Swann: keep only a fraction of the drift when it would push the self-view across its own
        # sign (disconfirming evidence is resisted); a confirming move flows at the full rate.
        rate = SCHEMA_LEARN * (SCHEMA_RESIST if gap * img[k] < 0 else 1.0)
        new_img[k] = _clamp(img[k] + rate * gap)
    return {**personality, "self": {"image": new_img, "esteem": e}}


def self_signal(self_state, appraisal):
    """How this experience sits with the self-view: ``align`` > 0 affirms who they think they are,
    < 0 contradicts it. Used for the cockpit flash (the tilt computes this too)."""
    img = _image(self_state)
    if sum(abs(v) for v in img.values()) <= 0.0:
        return {"align": 0.0}
    implied = rule_pull(appraisal, BASIS, M)       # the trait direction this event implies
    return {"align": _clamp(sum(img[k] * implied[k] for k in BASIS))}


# --- Read-outs -----------------------------------------------------------------
def read_self(personality):
    """Self-concept rows for the UI: per-OCEAN self-image vs the actual trait (the gap is
    self-deception), plus esteem and its dispositional baseline."""
    self_state = personality.get("self") or default_self(personality)
    img = _image(self_state)
    traits = personality.get("traits") or {}
    rows = [{"key": k, "label": BASIS_NAMES[k], "image": img[k], "actual": float(traits.get(k, 0.0))}
            for k in BASIS]
    return {"image": rows, "esteem": float(self_state.get("esteem", 0.0)), "base": seed_base(personality)}


def incongruence(personality):
    """Mean absolute gap between the self-image and the actual traits (how self-deceived they are)."""
    self_state = personality.get("self") or {}
    img = _image(self_state)
    traits = personality.get("traits") or {}
    return sum(abs(img[k] - float(traits.get(k, 0.0))) for k in BASIS) / len(BASIS)
