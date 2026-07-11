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

Slice 2 adds the ASPIRED selves (Higgins' self-discrepancy theory), all derived, no new
state: the IDEAL self is who the formed values say they want to be (``SELF_IDEAL_M``), the
OUGHT self who the moral foundations say they should be (``SELF_OUGHT_M``). Falling short
of the ideal -- the image-vs-ideal gap, weighted by how strongly the ideal is held --
chronically depresses the esteem baseline (DEJECTION, ``effective_base``); falling short
of the ought adds vigilance to the reading (AGITATION, cognition's ``_self_tilt``). You can
only fall short of an ideal you actually hold: a blank character has no gap.

Only ``image`` and ``esteem`` are persisted; every baseline and aspiration is derived each
turn. Pure arithmetic (no LLM, no numpy) -- it degrades exactly like every other node.
"""

from core.config import (
    BASIS, BASIS_NAMES, M, SELF_BASE, SCHEMA_LEARN, SCHEMA_RESIST, ESTEEM_GAIN, ESTEEM_RELAX,
    SELF_IDEAL_M, SELF_OUGHT_M, SELF_GAP_SLOPE,
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


# --- Slice 2: the aspired selves (Higgins' self-discrepancy theory) -------------
def ideal_self(personality):
    """Who their formed VALUES say they want to be: the aspired OCEAN self, derived each
    turn (never stored). A blank character (values at 0) aspires to nothing yet."""
    values = ((personality.get("character") or {}).get("values")) or {}
    return rule_pull(values, BASIS, SELF_IDEAL_M)


def ought_self(personality):
    """Who their MORAL foundations say they should be (the duty-side aspired self)."""
    moral = ((personality.get("character") or {}).get("moral")) or {}
    return rule_pull(moral, BASIS, SELF_OUGHT_M)


def _weighted_gap(target, image):
    """How far the self-image falls short of an aspired self, weighted by how strongly
    each dimension of the aspiration is held -- you can only fall short of an ideal you
    actually hold, so a blank aspiration yields no gap."""
    total = sum(abs(target[k]) for k in BASIS)
    if total <= 0.0:
        return 0.0
    return sum(abs(target[k]) * abs(target[k] - image[k]) for k in BASIS) / total


def aspiration_gap(personality):
    """The actual-vs-IDEAL discrepancy (image vs who they want to be): dejection's source."""
    return _weighted_gap(ideal_self(personality), _image(personality.get("self")))


def ought_gap(personality):
    """The actual-vs-OUGHT discrepancy (image vs who they should be): agitation's source."""
    return _weighted_gap(ought_self(personality), _image(personality.get("self")))


def effective_base(personality):
    """The esteem baseline actually lived with: the dispositional set-point, chronically
    depressed by falling short of who they want to be (Higgins: the ideal-gap pulls the
    floor DOWN only; acceptance and mastery lift esteem above it, never the gap)."""
    return _clamp(seed_base(personality) - SELF_GAP_SLOPE * aspiration_gap(personality))


def default_self(personality):
    """A newborn self: the self-image is an accurate mirror of the birth traits, and self-regard
    sits at its dispositional baseline -- divergence is then EARNED by living."""
    traits = personality.get("traits") or {}
    return {"image": {k: float(traits.get(k, 0.0)) for k in BASIS}, "esteem": seed_base(personality)}


def _image(self_state):
    img = (self_state or {}).get("image") or {}
    return {k: float(img.get(k, 0.0)) for k in BASIS}


def refresh(personality):
    """Relax self-esteem toward its EFFECTIVE baseline -- the dispositional set-point,
    depressed by how far the self-image falls short of the ideal self (Higgins dejection).
    The 'chronic regard re-asserts' step, mirroring drives.refresh. Returns a new personality."""
    self_state = personality.get("self") or default_self(personality)
    base = effective_base(personality)
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
    self-deception) and vs the IDEAL self (the gap is aspiration), plus esteem against its
    EFFECTIVE baseline (the dispositional set-point, depressed by falling short)."""
    self_state = personality.get("self") or default_self(personality)
    img = _image(self_state)
    traits = personality.get("traits") or {}
    ideal = ideal_self(personality)
    rows = [{"key": k, "label": BASIS_NAMES[k], "image": img[k],
             "actual": float(traits.get(k, 0.0)), "ideal": ideal[k]}
            for k in BASIS]
    return {"image": rows, "esteem": float(self_state.get("esteem", 0.0)),
            "base": effective_base(personality),
            "aspiration_gap": aspiration_gap(personality),
            "ought_gap": ought_gap(personality)}


def incongruence(personality):
    """Mean absolute gap between the self-image and the actual traits (how self-deceived they are)."""
    self_state = personality.get("self") or {}
    img = _image(self_state)
    traits = personality.get("traits") or {}
    return sum(abs(img[k] - float(traits.get(k, 0.0))) for k in BASIS) / len(BASIS)
