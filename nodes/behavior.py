"""BEHAVIOR & LIFE OUTCOME: the enacted stance -- the sink of the diagram, and the loop-closer.

Every other node reads the world or shapes the self; this one ACTS, and what its acting
brings back becomes new lived experience. Three levels, one node:

  * TWO FORMED SENSITIVITIES (Gray's two systems), persisted, slow:
    ``behavior["approach"]`` (BAS -- the gain on reward cues, anchored to extraversion/
    openness) and ``behavior["inhibition"]`` (BIS -- the gain on threat cues, anchored to
    neuroticism). Independent on purpose: both loud is CONFLICTED (the anxious striver),
    both quiet is apathy -- states one approach<->avoid axis cannot express. Set-points
    derive from the current traits each turn (like drive weights from values); the house
    dynamics apply (relax toward set-point, pushes through diminishing returns).
  * A CARRIED ACTION READINESS (Frijda), fast: ``behavior["set"] = {tendency, mode}`` --
    each event's reward/threat cues, through the sensitivities, yield a lean-in/pull-back
    reading that is blended with inertia and carried to the next turn.
  * THE ENACTED STANCE, per turn: the mode is voiced in the reply as an inclination
    (expression reads it), and ``note_act`` freezes the act -- stance + mode + the style
    it was spoken with -- as ``behavior["last"]``, the record the next turn's outcome is
    credited against.

THE RING (both edges bounded, see the stability notes):

  * INTAKE GATE (behavior -> foundational input): the carried tendency multiplies the next
    experience's INTENSITY (``cognition._behavior_tilt``, the first step of the lens).
    Withdrawal lets life land softer, so everything forms less -- the shy spiral -- and the
    spiral is structurally damped: the inhibition-training signal below is scaled by the
    already-gated intensity, so withdrawing attenuates the very signal that deepens it.
  * OPERANT CREDIT (life outcome -> disposition): rewarded OWN action teaches approach
    hardest (the ``0.3 + 0.7*agency`` contingency gate -- windfalls barely teach); threat
    trains inhibition twice as fast as safe mastered challenge extinguishes it
    (bad-is-stronger, mirroring the drives' frustration asymmetry); and a carried lean-IN
    earns ``reception`` credit -- the next incoming message read as how the act landed.
    There is deliberately no signed stance*reception product: kindness must never entrench
    shyness (withdrawal earns no credit of either sign).

Stability: every sensitivity sits in a contraction (BEHAV_TAU pull to a trait-anchored
set-point + diminishing returns), so constant bombardment has a fixed point strictly inside
(-1, 1); the intake gate is a memoryless multiplier in [1 - EXPOSURE, 1 + EXPOSURE]; the
round-trip loop gain per turn (~EXPOSURE * BEHAV_GAIN) is dominated by the pull-back. The
needs and esteem are NOT gated -- the ungated ache is the exit ramp from avoidance.

``reception()`` is exported for Expression Slice 2 (style reinforced by how utterances
land). Pure arithmetic; no LLM, no numpy -- it degrades exactly like every other node.
"""

from core.config import (
    BASIS, BEHAV, BEHAV_NAMES, BEHAV_MU, BEHAV_TAU, BEHAV_GAIN, BEHAV_CREDIT,
    BEHAV_INH_UP, BEHAV_INH_DOWN, BEHAV_SET_BLEND, BEHAV_EXPOSURE, BEHAV_ACTIVE_THRESH,
)
from core.updater import apply_diminishing


def _clamp(value, lo=-1.0, hi=1.0):
    return max(lo, min(hi, value))


def set_points(personality):
    """Each sensitivity's trait-anchored resting point (derived each turn, never stored)."""
    traits = (personality or {}).get("traits") or {}
    return {
        b: _clamp(sum(w * traits.get(k, 0.0) for k, w in BEHAV_MU[b].items()))
        for b in BEHAV
    }


def default_behavior(personality):
    """Birth: sensitivities AT their trait set-points (divergence is earned by living),
    readiness steady, no act yet."""
    sp = set_points(personality)
    return {"approach": sp["approach"], "inhibition": sp["inhibition"],
            "set": {"tendency": 0.0, "mode": "steady"}, "last": None}


def refresh(personality):
    """Relax each sensitivity toward its trait-anchored set-point (the disposition
    re-asserts; mirrors drives.refresh). The carried action set is left untouched --
    the stance you went to sleep with is the one the next experience meets."""
    b = personality.get("behavior") or default_behavior(personality)
    sp = set_points(personality)
    out = dict(b)
    for k in BEHAV:
        x = float(b.get(k, 0.0))
        out[k] = _clamp(x + BEHAV_TAU * (sp[k] - x))
    return {**personality, "behavior": out}


# --- the action reading: cues -> activations -> readiness -----------------------
def _cues(appraisal):
    """This event's appetitive and aversive pull, from the interpreted appraisal.
    (threat_challenge is signed: +challenge feeds approach, -threat feeds inhibition.)"""
    v = appraisal.get("valence", 0.0)
    tc = appraisal.get("threat_challenge", 0.0)
    s = appraisal.get("social", 0.0)
    cue_app = _clamp(0.5 * max(0.0, v) + 0.5 * max(0.0, tc) + 0.2 * max(0.0, s), 0.0, 1.0)
    cue_av = _clamp(0.7 * max(0.0, -tc) + 0.3 * max(0.0, -v), 0.0, 1.0)
    return cue_app, cue_av


def _activations(behavior, appraisal):
    """How loudly each system fires: its sensitivity (mapped to a [0,1] gain) x its cue."""
    cue_app, cue_av = _cues(appraisal)
    act_app = _clamp((0.5 + 0.5 * float(behavior.get("approach", 0.0))) * cue_app, 0.0, 1.0)
    act_inh = _clamp((0.5 + 0.5 * float(behavior.get("inhibition", 0.0))) * cue_av, 0.0, 1.0)
    return act_app, act_inh


def _mode(act_app, act_inh, tendency):
    t = BEHAV_ACTIVE_THRESH
    if act_app >= t and act_inh >= t:
        return "conflicted"                      # drawn in and braced at once
    if tendency >= t:
        return "approach"
    if tendency <= -t:
        return "withdraw"
    return "steady"


# --- the ring's two edges ---------------------------------------------------------
def intake(personality):
    """The engagement factor the carried stance applies to the next experience's
    intensity: in [1 - EXPOSURE, 1 + EXPOSURE]. 1.0 when steady or absent."""
    b = (personality or {}).get("behavior") or {}
    tendency = _clamp(float((b.get("set") or {}).get("tendency", 0.0)))
    return 1.0 + BEHAV_EXPOSURE * tendency


def reception(appraisal):
    """How the world answered the character's last act: the warmth of this turn's
    (interpreted) appraisal, weighted up when it is social -- i.e. plausibly addressed
    to them. Exported for Expression Slice 2."""
    v = appraisal.get("valence", 0.0)
    s = appraisal.get("social", 0.0)
    return _clamp(v * (0.6 + 0.4 * max(0.0, s)))


def apply_event(personality, appraisal):
    """AFTER formation: the operant step (the world's answer trains the sensitivities),
    then the new action readiness is read and blended into the carried set.

    ``appraisal`` is the interpreted -- and therefore already intake-gated -- reading,
    which is what structurally damps the spiral: a withdrawn character's softened
    intensity also weakens the inhibition-training signal below.
    """
    b = dict(personality.get("behavior") or default_behavior(personality))
    last = b.get("last")
    carried = float((last or {}).get("tendency", 0.0))
    rec = reception(appraisal) if last is not None else None

    # approach: rewarded OWN action teaches hardest; a carried lean-IN earns reception
    # credit. No signed stance*reception product -- withdrawal earns no credit either way.
    reward = _clamp(0.6 * appraisal.get("outcome", 0.0) + 0.4 * appraisal.get("valence", 0.0))
    acted = max(0.0, appraisal.get("agency", 0.0))
    credit = BEHAV_CREDIT * max(0.0, carried) * rec if rec is not None else 0.0
    push_app = _clamp(BEHAV_GAIN * ((0.3 + 0.7 * acted) * reward + credit))

    # inhibition: threat trains it (scaled by the gated intensity -- the damping), and
    # safe mastered challenge (challenge met with positive affect) extinguishes it slower.
    tc = appraisal.get("threat_challenge", 0.0)
    v = appraisal.get("valence", 0.0)
    inten = appraisal.get("intensity", 0.0)
    push_inh = _clamp(BEHAV_GAIN * (0.5 + 0.5 * inten)
                      * (BEHAV_INH_UP * max(0.0, -tc)
                         - BEHAV_INH_DOWN * max(0.0, tc) * max(0.0, v)))

    formed = apply_diminishing({k: float(b.get(k, 0.0)) for k in BEHAV},
                               {"approach": push_app, "inhibition": push_inh}, BEHAV)
    nb = {**b, **formed}

    # the fresh sensitivities read this event's cues -> new readiness, blended with inertia
    act_app, act_inh = _activations(nb, appraisal)
    tendency_now = _clamp(act_app - act_inh)
    old = float((b.get("set") or {}).get("tendency", 0.0))
    tendency = _clamp(old + BEHAV_SET_BLEND * (tendency_now - old))
    nb["set"] = {"tendency": tendency, "mode": _mode(act_app, act_inh, tendency)}
    nb["reception"] = rec
    return {**personality, "behavior": nb}


def note_act(personality, style=None):
    """Freeze the act being emitted this turn -- the stance and mode the reply speaks
    with, plus the expression style it is spoken in -- as ``behavior["last"]``: the
    record next turn's outcome is credited against (and Expression Slice 2's contract)."""
    b = dict(personality.get("behavior") or default_behavior(personality))
    s = b.get("set") or {}
    b["last"] = {
        "tendency": float(s.get("tendency", 0.0)),
        "mode": s.get("mode", "steady"),
        "style": dict(style or {}),
        "turn": personality.get("experience_count", 0),
    }
    return {**personality, "behavior": b}


# --- Read-outs ---------------------------------------------------------------------
_MODE_PHRASE = {
    "approach": "leaning in",
    "withdraw": "holding back",
    "conflicted": "pulled two ways",
    "steady": "steady",
}


def read_behavior(personality, before=None):
    """Behavior snapshot for the UI: the two sensitivities vs their set-points, the
    carried readiness, this turn's reception, and the intake factor. ``before`` is the
    behavior state as it stood when the experience arrived -- the intake reported is the
    factor that actually gated THIS event (the carried stance), not next turn's."""
    b = personality.get("behavior") or default_behavior(personality)
    sp = set_points(personality)
    before = before or {}
    rows = [{
        "key": k,
        "label": BEHAV_NAMES[k],
        "value": float(b.get(k, 0.0)),
        "base": sp[k],
        "delta": float(b.get(k, 0.0)) - float(before.get(k, b.get(k, 0.0))),
    } for k in BEHAV]
    s = b.get("set") or {}
    mode = s.get("mode", "steady")
    gate_set = (before.get("set") if isinstance(before.get("set"), dict) else None) or s
    gate = 1.0 + BEHAV_EXPOSURE * _clamp(float(gate_set.get("tendency", 0.0)))
    return {
        "sensitivities": rows,
        "set": {"tendency": float(s.get("tendency", 0.0)), "mode": mode},
        "line": _MODE_PHRASE.get(mode, "steady"),
        "reception": b.get("reception"),
        "intake": gate,
    }
