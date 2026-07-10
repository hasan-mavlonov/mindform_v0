"""MOTIVATION & DRIVES: what pushes the character -- the need state that makes it *want*.

Three Self-Determination Theory basic needs -- autonomy, competence, relatedness -- each
carried as a fluctuating TENSION in [0, 1]: how unmet / loud the need is right now. Where
TEMPERAMENT is the innate OCEAN baseline and CHARACTER (the Schwartz values) is what the
person chronically prizes, DRIVES are the fast motivational STATE. WHICH needs matter is
seeded from the formed values (``DRIVE_SEED``); how starved each is rises and falls turn to
turn -- so the character comes to *want* things, not just react.

Per turn (``engine_bridge.run_turn``):

  * ``refresh``      -- recompute each need's resting WEIGHT from the current values (floored,
                        so every need stays a little alive) and relax TENSION toward it: an
                        unmet need quietly rebuilds pressure. Mirrors
                        ``updater.relax_to_temperament`` -- tension is to weight what a trait
                        ``x`` is to its baseline ``mu``.
  * ``cognition._drive_tilt`` reads the tensions to colour the appraisal (goal-relevance +
                        goal-congruence): the WANTING enters perception there.
  * ``apply_event``  -- a satisfying experience drops tension below rest (temporary relief), a
                        frustrating one raises it; regeneration brings the need back over later
                        turns, so satisfaction fades and the need returns.

Only TENSION is persisted (``personality["drives"]``); WEIGHT is derived from the values each
turn. Pure arithmetic -- no LLM, no numpy -- so it degrades exactly like every other node.
"""

from core.config import (
    DRIVES, DRIVE_NAMES, DRIVE_SEED, DRIVE_SAT,
    DRIVE_WEIGHT_FLOOR, DRIVE_WEIGHT_SLOPE, DRIVE_REGEN, DRIVE_SAT_GAIN,
)
from core.impact import rule_pull


def _clamp01(value):
    return max(0.0, min(1.0, value))


def default_drives():
    """A blank motivational state: every need at zero tension (put at rest by ``refresh``)."""
    return {d: {"tension": 0.0} for d in DRIVES}


def seed_weights(values):
    """Each need's chronic resting weight, projected from the formed Schwartz values.

        weight[d] = clamp01(FLOOR + SLOPE * sum_v DRIVE_SEED[d][v] * values[v])

    The FLOOR encodes SDT's claim that all three needs matter to everyone, so a blank
    character starts with every need lively; the values projection modulates them up/down.
    """
    values = values or {}
    return {
        d: _clamp01(DRIVE_WEIGHT_FLOOR + DRIVE_WEIGHT_SLOPE
                    * sum(w * values.get(v, 0.0) for v, w in DRIVE_SEED[d].items()))
        for d in DRIVES
    }


def tensions(drives):
    """Current tension per need as a plain dict (missing -> 0.0)."""
    drives = drives or {}
    return {d: float((drives.get(d) or {}).get("tension", 0.0)) for d in DRIVES}


def rest_drives(values):
    """Drives at rest: each need's tension sitting at its resting weight (birth / backfill)."""
    return {d: {"tension": w} for d, w in seed_weights(values).items()}


def refresh(personality):
    """Recompute resting weights from the current values and relax tension toward them.

    The 'need re-asserts' step: with nothing to satisfy it, a cared-about need's tension
    drifts up toward its weight (and an over-relieved need drifts back up to rest). Returns
    a new personality (input unchanged); only the drive tensions change.
    """
    drives = personality.get("drives") or default_drives()
    values = ((personality.get("character") or {}).get("values")) or {}
    weight = seed_weights(values)
    t = tensions(drives)
    relaxed = {d: {"tension": _clamp01(t[d] + DRIVE_REGEN * (weight[d] - t[d]))} for d in DRIVES}
    return {**personality, "drives": relaxed}


def satisfaction(appraisal):
    """Signed per-need engagement of an appraisal: > 0 feeds the need, < 0 frustrates it.

    Uses ``DRIVE_SAT`` (agency->autonomy, outcome->competence, social->relatedness), NOT the
    values prior -- feeding a need is not the same as an event teaching a value's worth.
    """
    return rule_pull(appraisal, DRIVES, DRIVE_SAT)


def apply_event(drives, appraisal):
    """The fast step: a satisfying event lowers tension (relief), a frustrating one raises it.

    Returns new drives (input unchanged).
    """
    signal = satisfaction(appraisal)
    t = tensions(drives or default_drives())
    return {d: {"tension": _clamp01(t[d] - DRIVE_SAT_GAIN * signal[d])} for d in DRIVES}


# --- Read-outs -----------------------------------------------------------------
def read_drives(personality):
    """Per-need rows for the UI, in fixed order: name, weight (resting level), tension (current
    loudness). Weight is derived from the current values so a just-created character reads at
    rest even before its first turn."""
    drives = personality.get("drives")
    values = ((personality.get("character") or {}).get("values")) or {}
    weight = seed_weights(values)
    t = tensions(drives) if isinstance(drives, dict) else weight   # no state yet -> show at rest
    return [{"key": d, "name": DRIVE_NAMES[d], "weight": weight[d], "tension": t[d]} for d in DRIVES]


def dominant_drive(personality):
    """The loudest current need -- what most pushes the character right now."""
    rows = read_drives(personality)
    top = max(rows, key=lambda r: r["tension"])
    return {"key": top["key"], "name": top["name"], "tension": top["tension"]}
