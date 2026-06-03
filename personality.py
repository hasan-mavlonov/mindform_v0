"""Persistent per-agent personality state.

Each of the five trait axes carries:

    state  -- fast "mood": every experience pushes it, and it decays toward 0.
    trait  -- slow "disposition": integrates sustained state, with diminishing
              returns, and relaxes back toward this agent's CORE absent reinforcement.
    trait_var -- slow within-person variability of expression (Whole Trait Theory).

The agent also carries a per-axis ``core`` -- its innate temperament / baseline, the
"home" each trait relaxes toward (homeostasis). Two agents with different cores
diverge under identical experience: this is the "changed based on who he was" part.

All values live in [-1, 1]. The slow ``trait`` is what we mean by "personality"; the
fast ``state`` is the current reaction. See updater.py for the dynamics and config.py
for the constants. State persists to disk as JSON.
"""

import json
import os

from config import BASIS, BASIS_NAMES, SETPOINT

PERSONALITY_FILE = "data/personality.json"


def default_personality(core=None):
    """A blank-slate agent (every axis 0), or one seeded with a ``core`` temperament.

    ``core`` is an optional {axis: value} map (axis = single letter, e.g. "N"); axes
    left out default to SETPOINT. Disposition starts AT the core, so an agent begins
    as its temperament and drifts around it. Use ``new_agent`` for the common case.
    """
    core = core or {}
    core = {d: float(core.get(d, SETPOINT)) for d in BASIS}
    return {
        "traits": {d: {"state": 0.0, "trait": core[d], "trait_var": 0.0} for d in BASIS},
        "core": core,
        "experience_count": 0,
    }


def new_agent(core):
    """Spawn an agent with a temperament, e.g. ``new_agent({"N": 0.6, "E": -0.3})``
    for an anxious introvert. Sugar for ``default_personality(core)``."""
    return default_personality(core)


def load_personality():
    if not os.path.exists(PERSONALITY_FILE):
        personality = default_personality()
        save_personality(personality)
        return personality
    with open(PERSONALITY_FILE, "r") as f:
        personality = migrate(json.load(f))
    save_personality(personality)   # normalize the on-disk form
    return personality


def save_personality(personality):
    os.makedirs(os.path.dirname(PERSONALITY_FILE), exist_ok=True)
    with open(PERSONALITY_FILE, "w") as f:
        json.dump(personality, f, indent=4)


def migrate(data):
    """Normalize any historical format to the two-timescale {state, trait} struct.

    Handles: the current two-timescale form; the single-layer {"traits": {k: float}}
    form (the float becomes the slow trait, mood starts at 0); the older
    {"dims": {k: {"trait": ...}}} struct; and the original flat {long_name: value}.
    """
    personality = default_personality()
    personality["experience_count"] = data.get("experience_count", 0)

    core = data.get("core")
    if isinstance(core, dict):                                # preserve temperament
        for key in personality["core"]:
            if key in core:
                personality["core"][key] = float(core[key])

    traits = data.get("traits")
    if isinstance(traits, dict):
        for key, value in traits.items():
            if key not in personality["traits"]:
                continue
            if isinstance(value, dict):                       # already two-timescale
                personality["traits"][key]["state"] = float(value.get("state", 0.0))
                personality["traits"][key]["trait"] = float(value.get("trait", 0.0))
                personality["traits"][key]["trait_var"] = float(value.get("trait_var", 0.0))
            else:                                             # single-layer float
                personality["traits"][key]["trait"] = float(value)
        return personality

    if "dims" in data:                                        # legacy dims struct
        for key, dim in data["dims"].items():
            if key in personality["traits"]:
                personality["traits"][key]["trait"] = float(dim.get("trait", 0.0))
        return personality

    name_to_key = {v: k for k, v in BASIS_NAMES.items()}      # original flat names
    for name, value in data.items():
        key = name_to_key.get(name)
        if key in personality["traits"]:
            personality["traits"][key]["trait"] = float(value)
    return personality


def read_traits(personality):
    """Human-readable disposition (the slow trait): {long_name: value}."""
    return {BASIS_NAMES[d]: layers["trait"] for d, layers in personality["traits"].items()}


def read_state(personality):
    """Human-readable current mood (the fast state): {long_name: value}."""
    return {BASIS_NAMES[d]: layers["state"] for d, layers in personality["traits"].items()}


def read_core(personality):
    """Human-readable innate temperament / baseline each trait relaxes toward."""
    core = personality.get("core", {})
    return {BASIS_NAMES[d]: float(core.get(d, SETPOINT)) for d in BASIS}


def read_variability(personality):
    """Human-readable within-person variance of expression: {long_name: value}.

    The dispersion of the trait's state-density (Whole Trait Theory); its sqrt is
    the reported SD (see evaluation.observed_dispersion).
    """
    return {BASIS_NAMES[d]: layers.get("trait_var", 0.0)
            for d, layers in personality["traits"].items()}
