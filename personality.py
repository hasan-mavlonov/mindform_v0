"""Persistent personality state on two timescales.

Each of the five trait axes carries two values:

    state  -- fast "mood": every experience pushes it, and it decays toward 0.
    trait  -- slow "disposition": integrates sustained state, with diminishing
              returns, and relaxes back toward a set-point absent reinforcement.

Both live in [-1, 1] and start at 0.0. The slow ``trait`` is what we mean by
"personality"; the fast ``state`` is the current reaction. See updater.py for the
dynamics and config.py for the constants. State persists to disk as JSON.
"""

import json
import os

from config import BASIS, BASIS_NAMES

PERSONALITY_FILE = "data/personality.json"


def default_personality():
    return {
        "traits": {d: {"state": 0.0, "trait": 0.0} for d in BASIS},
        "experience_count": 0,
    }


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

    traits = data.get("traits")
    if isinstance(traits, dict):
        for key, value in traits.items():
            if key not in personality["traits"]:
                continue
            if isinstance(value, dict):                       # already two-timescale
                personality["traits"][key]["state"] = float(value.get("state", 0.0))
                personality["traits"][key]["trait"] = float(value.get("trait", 0.0))
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
