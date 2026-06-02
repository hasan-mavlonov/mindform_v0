"""Persistent personality state: five OCEAN traits, each in [-1, 1].

A trait starts neutral at 0.0. Experiences push traits via signed deltas applied
with diminishing returns (see updater.py), so a trait moves fast while near 0 and
ever more slowly as it approaches +/-1. State persists to disk as JSON.
"""

import json
import os

from config import BASIS, BASIS_NAMES

PERSONALITY_FILE = "data/personality.json"


def default_personality():
    return {"traits": {d: 0.0 for d in BASIS}, "experience_count": 0}


def load_personality():
    if not os.path.exists(PERSONALITY_FILE):
        personality = default_personality()
        save_personality(personality)
        return personality
    with open(PERSONALITY_FILE, "r") as f:
        return migrate(json.load(f))


def save_personality(personality):
    os.makedirs(os.path.dirname(PERSONALITY_FILE), exist_ok=True)
    with open(PERSONALITY_FILE, "w") as f:
        json.dump(personality, f, indent=4)


def migrate(data):
    """Upgrade legacy formats (flat long-name dict, or the state/trait struct)."""
    if "traits" in data:
        return data

    personality = default_personality()
    if "dims" in data:                       # two-timescale struct -> trait only
        for key, dim in data["dims"].items():
            if key in personality["traits"]:
                personality["traits"][key] = dim.get("trait", 0.0)
    else:                                    # original flat {long_name: value}
        name_to_key = {v: k for k, v in BASIS_NAMES.items()}
        for name, value in data.items():
            key = name_to_key.get(name)
            if key is not None:
                personality["traits"][key] = value

    save_personality(personality)
    return personality


def read_traits(personality):
    """Human-readable read-out: {long_name: value}."""
    return {BASIS_NAMES[d]: v for d, v in personality["traits"].items()}
