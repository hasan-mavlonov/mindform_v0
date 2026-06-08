"""Persistent personality state: identity, temperament, and the five OCEAN traits.

A character is *born* via genesis (temperament.py): a biography seeds immutable
identity facts plus a per-trait OCEAN baseline ``mu`` and stickiness ``tau``, and
the current traits start AT that baseline (x = mu). Experiences then push the
current traits via signed deltas applied with diminishing returns (updater.py).
A blank character (no genesis) starts neutral at 0. State persists as JSON.
"""

import json
import os

from config import BASIS, BASIS_NAMES, DEFAULT_TAU

PERSONALITY_FILE = "data/personality.json"


def default_temperament():
    """Neutral baseline: every trait set-point at 0, default stickiness."""
    return {
        "mu": {d: 0.0 for d in BASIS},
        "tau": {d: DEFAULT_TAU for d in BASIS},
    }


def default_personality():
    """A blank character: born at a neutral temperament (traits start at mu = 0)."""
    return {
        "identity": {},
        "temperament": default_temperament(),
        "traits": {d: 0.0 for d in BASIS},
        "experience_count": 0,
    }


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


def _ensure_temperament(personality):
    """Backfill identity + temperament onto a pre-temperament save.

    A character saved before temperament existed keeps its current traits as its
    baseline (mu = traits), so nothing jumps; stickiness gets the default.
    """
    changed = False
    if "identity" not in personality:
        personality["identity"] = {}
        changed = True
    if "temperament" not in personality:
        traits = personality.get("traits", {})
        personality["temperament"] = {
            "mu": {d: traits.get(d, 0.0) for d in BASIS},
            "tau": {d: DEFAULT_TAU for d in BASIS},
        }
        changed = True
    return changed


def migrate(data):
    """Upgrade legacy formats and backfill temperament on older saves."""
    if "traits" in data:                     # current shape (maybe pre-temperament)
        personality = data
    elif "dims" in data:                     # two-timescale struct -> trait only
        personality = default_personality()
        for key, dim in data["dims"].items():
            if key in personality["traits"]:
                personality["traits"][key] = dim.get("trait", 0.0)
    else:                                    # original flat {long_name: value}
        personality = default_personality()
        name_to_key = {v: k for k, v in BASIS_NAMES.items()}
        for name, value in data.items():
            key = name_to_key.get(name)
            if key is not None:
                personality["traits"][key] = value

    changed = _ensure_temperament(personality)
    if personality is not data or changed:
        save_personality(personality)
    return personality


def read_traits(personality):
    """Human-readable read-out: {long_name: value}."""
    return {BASIS_NAMES[d]: v for d, v in personality["traits"].items()}


def read_temperament(personality):
    """Human-readable temperament: {long_name: {"mu": baseline, "tau": stickiness}}."""
    temp = personality["temperament"]
    return {BASIS_NAMES[d]: {"mu": temp["mu"][d], "tau": temp["tau"][d]} for d in BASIS}
