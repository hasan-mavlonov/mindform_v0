"""Persistent personality state: identity, temperament, and the five OCEAN traits.

A character is *born* via genesis/creation (temperament.py): identity facts plus a
per-trait OCEAN baseline ``mu`` and stickiness ``tau``; the current traits start AT
the baseline (x = mu). Experiences then push the traits with diminishing returns
(updater.py). State persists as JSON.

Two stores live here:
  * a single default character -> data/personality.json         (simulation.py demo)
  * a named-character roster    -> data/characters/<slug>.json   (interactive shell)
"""

import json
import os
import re

from core.config import BASIS, BASIS_NAMES, DEFAULT_TAU, VALUES, MORAL, DRIVES
from nodes.character import default_character
from nodes.drives import rest_drives
from nodes.self_concept import default_self, seed_base
from nodes.behavior import default_behavior

PERSONALITY_FILE = "data/personality.json"
CHARACTERS_DIR = "data/characters"


def default_temperament():
    """Neutral baseline: every trait set-point at 0, default stickiness."""
    return {
        "mu": {d: 0.0 for d in BASIS},
        "tau": {d: DEFAULT_TAU for d in BASIS},
    }


def default_personality():
    """A blank character: born at a neutral temperament (traits start at mu = 0)."""
    character = default_character()
    personality = {
        "identity": {},
        "temperament": default_temperament(),
        "traits": {d: 0.0 for d in BASIS},
        "character": character,
        "drives": rest_drives(character["values"]),   # needs seeded at their resting level
        "experience_count": 0,
    }
    personality["self"] = default_self(personality)   # self-image mirrors the birth traits
    personality["behavior"] = default_behavior(personality)   # stance at its trait set-points
    return personality


def _ensure_temperament(personality):
    """Backfill identity + temperament onto a pre-temperament save (mu = traits)."""
    if "identity" not in personality:
        personality["identity"] = {}
    if "temperament" not in personality:
        traits = personality.get("traits", {})
        personality["temperament"] = {
            "mu": {d: traits.get(d, 0.0) for d in BASIS},
            "tau": {d: DEFAULT_TAU for d in BASIS},
        }
    return personality


def _ensure_character(personality):
    """Backfill the character (Schwartz values + habits) onto a pre-character save."""
    character = personality.get("character")
    if not isinstance(character, dict):
        personality["character"] = default_character()
    else:
        character.setdefault("habits", [])
        character.setdefault("beliefs", [])
        character.setdefault("beliefs_reviewed", 0)
        values = character.setdefault("values", {})
        for v in VALUES:                      # seed any missing value at neutral
            values.setdefault(v, 0.0)
        moral = character.setdefault("moral", {})
        for m in MORAL:                       # seed any missing foundation at neutral
            moral.setdefault(m, 0.0)
    return personality


def _ensure_drives(personality):
    """Backfill the motivational drive state onto a pre-drives save (needs at rest = weight)."""
    values = ((personality.get("character") or {}).get("values")) or {}
    drives = personality.get("drives")
    if not isinstance(drives, dict):
        personality["drives"] = rest_drives(values)
    else:
        rest = rest_drives(values)
        for d in DRIVES:                      # seed any missing need at its resting level
            drives.setdefault(d, rest[d])
    return personality


def _ensure_self(personality):
    """Backfill the self-concept onto a pre-self save (self-image = current traits, esteem at base)."""
    self_state = personality.get("self")
    if not isinstance(self_state, dict) or "image" not in self_state:
        personality["self"] = default_self(personality)
    else:
        image = self_state.setdefault("image", {})
        traits = personality.get("traits") or {}
        for d in BASIS:                        # seed any missing OCEAN self-image dim from the trait
            image.setdefault(d, float(traits.get(d, 0.0)))
        self_state.setdefault("esteem", seed_base(personality))
    return personality


def _ensure_behavior(personality):
    """Backfill the behavior state onto a pre-behavior save (sensitivities at their
    trait-anchored set-points, readiness steady, no act yet)."""
    if not isinstance(personality.get("behavior"), dict):
        personality["behavior"] = default_behavior(personality)
    return personality


def migrate(data):
    """Upgrade legacy formats and backfill temperament. Pure -- the caller persists."""
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
    return _ensure_behavior(
        _ensure_self(_ensure_drives(_ensure_character(_ensure_temperament(personality)))))


def _read(path):
    with open(path, "r") as f:
        return migrate(json.load(f))


def _write(path, personality):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(personality, f, indent=4)


# --- Single default character (data/personality.json) ----------------------
def load_personality():
    if not os.path.exists(PERSONALITY_FILE):
        personality = default_personality()
        save_personality(personality)
        return personality
    return _read(PERSONALITY_FILE)


def save_personality(personality):
    _write(PERSONALITY_FILE, personality)


# --- Named-character roster (data/characters/<slug>.json) -------------------
def _slug(name):
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return slug or "unnamed"


def character_path(name):
    return os.path.join(CHARACTERS_DIR, _slug(name) + ".json")


def save_character(personality):
    """Save a character to the roster, keyed by its identity name. Returns the path."""
    name = (personality.get("identity") or {}).get("name")
    path = character_path(name)
    _write(path, personality)
    return path


def load_character(name):
    return _read(character_path(name))


def list_characters():
    """Every saved character (migrated), sorted by file name."""
    if not os.path.isdir(CHARACTERS_DIR):
        return []
    characters = []
    for filename in sorted(os.listdir(CHARACTERS_DIR)):
        if filename.endswith(".json") and not filename.endswith(".memories.json"):
            characters.append(_read(os.path.join(CHARACTERS_DIR, filename)))
    return characters


def read_traits(personality):
    """Human-readable read-out: {long_name: value}."""
    return {BASIS_NAMES[d]: v for d, v in personality["traits"].items()}


def read_temperament(personality):
    """Human-readable temperament: {long_name: {"mu": baseline, "tau": stickiness}}."""
    temp = personality["temperament"]
    return {BASIS_NAMES[d]: {"mu": temp["mu"][d], "tau": temp["tau"][d]} for d in BASIS}
