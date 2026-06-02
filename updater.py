def clamp(value, minimum=-1.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def apply_trait_changes(personality, trait_changes):
    for trait, delta in trait_changes.items():
        current = personality[trait]

        effective_delta = (
                delta *
                (1 - max(0, current * delta))
        )

        personality[trait] += effective_delta

        personality[trait] = clamp(
            personality[trait]
        )

    return personality


import json

PERSONALITY_FILE = "data/personality.json"


def load_personality():
    with open(PERSONALITY_FILE, "r") as f:
        return json.load(f)


def save_personality(personality):
    with open(PERSONALITY_FILE, "w") as f:
        json.dump(personality, f, indent=4)
