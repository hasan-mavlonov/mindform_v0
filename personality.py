import json

PERSONALITY_FILE = "data/personality.json"


def load_personality():

    with open(PERSONALITY_FILE, "r") as f:
        return json.load(f)


def save_personality(personality):

    with open(PERSONALITY_FILE, "w") as f:
        json.dump(personality, f, indent=4)


def clamp(value, min_value=-1.0, max_value=1.0):
    return max(min_value, min(max_value, value))