"""Persistent OCEAN personality state for MindForm.

The personality is the long-lived state of the agent. Each trait is a value in
the range [-1, 1], starting from a neutral 0.0. State is persisted to disk as
JSON so it survives across interactions.
"""

import json
import os

PERSONALITY_FILE = "data/personality.json"

TRAITS = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)

DEFAULT_PERSONALITY = {trait: 0.0 for trait in TRAITS}


def load_personality():
    """Load the personality state, creating a neutral default if none exists."""
    if not os.path.exists(PERSONALITY_FILE):
        save_personality(dict(DEFAULT_PERSONALITY))
        return dict(DEFAULT_PERSONALITY)

    with open(PERSONALITY_FILE, "r") as f:
        return json.load(f)


def save_personality(personality):
    """Persist the personality state to disk."""
    os.makedirs(os.path.dirname(PERSONALITY_FILE), exist_ok=True)
    with open(PERSONALITY_FILE, "w") as f:
        json.dump(personality, f, indent=4)
