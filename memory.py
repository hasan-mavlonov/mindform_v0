"""Persistent storage of interactions and the traits inferred from them.

Each memory records the raw text of an interaction alongside the trait scores
predicted for it.
"""

import json
import os

MEMORY_FILE = "data/memories.json"


def load_memories():
    """Load all stored memories, creating an empty store if none exists."""
    if not os.path.exists(MEMORY_FILE):
        save_memories([])
        return []

    with open(MEMORY_FILE, "r") as f:
        return json.load(f)


def save_memories(memories):
    """Persist the full list of memories to disk."""
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    with open(MEMORY_FILE, "w") as f:
        json.dump(memories, f, indent=4)


def create_memory(text, traits):
    """Append a new memory and return it.

    A memory has the shape::

        {"text": "...", "traits": {"openness": ..., ...}}
    """
    memory = {"text": text, "traits": traits}

    memories = load_memories()
    memories.append(memory)
    save_memories(memories)

    return memory
