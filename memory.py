"""Persistent memory of experiences: text, embedding, appraisal, and push.

Memories are namespaced per character (``data/characters/<slug>.memories.json``)
so different characters keep separate recurrence histories. With no name they fall
back to a single shared log (``data/memories.json``), which the simulation demo uses.
"""

import json
import os

import numpy as np

from config import RECURRENCE_THRESHOLD
from personality import read_traits, _slug, CHARACTERS_DIR

DEFAULT_MEMORY_FILE = "data/memories.json"


def _memory_path(name=None):
    if not name:
        return DEFAULT_MEMORY_FILE
    return os.path.join(CHARACTERS_DIR, f"{_slug(name)}.memories.json")


def load_memories(name=None):
    path = _memory_path(name)
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def save_memories(memories, name=None):
    path = _memory_path(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(memories, f, indent=4)


def create_memory(text, embedding, appraisal, push, personality_after, name=None):
    """Append a new memory (namespaced to the character) and return it."""
    if name is None:
        name = (personality_after.get("identity") or {}).get("name")
    emb = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
    memory = {
        "text": text,
        "embedding": emb,
        "appraisal": appraisal,
        "push": push,
        "traits_after": read_traits(personality_after),
    }
    memories = load_memories(name)
    memories.append(memory)
    save_memories(memories, name)
    return memory


def _cosine(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def recurrence(embedding, name=None):
    """How many of this character's past memories resemble this one."""
    return sum(
        1 for m in load_memories(name)
        if m.get("embedding")
        and _cosine(embedding, m["embedding"]) >= RECURRENCE_THRESHOLD
    )
