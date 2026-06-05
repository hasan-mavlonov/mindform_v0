"""Persistent memory of experiences: text, embedding, appraisal, and push.

Memories are cached in-process after the first read, so a turn that calls both
recurrence() and create_memory() touches the disk once for reading plus one write,
instead of re-parsing the whole file on every call.
"""

import json
import os

import numpy as np

from config import RECURRENCE_THRESHOLD
from personality import read_traits

MEMORY_FILE = "data/memories.json"

_memories = None   # in-process cache; the running app is the only writer


def load_memories():
    """Return all memories, reading from disk only on the first call."""
    global _memories
    if _memories is None:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r") as f:
                _memories = json.load(f)
        else:
            _memories = []
            save_memories(_memories)
    return _memories


def save_memories(memories):
    global _memories
    _memories = memories
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    with open(MEMORY_FILE, "w") as f:
        json.dump(memories, f, indent=4)


def create_memory(text, embedding, appraisal, push, personality_after):
    """Append a new memory (cache + disk) and return it."""
    emb = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
    memory = {
        "text": text,
        "embedding": emb,
        "appraisal": appraisal,
        "push": push,
        "traits_after": read_traits(personality_after),
    }
    memories = load_memories()
    memories.append(memory)
    save_memories(memories)
    return memory


def _cosine(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def recurrence(embedding):
    """How many past memories resemble this one (for display / insight)."""
    return sum(
        1 for m in load_memories()
        if m.get("embedding")
        and _cosine(embedding, m["embedding"]) >= RECURRENCE_THRESHOLD
    )
