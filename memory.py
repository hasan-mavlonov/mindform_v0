"""Persistent memory of experiences: text, embedding, appraisal, and push."""

import json
import os

import numpy as np

from config import RECURRENCE_THRESHOLD, RETRIEVAL_K
from personality import read_traits, read_state

MEMORY_FILE = "data/memories.json"


def load_memories():
    if not os.path.exists(MEMORY_FILE):
        save_memories([])
        return []
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)


def save_memories(memories):
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    with open(MEMORY_FILE, "w") as f:
        json.dump(memories, f, indent=4)


def create_memory(text, embedding, appraisal, push, personality_after):
    """Append a new memory and return it."""
    emb = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
    memory = {
        "text": text,
        "embedding": emb,
        "appraisal": appraisal,
        "push": push,
        "traits_after": read_traits(personality_after),
        "state_after": read_state(personality_after),
    }
    memories = load_memories()
    memories.append(memory)
    save_memories(memories)
    return memory


def _cosine(a, b):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)

    if a.shape != b.shape:
        return 0.0

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    return float(a @ b / (na * nb)) if na and nb else 0.0

def _matches(embedding):
    """Past memories above the recurrence threshold, most similar first."""
    scored = []
    for m in load_memories():
        emb = m.get("embedding")
        if emb:
            sim = _cosine(embedding, emb)
            if sim >= RECURRENCE_THRESHOLD:
                scored.append((sim, m))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored


def retrieve_similar(embedding, k=RETRIEVAL_K):
    """The k most similar past memories; their appraisals condition a new experience."""
    return [m for _, m in _matches(embedding)[:k]]


def recurrence(embedding):
    """How many past memories resemble this one (chronicity signal for formation)."""
    return len(_matches(embedding))
