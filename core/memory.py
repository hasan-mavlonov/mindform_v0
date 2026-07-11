"""Persistent memory of experiences: a complete text log + a compact embedding sidecar.

Each experience is logged per character (``data/characters/<slug>.memories.json``) as the
small, human-readable record -- ``text``, ``appraisal``, ``push``, ``traits_after`` -- so
the *full* history is always kept (nothing is forgotten). The bulky part, the 384-float
MiniLM embedding, lives separately in a compact binary sidecar
(``data/characters/<slug>.memories.embeddings.npy``), one row per memory, index-aligned
with the log. That keeps the JSON ~20-50x smaller while preserving exact recall.

With no name, memory falls back to a single shared log (``data/memories.json``), which the
simulation demo uses. Older logs that still carry inline ``embedding`` fields are migrated
into the sidecar on first touch -- losing nothing.

Today the only read is ``recurrence`` (how many past experiences resemble this one, for
habits / the "seen before" count); the sidecar matrix is also what a future ``recall``
API will query.
"""

import json
import os

import numpy as np

from core.config import RECURRENCE_THRESHOLD
from core.personality import read_traits, _slug, CHARACTERS_DIR

DEFAULT_MEMORY_FILE = "data/memories.json"


def _memory_path(name=None):
    if not name:
        return DEFAULT_MEMORY_FILE
    return os.path.join(CHARACTERS_DIR, f"{_slug(name)}.memories.json")


def _embeddings_path(name=None):
    return _memory_path(name)[:-len(".json")] + ".embeddings.npy"


# --- the text log (complete, human-readable) -------------------------------
def load_memories(name=None):
    path = _memory_path(name)
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def _write_memories(memories, name=None):
    path = _memory_path(name)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(memories, f, indent=4)


def save_memories(memories, name=None):
    """Write the (slim) text log. Embeddings are managed via the .npy sidecar."""
    _write_memories(memories, name)


# --- the embedding sidecar (compact binary, index-aligned with the log) ----
def _save_embeddings(matrix, name=None):
    path = _embeddings_path(name)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, matrix)


def _migrate_inline_embeddings(name=None):
    """One-time upgrade: lift legacy inline ``embedding`` fields into the sidecar and
    rewrite the log slim. Only runs when there's no sidecar yet and every memory has an
    embedding (so row<->memory alignment is exact); otherwise leaves the log untouched."""
    mpath, epath = _memory_path(name), _embeddings_path(name)
    if os.path.exists(epath) or not os.path.exists(mpath):
        return
    with open(mpath, "r") as f:
        memories = json.load(f)
    if not memories or any("embedding" not in m for m in memories):
        return
    _save_embeddings(np.asarray([m["embedding"] for m in memories], dtype=float), name)
    for m in memories:
        m.pop("embedding", None)
    _write_memories(memories, name)


def load_embeddings(name=None):
    """The N x D matrix of episode embeddings (migrating legacy logs first).

    Empty ``(0, 0)`` when the character has no memories yet.
    """
    _migrate_inline_embeddings(name)
    path = _embeddings_path(name)
    if not os.path.exists(path):
        return np.zeros((0, 0))
    return np.load(path)


def _append_embedding(embedding, name=None):
    row = np.asarray(embedding, dtype=float).reshape(1, -1)
    matrix = load_embeddings(name)
    matrix = row if matrix.size == 0 else np.vstack([matrix, row])
    _save_embeddings(matrix, name)


# --- writing + reading -----------------------------------------------------
def create_memory(text, embedding, appraisal, push, personality_after, name=None):
    """Append a new memory: its record to the log, its embedding to the sidecar."""
    if name is None:
        name = (personality_after.get("identity") or {}).get("name")
    memory = {
        "text": text,
        "appraisal": appraisal,
        "push": push,
        "traits_after": read_traits(personality_after),
    }
    # the stance this experience was met WITH (behavior's carried set gated its intake) --
    # stored per record so the act history accrues in the hub
    stance = ((personality_after.get("behavior") or {}).get("set")) or {}
    if stance:
        memory["stance"] = {"tendency": stance.get("tendency", 0.0),
                            "mode": stance.get("mode", "steady")}
    memories = load_memories(name)
    memories.append(memory)
    _write_memories(memories, name)
    _append_embedding(embedding, name)
    return memory


def _similarities(query, matrix):
    """Cosine of ``query`` against every row of ``matrix`` (zeros if empty)."""
    if matrix.size == 0:
        return np.zeros((0,))
    q = np.asarray(query, dtype=float).reshape(-1)
    qn = np.linalg.norm(q)
    if qn == 0:
        return np.zeros(matrix.shape[0])
    norms = np.linalg.norm(matrix, axis=1)
    sims = np.zeros(matrix.shape[0])
    nz = norms > 0
    sims[nz] = (matrix[nz] @ q) / (norms[nz] * qn)
    return sims


def recurrence(embedding, name=None):
    """How many of this character's past memories resemble this one."""
    sims = _similarities(embedding, load_embeddings(name))
    return int((sims >= RECURRENCE_THRESHOLD).sum())


def recall(query_embedding, name=None, k=3, min_score=0.25):
    """The k past memories most relevant to ``query_embedding`` (most similar first).

    Returns up to ``k`` memory records (the slim text-log entries) each tagged with a
    ``score`` (cosine), dropping anything below ``min_score`` so only genuinely related
    experiences surface. Empty when there's no relevant history -- the same offline-safe
    contract as ``recurrence`` (needs the sidecar + numpy). Call it BEFORE storing the
    current experience so a message never just recalls itself.
    """
    matrix = load_embeddings(name)
    if matrix.size == 0:
        return []
    sims = _similarities(query_embedding, matrix)
    memories = load_memories(name)
    n = min(len(memories), sims.shape[0])
    if n == 0:
        return []
    out = []
    for i in np.argsort(sims[:n])[::-1][:k]:        # top-k by similarity, descending
        score = float(sims[int(i)])
        if score < min_score:
            break                                   # the rest score no higher
        record = dict(memories[int(i)])
        record["score"] = score
        out.append(record)
    return out
