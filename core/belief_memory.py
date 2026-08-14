"""Persistent embeddings for the belief store: SEMANTIC memory made recallable.

Where ``core/memory.py`` is EPISODIC memory (a full experience log), this is SEMANTIC --
the open propositions a character comes to hold (``character["beliefs"]``, formed by
``nodes/beliefs.py``). Beliefs already live inside the character's own JSON save; this
module adds ONLY the missing piece, a compact embedding sidecar
(``data/characters/<slug>.beliefs.embeddings.npy``), index-aligned with
``character["beliefs"]``, so beliefs can be RECALLED by relevance -- "what do I already
believe that bears on this?" -- not just formed and displayed.

Unlike episodic memory, a row is appended only when a genuinely NEW belief is formed
(``nodes.character.update_beliefs`` never reorders or removes a belief, only reinforces an
existing one or appends a new one -- so the sidecar's append-only, index-aligned contract
holds exactly, the same guarantee ``core/memory.py`` relies on for its own log). There is
no legacy inline-embedding migration case here (beliefs never carried one) and no separate
text log to keep slim -- the belief list itself already lives in the character's JSON.

``recall_beliefs`` mirrors ``core.memory.recall``'s contract and reuses its cosine math
(``core.memory._similarities``) rather than duplicating it.
"""

import os
import threading

import numpy as np

from core.config import BELIEF_MIN_SCORE, BELIEF_RECALL_GAIN
from core.memory import _similarities
from core.personality import _slug, CHARACTERS_DIR

# Guards a belief-embedding append as one critical section -- the same class of desync
# risk core.memory.create_memory's own lock protects against, for the same reason
# (concurrent turns on the same character, on the cockpit's thread-per-turn server).
_write_lock = threading.Lock()


def _belief_embeddings_path(name=None):
    return os.path.join(CHARACTERS_DIR, f"{_slug(name)}.beliefs.embeddings.npy")


def load_belief_embeddings(name=None):
    """The N x D matrix of belief-statement embeddings. Empty ``(0, 0)`` when none yet."""
    path = _belief_embeddings_path(name)
    if not os.path.exists(path):
        return np.zeros((0, 0))
    return np.load(path)


def append_belief_embeddings(rows, name=None):
    """Append new belief-statement embeddings, in the same order those beliefs were just
    added to ``character["beliefs"]``. No-op on empty ``rows`` (the common case -- most
    experiences reinforce an existing belief or form none at all)."""
    rows = np.asarray(rows, dtype=float)
    if rows.size == 0:
        return
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    path = _belief_embeddings_path(name)
    with _write_lock:
        matrix = load_belief_embeddings(name)
        matrix = rows if matrix.size == 0 else np.vstack([matrix, rows])
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.save(path, matrix)


def recall_beliefs(query_embedding, character, name=None, k=3, min_score=BELIEF_MIN_SCORE):
    """The k beliefs most relevant to ``query_embedding`` (most firmly-held-and-relevant
    first). Mirrors ``core.memory.recall``'s shape: a raw-cosine floor (``min_score``)
    gates eligibility -- so only genuinely relevant beliefs are ever candidates -- then
    ties are broken by conviction (``BELIEF_RECALL_GAIN``, over ``|confidence|``): a
    firmly held belief comes to mind ahead of a shaky one at similar relevance. Empty when
    there's no belief history yet. Guarded against ``character["beliefs"]`` having grown
    past what the sidecar has caught up with (a belief added since the sidecar was last
    read is simply not yet recallable -- it still forms and displays normally).
    """
    beliefs = (character or {}).get("beliefs") or []
    matrix = load_belief_embeddings(name)
    n = min(len(beliefs), matrix.shape[0])
    if n == 0:
        return []
    sims = _similarities(query_embedding, matrix[:n])
    eligible = []
    for i in np.argsort(sims)[::-1]:                # descending by raw cosine
        score = float(sims[int(i)])
        if score < min_score:
            break                                    # the rest score no higher
        record = dict(beliefs[int(i)])
        record["score"] = score                      # honest cosine, never reweighted
        eligible.append(record)
    eligible.sort(
        key=lambda b: b["score"] * (1.0 + BELIEF_RECALL_GAIN * abs(float(b.get("confidence", 0.0)))),
        reverse=True,
    )
    return eligible[:k]
