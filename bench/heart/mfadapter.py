"""MindForm side of the HEART integration: create, ingest, freeze, recall.

Nothing in core/, nodes/ or web/ is modified. Two runtime accommodations, both
confined to this process and both documented:

  1. ``generate_reply`` is patched to a no-op during bulk ingestion. It is
     called by ``run_turn`` AFTER ``save_character`` and its output only enters
     the returned read-out, never persisted state -- so skipping it cannot
     change formation, and it saves one LLM call per memory.
  2. ``recall`` is called with ``min_score=0.0`` so it returns a full top-k the
     way naive RAG does, instead of stopping at the product's 0.25 relevance
     floor. Logged per retrieval as ``min_score``.
"""

import hashlib
import json
import os
import shutil
import time

from core.config import BASIS
from core.personality import (
    default_personality, save_character, load_character, character_path, read_traits,
)

# The three (four with beliefs) files that constitute a character's whole state.
def state_files(name):
    base = character_path(name)                       # data/characters/<slug>.json
    stem = base[: -len(".json")]
    return [
        base,
        f"{stem}.memories.json",
        f"{stem}.memories.embeddings.npy",
        f"{stem}.beliefs.embeddings.npy",
    ]


# --------------------------------------------------------------------------
# creation
# --------------------------------------------------------------------------
def create_neutral(name, occupation):
    """A blank MindForm character: every trait at 0.0, nothing inferred.

    HEART's withheld big_five is NOT used -- build_character(mu=...) would inject
    exactly the label the benchmark withholds. Occupation is the one substantive
    fact HEART's own baselines expose, so it is allowed here too.
    """
    p = default_personality()
    p["identity"] = {"name": name, "occupation": occupation}
    for path in state_files(name):
        if os.path.exists(path):
            os.remove(path)
    save_character(p)
    return p


# --------------------------------------------------------------------------
# ingestion
# --------------------------------------------------------------------------
class _ReplyPatch:
    """Disable reply generation for the duration of bulk ingestion."""

    def __enter__(self):
        import web.engine_bridge as eb
        self._eb = eb
        self._orig = eb.generate_reply
        eb.generate_reply = lambda *a, **k: ("", "skipped-for-ingestion")
        return self

    def __exit__(self, *exc):
        self._eb.generate_reply = self._orig
        return False


class NoThinkingPatch:
    """Inject ``reasoning_effort="none"`` into every chat call made in-process.

    The default Gemini model is a thinking model, and ``core/llm.py`` -- which is
    production code and out of scope for this harness -- does not pass the flag.
    Measured on one appraisal call: 6.5 s and 1,893 total tokens with reasoning
    on, versus 1.0 s and 1,028 with it off (865 of those tokens are invisible
    reasoning).

    This patches the OpenAI SDK's create() at the boundary rather than
    reimplementing ``complete_json``, so prompts, retries, parsing, temperature
    and max_tokens stay byte-identical and the ONLY difference between a patched
    and unpatched ingestion is the one kwarg. That is what makes the A/B
    interpretable.

    It changes how personality forms, so it is opt-in and always recorded in the
    run manifest -- never a silent default.
    """

    def __init__(self, effort="none"):
        self.effort = effort

    def __enter__(self):
        from openai.resources.chat import completions as _c
        self._cls = _c.Completions
        self._orig = self._cls.create
        effort = self.effort

        def create(inner_self, *args, **kwargs):
            kwargs.setdefault("reasoning_effort", effort)
            return self._orig(inner_self, *args, **kwargs)

        self._cls.create = create
        return self

    def __exit__(self, *exc):
        self._cls.create = self._orig
        return False


def ingest_memory(name, text):
    """One memory through MindForm's normal experience pipeline. Returns the snapshot."""
    from web.engine_bridge import run_turn
    return run_turn(name, text)


def trait_vector(personality):
    return {d: round(float(personality["traits"].get(d, 0.0)), 4) for d in BASIS}


def ingest_all(name, memories, logbook, on_step=None):
    """Ingest chronologically, logging state before/after each step.

    ``memories`` comes from heartdata.ingestible_memories -- anonymised ids and
    raw content_full only.
    """
    results = []
    with _ReplyPatch():
        for i, mem in enumerate(memories):
            p_before = load_character(name)
            before = trait_vector(p_before)
            t0 = time.time()
            err = None
            try:
                snap = ingest_memory(name, mem["text"])
            except Exception as exc:                       # keep the run alive, log it
                snap, err = None, f"{type(exc).__name__}: {exc}"
            dt_ms = (time.time() - t0) * 1000.0
            p_after = load_character(name)
            after = trait_vector(p_after)
            delta = {d: round(after[d] - before[d], 4) for d in BASIS}

            rec = {
                "memory_index": mem["original_index"],
                "chrono_position": mem["chrono_position"],
                "anon_id": mem["anon_id"],
                "timeline": mem["timeline"],
                "text_sha256": mem["text_sha256"],
                "text_chars": len(mem["text"]),
                "traits_before": before,
                "traits_after": after,
                "traits_delta": delta,
                "values_after": _top_values(p_after, 3),
                "drives_after": _drives(p_after),
                "esteem_after": round(float((p_after.get("self") or {}).get("esteem", 0.0)), 4),
                "experience_count": p_after.get("experience_count"),
                "appraisal_source": (snap or {}).get("appraisal_source"),
                "push_source": (snap or {}).get("source"),
                "latency_ms": round(dt_ms, 1),
                "error": err,
            }
            logbook.event("ingest_step", character=name, **rec)
            results.append(rec)
            if on_step:
                on_step(i, len(memories), mem, after, dt_ms)
    return results


# --------------------------------------------------------------------------
# snapshots
# --------------------------------------------------------------------------
def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_hash(name):
    """One hash over every state file. Missing files are recorded as absent."""
    h = hashlib.sha256()
    parts = {}
    for path in state_files(name):
        key = os.path.basename(path)
        if os.path.exists(path):
            d = _sha256_file(path)
            parts[key] = d
        else:
            d = "absent"
            parts[key] = d
        h.update(f"{key}:{d}|".encode())
    return h.hexdigest(), parts


def freeze(name, run_dir, label="frozen"):
    """Copy the character's state files into the run directory. Returns (id, manifest)."""
    dest = os.path.join(run_dir, "snapshots", label)
    os.makedirs(dest, exist_ok=True)
    copied = []
    for path in state_files(name):
        if os.path.exists(path):
            shutil.copy2(path, os.path.join(dest, os.path.basename(path)))
            copied.append(os.path.basename(path))
    sid, parts = snapshot_hash(name)
    manifest = {"snapshot_id": sid, "label": label, "files": parts,
                "copied": copied, "dir": dest}
    with open(os.path.join(dest, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return sid, manifest


def restore(name, run_dir, label="frozen"):
    """Restore the frozen state files over the live ones. Returns the snapshot id."""
    src = os.path.join(run_dir, "snapshots", label)
    for path in state_files(name):
        cand = os.path.join(src, os.path.basename(path))
        if os.path.exists(cand):
            shutil.copy2(cand, path)
        elif os.path.exists(path):
            os.remove(path)
    return snapshot_hash(name)[0]


# --------------------------------------------------------------------------
# read-only retrieval + state read-out
# --------------------------------------------------------------------------
def recall_memories(name, query_text, k, min_score=0.0, use_drive_bias=True):
    """MindForm's own recall. Read-only: nothing is stored, nothing is saved.

    Returns (records, meta). Each record carries the honest cosine in ``score``.
    """
    from core.encoder import encode_text
    from core.memory import recall as mf_recall

    personality = load_character(name)
    turn = personality.get("experience_count")
    emb = encode_text(query_text)
    pool = mf_recall(emb, name=name, k=k, min_score=min_score, current_turn=turn)

    biased = False
    if use_drive_bias:
        try:
            from nodes.drives import recall_bias
            pool = recall_bias(pool, personality.get("drives"), k=k)
            biased = True
        except Exception:
            pass

    out = []
    for r in pool:
        out.append({
            "anon_id": _match_anon_id(r),
            "timeline": _match_timeline(r),
            "text": r.get("text", ""),
            "score": round(float(r.get("score", 0.0)), 6),
            "intensity": (r.get("appraisal") or {}).get("intensity"),
            "turn": r.get("turn"),
        })
    meta = {
        "retriever": "mindform.recall",
        "embedder": "all-MiniLM-L6-v2 (core.encoder)",
        "k": k, "min_score": min_score, "drive_bias": biased,
        "current_turn": turn,
    }
    return out, meta


def naive_rag(name, query_text, k):
    """Plain cosine top-k over the SAME embeddings MindForm stored.

    This is the fairness control: arm A and arm D1 then differ only in ranking,
    with the embedder held constant. It is not HEART's published naive RAG
    (which uses qwen3-embedding-4b) -- that difference is logged, never hidden.
    """
    import numpy as np
    from core.encoder import encode_text
    from core.memory import load_embeddings, load_memories

    matrix = load_embeddings(name)
    memories = load_memories(name)
    if matrix.size == 0 or not memories:
        return [], {"retriever": "naive_rag", "embedder": "all-MiniLM-L6-v2", "k": k}

    q = np.asarray(encode_text(query_text), dtype=float)
    m = np.asarray(matrix, dtype=float)
    denom = (np.linalg.norm(m, axis=1) * (np.linalg.norm(q) or 1.0))
    denom[denom == 0] = 1.0
    sims = (m @ q) / denom

    n = min(len(memories), sims.shape[0])
    order = np.argsort(sims[:n])[::-1][:k]
    out = []
    for i in order:
        rec = memories[int(i)]
        out.append({
            "anon_id": _match_anon_id(rec),
            "timeline": _match_timeline(rec),
            "text": rec.get("text", ""),
            "score": round(float(sims[int(i)]), 6),
            "intensity": (rec.get("appraisal") or {}).get("intensity"),
            "turn": rec.get("turn"),
        })
    meta = {"retriever": "naive_rag_cosine", "embedder": "all-MiniLM-L6-v2 (core.encoder)",
            "k": k, "min_score": None, "drive_bias": False}
    return out, meta


# MindForm stores memory text, not HEART ids. We keep a text-sha256 -> anon_id
# map on the bench side so every retrieval trace names the HEART memory it came
# from, without the engine having to know anything about HEART.
_ID_MAP = {}


def register_id_map(memories):
    """memories: output of heartdata.ingestible_memories."""
    for m in memories:
        _ID_MAP[m["text_sha256"]] = {"anon_id": m["anon_id"], "timeline": m["timeline"]}


def _match_anon_id(record):
    text = record.get("text", "")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    hit = _ID_MAP.get(digest)
    return hit["anon_id"] if hit else "sha256:" + digest[:16]


def _match_timeline(record):
    digest = hashlib.sha256(record.get("text", "").encode("utf-8")).hexdigest()
    hit = _ID_MAP.get(digest)
    return hit["timeline"] if hit else "?"


def _top_values(personality, n):
    vals = ((personality.get("character") or {}).get("values")) or {}
    top = sorted(vals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:n]
    return [(k, round(float(v), 3)) for k, v in top if abs(v) > 1e-6]


def _drives(personality):
    """MindForm stores drives as {need: {"tension": float}} -- flatten to the tension."""
    d = personality.get("drives") or {}
    out = {}
    for k, v in d.items():
        val = v.get("tension") if isinstance(v, dict) else v
        try:
            out[k] = round(float(val), 3)
        except (TypeError, ValueError):
            continue
    return out


def _behavior(personality):
    """The two formed sensitivities plus the carried stance."""
    b = personality.get("behavior") or {}
    out = {k: round(float(v), 3) for k, v in b.items() if isinstance(v, (int, float))}
    stance = b.get("set") or {}
    if isinstance(stance, dict):
        if isinstance(stance.get("tendency"), (int, float)):
            out["tendency"] = round(float(stance["tendency"]), 3)
        if stance.get("mode"):
            out["mode"] = stance["mode"]
    return out


def state_summary(name, top_n=3):
    """The persistent state D2 exposes. Read-only."""
    p = load_character(name)
    beliefs = [b.get("statement", "") for b in ((p.get("character") or {}).get("beliefs") or [])]
    beliefs = [b for b in beliefs if b][:5]
    moral = ((p.get("character") or {}).get("moral")) or {}
    return {
        "traits": trait_vector(p),
        "top_values": _top_values(p, top_n),
        "top_moral": sorted(((k, round(float(v), 3)) for k, v in moral.items()
                             if abs(v) > 1e-6), key=lambda kv: -abs(kv[1]))[:top_n],
        "top_drives": sorted(_drives(p).items(), key=lambda kv: -kv[1])[:top_n],
        "esteem": round(float((p.get("self") or {}).get("esteem", 0.0)), 3),
        "self_image": {d: round(float(((p.get("self") or {}).get("image") or {}).get(d, 0.0)), 3)
                       for d in BASIS},
        "behavior": _behavior(p),
        "beliefs": beliefs,
        "experience_count": p.get("experience_count"),
    }
