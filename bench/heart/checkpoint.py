"""Crash-safe, resumable checkpoints for full-protocol character formation.

Forming a HEART character from all 1,000 episodic memories is a ~9 hour job with
five LLM calls per memory. Before this module existed there was no way to stop
it: MindForm's own ``save_character`` wrote the character after every memory, but
nothing recorded HOW FAR the run had got, and ``mfadapter.create_neutral``
deleted the state files at the start of every attempt. A run interrupted at
memory 252 therefore looked, on the next click, exactly like a run that had never
started -- and starting it again destroyed the 252 memories of work.

What this module adds:

  * a deterministic checkpoint identity per character, independent of any browser
    session, run id or timestamp -- ``<results>/checkpoints/<CHAR_ID>/``;
  * a record, written after EVERY completed memory, of how many memories are done
    and which one was last;
  * crash-safe writes: state files are copied into an unused slot, and only once
    they are fully on disk is the pointer file swapped in with ``os.replace``, so
    a crash can never leave a checkpoint that claims progress it does not have;
  * a compatibility fingerprint over the formation code and model, so a checkpoint
    is never silently resumed into a character formed by different rules;
  * adoption of an "orphaned" live character state -- one left behind on disk by
    an interrupted pre-checkpoint run -- verified memory by memory against the
    benchmark's own chronological ordering before it is trusted.

Nothing here touches core/, nodes/ or web/. It only copies the state files that
``mfadapter.state_files`` already defines.
"""

import datetime
import glob
import hashlib
import json
import os
import shutil

from bench.heart.config import RESULTS_ROOT

# Bumped whenever the meaning of a checkpoint changes in a way that makes older
# ones unsafe to resume.
CHECKPOINT_VERSION = 1
PROTOCOL = "heart-bench-full-v1"

# Formation is decided by these files plus the model. If any of them changes, a
# half-formed character was built under different rules and resuming it would
# silently mix two formation behaviours in one character -- so the fingerprint
# covers the code, not just the config.
_FORMATION_SOURCES = [
    "core/personality.py", "core/llm.py", "core/config.py",
    "nodes/llm_appraisal.py", "nodes/llm_impact.py", "nodes/values.py",
    "nodes/moral.py", "nodes/beliefs.py", "nodes/drives.py",
    "nodes/self_concept.py", "nodes/behavior.py", "nodes/expression.py",
    "nodes/temperament.py", "nodes/cognition.py", "nodes/character.py",
    "web/engine_bridge.py", "bench/heart/mfadapter.py",
]

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------
def root():
    return os.path.join(RESULTS_ROOT, "checkpoints")


def dir_for(character_id):
    """The one and only checkpoint directory for a character.

    Deterministic on purpose: no run id, no timestamp, no session. Whatever
    process starts the formation, and however it died, the next process looks
    here and finds it.
    """
    return os.path.join(root(), character_id)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def formation_config_hash():
    """Fingerprint of everything that decides how a character forms."""
    from core.config import LLM_MODEL, BASIS
    h = hashlib.sha256()
    h.update(f"{PROTOCOL}|{CHECKPOINT_VERSION}|{LLM_MODEL}|{','.join(BASIS)}|".encode())
    for rel in _FORMATION_SOURCES:
        path = os.path.join(_REPO_ROOT, rel)
        digest = _sha256_file(path) if os.path.exists(path) else "absent"
        h.update(f"{rel}:{digest}|".encode())
    return h.hexdigest()


def order_hash(memories):
    """Fingerprint of the exact chronological memory ordering.

    Resume is only safe if memory 253 on the second run is the same memory that
    would have been 253 on the first. This pins the whole ordering, so any change
    to HEART's data or to the sort is caught before a single memory is ingested.
    """
    h = hashlib.sha256()
    for m in memories:
        h.update(f"{m['chrono_position']}:{m['anon_id']}:{m['text_sha256']}|".encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def read(character_id):
    """The checkpoint for a character, or None. Never raises on a bad file."""
    path = os.path.join(dir_for(character_id), "checkpoint.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def verify(character_id, ck=None):
    """Check a checkpoint's state files are all present and unmodified.

    Returns (ok, detail). The pointer file is written last and only after the
    slot is complete, so a mismatch here means the files were touched after the
    fact, not that a crash landed mid-write.
    """
    ck = ck or read(character_id)
    if not ck:
        return False, "no checkpoint"
    slot_dir = os.path.join(dir_for(character_id), ck.get("slot") or "")
    if not os.path.isdir(slot_dir):
        return False, f"slot directory missing: {ck.get('slot')}"
    for fname, want in (ck.get("files") or {}).items():
        path = os.path.join(slot_dir, fname)
        if want == "absent":
            if os.path.exists(path):
                return False, f"{fname} should be absent but exists"
            continue
        if not os.path.exists(path):
            return False, f"{fname} missing from slot"
        if _sha256_file(path) != want:
            return False, f"{fname} does not match its recorded hash"
    return True, "ok"


def compatible(character_id, memories, ck=None):
    """Whether a checkpoint may be resumed under the current code and data.

    Returns (ok, reason). A refusal here is never overridden silently -- the
    caller must either start over or be told to pass an explicit override.
    """
    ck = ck or read(character_id)
    if not ck:
        return False, "no checkpoint"
    if ck.get("checkpoint_version") != CHECKPOINT_VERSION:
        return False, (f"checkpoint format v{ck.get('checkpoint_version')} predates "
                       f"this build (v{CHECKPOINT_VERSION})")
    if ck.get("order_hash") != order_hash(memories):
        return False, ("the chronological memory ordering has changed since this "
                       "checkpoint was written")
    if ck.get("formation_config_hash") != formation_config_hash():
        return False, ("the formation code or model has changed since this "
                       "checkpoint was written")
    return True, "ok"


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def _fsync_dir(path):
    """Persist a directory entry itself, so a rename survives a power cut."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _copy_fsync(src, dst):
    shutil.copyfile(src, dst)
    with open(dst, "rb+") as fh:
        os.fsync(fh.fileno())


def save(character_id, bench_name, done_count, last_memory, order_h, stats=None):
    """Record that ``done_count`` memories are completely ingested.

    The write is ordered so that it is safe to be killed at any instant:

        1. copy the live state files into the slot the pointer is NOT using
        2. fsync every copied file and the slot directory
        3. write the new pointer to a temp file and fsync it
        4. os.replace the temp file over checkpoint.json  <- atomic
        5. fsync the checkpoint directory

    Until step 4 the old checkpoint is still the live one and still valid; after
    step 4 the new slot is already complete on disk. There is no instant at which
    checkpoint.json names a slot that is not fully written.
    """
    from bench.heart import mfadapter

    ck_dir = dir_for(character_id)
    os.makedirs(ck_dir, exist_ok=True)
    prev = read(character_id)
    slot = "slot-b" if (prev or {}).get("slot") == "slot-a" else "slot-a"
    slot_dir = os.path.join(ck_dir, slot)

    if os.path.isdir(slot_dir):
        shutil.rmtree(slot_dir)
    os.makedirs(slot_dir)

    files = {}
    for path in mfadapter.state_files(bench_name):
        fname = os.path.basename(path)
        if os.path.exists(path):
            _copy_fsync(path, os.path.join(slot_dir, fname))
            files[fname] = _sha256_file(os.path.join(slot_dir, fname))
        else:
            files[fname] = "absent"
    _fsync_dir(slot_dir)

    from core.config import LLM_MODEL
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "protocol": PROTOCOL,
        "character_id": character_id,
        "bench_name": bench_name,
        "model": LLM_MODEL,
        "formation_config_hash": formation_config_hash(),
        "order_hash": order_h,
        "completed_memory_count": done_count,
        "last_completed_index": (last_memory or {}).get("original_index"),
        "last_completed_memory_id": (last_memory or {}).get("anon_id"),
        "last_completed_chrono_position": (last_memory or {}).get("chrono_position"),
        "last_completed_timeline": (last_memory or {}).get("timeline"),
        "slot": slot,
        "files": files,
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "stats": stats or {},
    }

    tmp = os.path.join(ck_dir, "checkpoint.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, os.path.join(ck_dir, "checkpoint.json"))
    _fsync_dir(ck_dir)

    # The slot the pointer no longer names is now dead weight; drop it so a
    # checkpoint directory never grows past two copies of the state.
    old = (prev or {}).get("slot")
    if old and old != slot:
        shutil.rmtree(os.path.join(ck_dir, old), ignore_errors=True)
    return payload


def restore(character_id, bench_name, ck=None):
    """Put a checkpoint's state files back as the live character.

    Verified first: a checkpoint that does not match its own hashes is never
    restored, because a half-copied character would form the rest of its
    memories on top of corrupt state and look perfectly normal doing it.
    """
    from bench.heart import mfadapter

    ck = ck or read(character_id)
    ok, detail = verify(character_id, ck)
    if not ok:
        raise RuntimeError(f"checkpoint for {character_id} is unusable: {detail}")
    slot_dir = os.path.join(dir_for(character_id), ck["slot"])
    for path in mfadapter.state_files(bench_name):
        fname = os.path.basename(path)
        src = os.path.join(slot_dir, fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(src):
            _copy_fsync(src, path)
        elif os.path.exists(path):
            os.remove(path)
    return ck


def clear(character_id):
    """Delete a character's checkpoint entirely. Only ever called explicitly."""
    shutil.rmtree(dir_for(character_id), ignore_errors=True)


# ---------------------------------------------------------------------------
# recovery of pre-checkpoint runs
# ---------------------------------------------------------------------------
def inspect_live_state(bench_name, memories):
    """What, if anything, a left-behind live character state is worth.

    An interrupted run from before checkpointing existed leaves a perfectly good
    character in data/characters/ with no record of its progress. Its memory log
    holds the raw text of everything it ingested, so the claim "N memories done"
    can be checked rather than trusted: memory k in the log must be memory k of
    the benchmark's chronological ordering, for every k.

    Returns a dict describing what was found, including ``verified_count`` --
    the number of leading memories that provably match.
    """
    from bench.heart import mfadapter
    from core.personality import character_path

    base = character_path(bench_name)
    out = {"exists": os.path.exists(base), "path": base, "experience_count": None,
           "logged_memories": 0, "verified_count": 0, "mismatch_at": None,
           "files": {}}
    if not out["exists"]:
        return out

    for path in mfadapter.state_files(bench_name):
        out["files"][os.path.basename(path)] = (
            os.path.getsize(path) if os.path.exists(path) else None)

    try:
        with open(base, encoding="utf-8") as fh:
            personality = json.load(fh)
        out["experience_count"] = personality.get("experience_count")
    except (json.JSONDecodeError, OSError):
        return out

    mem_path = base[: -len(".json")] + ".memories.json"
    try:
        with open(mem_path, encoding="utf-8") as fh:
            logged = json.load(fh)
    except (json.JSONDecodeError, OSError):
        logged = []
    if not isinstance(logged, list):
        logged = []
    out["logged_memories"] = len(logged)

    # Match the memory log against the benchmark ordering, one for one.
    for i, rec in enumerate(logged):
        if i >= len(memories):
            out["mismatch_at"] = i
            break
        text = (rec or {}).get("text") if isinstance(rec, dict) else None
        want = hashlib.sha256((memories[i]["text"] or "").encode("utf-8")).hexdigest()
        got = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
        if got != want:
            out["mismatch_at"] = i
            break
        out["verified_count"] = i + 1
    return out


def adopt_live_state(character_id, bench_name, memories):
    """Turn a verified left-behind character state into a real checkpoint.

    Only the provably-matching leading run of memories is adopted. If the log
    diverges from the benchmark ordering at memory k, the checkpoint is written
    at k, not at whatever the character claimed -- the rest is discarded rather
    than trusted.
    """
    info = inspect_live_state(bench_name, memories)
    n = info["verified_count"]
    if not n:
        return None, info
    ck = save(character_id, bench_name, n, memories[n - 1], order_hash(memories),
              stats={"adopted_from_live_state": True,
                     "claimed_experience_count": info["experience_count"],
                     "logged_memories": info["logged_memories"]})
    return ck, info


# ---------------------------------------------------------------------------
# status, for the UI
# ---------------------------------------------------------------------------
def describe(character_id, total_memories, memories=None):
    """One dict saying where a character stands: the UI's single source of truth."""
    ck = read(character_id)
    if not ck:
        return {"state": "none", "done": 0, "total": total_memories,
                "resumable": False, "resume_from": 1}
    ok, detail = verify(character_id, ck)
    done = int(ck.get("completed_memory_count") or 0)
    out = {
        "state": "complete" if done >= total_memories else "partial",
        "done": done,
        "total": total_memories,
        "resume_from": done + 1,
        "resumable": ok and done < total_memories,
        "verified": ok,
        "detail": detail,
        "updated_at": ck.get("updated_at"),
        "last_memory_id": ck.get("last_completed_memory_id"),
        "adopted": bool((ck.get("stats") or {}).get("adopted_from_live_state")),
    }
    if not ok:
        out["state"] = "corrupt"
        out["resumable"] = False
    elif memories is not None:
        compat_ok, reason = compatible(character_id, memories, ck)
        out["compatible"] = compat_ok
        if not compat_ok:
            out["state"] = "stale"
            out["resumable"] = False
            out["detail"] = reason
    return out


def all_status():
    """Every character that has a checkpoint on disk, for CLI reporting."""
    out = {}
    for path in glob.glob(os.path.join(root(), "*", "checkpoint.json")):
        cid = os.path.basename(os.path.dirname(path))
        ck = read(cid)
        if ck:
            out[cid] = ck
    return out
