"""Which HEART characters are prepared, and how deeply.

A character is "prepared" when a frozen MindForm snapshot exists for it. The
depth of that preparation is what separates a development test from a
protocol-faithful one: HEART gives each character 1,000 episodic memories, and
a snapshot built from 50 of them is a different character than one built from
all 1,000. The UI must never blur that, so every snapshot reports its depth and
whether it is full.
"""

import glob
import json
import os

from bench.heart.config import RESULTS_ROOT
from bench.heart import heartdata

FULL_DEPTH = 1000


def _character_of(snap_dir):
    """Find the personality file in a snapshot dir and read who it is.

    A snapshot holds several .json files (personality, memory log, manifest);
    only the personality one is a dict carrying "identity".
    """
    for path in sorted(glob.glob(os.path.join(snap_dir, "*.json"))):
        if path.endswith("manifest.json") or path.endswith(".memories.json"):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                p = json.load(fh)
        except Exception:
            continue
        if not isinstance(p, dict) or "identity" not in p:
            continue
        name = ((p.get("identity") or {}).get("name") or "")
        cid = next((tok for tok in name.split() if tok.startswith("CHAR_")), None)
        if cid:
            return cid, int(p.get("experience_count") or 0), name
    return None, None, None


def discover():
    """Every frozen snapshot on disk, newest-deepest first per character."""
    found = {}
    pattern = os.path.join(RESULTS_ROOT, "*", "snapshots", "*", "manifest.json")
    for manifest_path in glob.glob(pattern):
        snap_dir = os.path.dirname(manifest_path)
        cid, depth, bench_name = _character_of(snap_dir)
        if not cid:
            continue
        run_id = os.path.relpath(snap_dir, RESULTS_ROOT).split(os.sep)[0]
        try:
            with open(manifest_path, encoding="utf-8") as fh:
                manifest = json.load(fh)
        except Exception:
            manifest = {}
        entry = {
            "character": cid,
            "memories": depth,
            "full_protocol": depth >= FULL_DEPTH,
            "snapshot_id": manifest.get("snapshot_id"),
            "run_id": run_id,
            "label": os.path.basename(snap_dir),
            "run_dir": os.path.dirname(os.path.dirname(snap_dir)),
            "bench_name": bench_name,
            "mtime": os.path.getmtime(manifest_path),
        }
        found.setdefault(cid, []).append(entry)

    for cid in found:
        found[cid].sort(key=lambda e: (e["memories"], e["mtime"]), reverse=True)
    return found


def best_for(character_id):
    """The deepest prepared snapshot for a character, or None."""
    return (discover().get(character_id) or [None])[0]


def best_for_tier(character_id, tier):
    """The deepest snapshot in one tier ("dev" = <1000 memories, "full" = >=1000).

    A character can have snapshots in both tiers at once (a 50-memory dev run
    and, later, a full 1000-memory one) -- they are never the same snapshot, so
    picking one must never silently fall back to the other.
    """
    snaps = discover().get(character_id) or []
    if tier == "full":
        pool = [s for s in snaps if s["full_protocol"]]
    else:
        pool = [s for s in snaps if not s["full_protocol"]]
    return pool[0] if pool else None  # discover() already sorts deepest-first


def catalogue():
    """Every HEART character with its preparation status, for the UI.

    Reports the dev and full tiers separately -- a 50-memory dev snapshot and
    a "not yet prepared" full one are never collapsed into a single "prepared"
    flag, so the UI can never present a partial-memory run as the full one.
    """
    prepared = discover()
    out = []
    for cid, char in sorted(heartdata.load_characters().items()):
        snaps = prepared.get(cid) or []
        dev = next((s for s in snaps if not s["full_protocol"]), None)
        full = next((s for s in snaps if s["full_protocol"]), None)
        best = full or dev
        out.append({
            "character": cid,
            "occupation": char.get("occupation"),
            "questions": len(heartdata.questions_for(cid)),
            "total_memories": len(char.get("episodic_memory_set") or []),
            "dev_memories": dev["memories"] if dev else 0,
            "dev_prepared": bool(dev),
            "full_prepared": bool(full),
            # kept for older callers: "the best available snapshot, whichever tier"
            "prepared": bool(best),
            "prepared_memories": best["memories"] if best else 0,
            "full_protocol": bool(full),
            "snapshot_id": best["snapshot_id"] if best else None,
            "snapshot_run": best["run_id"] if best else None,
        })
    return out
