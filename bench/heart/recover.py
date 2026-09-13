"""Find and rescue formation progress left behind by an interrupted run.

    python -m bench.heart.recover --character CHAR_01          # look, change nothing
    python -m bench.heart.recover --character CHAR_01 --adopt  # turn it into a checkpoint

Before checkpointing existed, an interrupted full-protocol run left a perfectly
good half-formed character in data/characters/ and no record at all of how far it
had got -- and the next run deleted it. This tool reads everything that could
carry that progress and says plainly what survived.

It never deletes and never overwrites the live character state. ``--adopt`` only
writes a new checkpoint, and only for the leading run of memories it can prove
match the benchmark's chronological ordering one for one.
"""

import argparse
import glob
import json
import os

from bench.heart import checkpoint, heartdata, snapshots
from bench.heart.config import RESULTS_ROOT, require_heart_bench

W = 78


def _bench_name(character_id):
    return f"heartbench {character_id}"


def survey(character_id):
    """Everything on disk that bears on this character's formation progress."""
    char = heartdata.load_characters()[character_id]
    memories = heartdata.ingestible_memories(char)
    name = _bench_name(character_id)

    out = {
        "character": character_id,
        "bench_name": name,
        "total_memories": len(memories),
        "checkpoint": checkpoint.read(character_id),
        "live": checkpoint.inspect_live_state(name, memories),
        "snapshots": snapshots.discover().get(character_id) or [],
        "runs": [],
    }
    if out["checkpoint"]:
        ok, detail = checkpoint.verify(character_id, out["checkpoint"])
        out["checkpoint_ok"], out["checkpoint_detail"] = ok, detail

    # Every run log that ingested memories for this character, and how far it got.
    for events_path in sorted(glob.glob(os.path.join(RESULTS_ROOT, "*", "events.jsonl"))):
        run_id = os.path.basename(os.path.dirname(events_path))
        steps, last = 0, None
        try:
            with open(events_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or '"ingest_step"' not in line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("event") != "ingest_step":
                        continue
                    if character_id not in str(rec.get("character") or ""):
                        continue
                    steps += 1
                    last = rec
        except OSError:
            continue
        if steps:
            out["runs"].append({
                "run_id": run_id, "ingest_steps": steps,
                "last_experience_count": (last or {}).get("experience_count"),
                "last_anon_id": (last or {}).get("anon_id"),
                "last_ts": (last or {}).get("ts"),
            })
    out["runs"].sort(key=lambda r: r["ingest_steps"], reverse=True)
    return out


def render(s):
    L = []
    a = L.append
    a("=" * W)
    a(f"FORMATION RECOVERY — {s['character']}  ({s['total_memories']} memories total)")
    a("=" * W)

    a("")
    a("CHECKPOINT")
    ck = s["checkpoint"]
    if not ck:
        a("  none — no checkpoint directory for this character")
    else:
        a(f"  completed        {ck.get('completed_memory_count')}/{s['total_memories']}")
        a(f"  last memory      {ck.get('last_completed_memory_id')}")
        a(f"  written          {ck.get('updated_at')}")
        a(f"  integrity        {'OK' if s.get('checkpoint_ok') else 'FAILED — ' + str(s.get('checkpoint_detail'))}")
        a(f"  location         {checkpoint.dir_for(s['character'])}")

    a("")
    a("LIVE CHARACTER STATE  (data/characters/)")
    live = s["live"]
    if not live["exists"]:
        a("  none — the character file does not exist")
        a("  Nothing was left behind. If a run was interrupted, its state was")
        a("  deleted by a later run that started over from memory 1.")
    else:
        a(f"  path             {live['path']}")
        a(f"  experience_count {live['experience_count']}  (what the character claims)")
        a(f"  memories logged  {live['logged_memories']}")
        a(f"  VERIFIED         {live['verified_count']} memories match the benchmark "
          f"ordering exactly")
        if live["mismatch_at"] is not None:
            a(f"  diverges at      memory {live['mismatch_at'] + 1} — everything from "
              f"there on is discarded")
        for fname, size in (live["files"] or {}).items():
            a(f"    {fname:<44} {('%d bytes' % size) if size else 'absent'}")

    a("")
    a("FROZEN SNAPSHOTS")
    if not s["snapshots"]:
        a("  none")
    for snap in s["snapshots"]:
        a(f"  {snap['memories']:>5} memories  run={snap['run_id']:<28} "
          f"{'FULL' if snap['full_protocol'] else 'dev'}")

    a("")
    a("RUN LOGS THAT INGESTED THIS CHARACTER")
    if not s["runs"]:
        a("  none")
    for r in s["runs"][:10]:
        a(f"  {r['run_id']:<40} {r['ingest_steps']:>5} memories  "
          f"last={r['last_anon_id']}  {r['last_ts']}")

    a("")
    a("-" * W)
    a("VERDICT")
    a("-" * W)
    best = max((s["live"]["verified_count"],
                int((ck or {}).get("completed_memory_count") or 0)))
    if ck and int(ck.get("completed_memory_count") or 0) >= best and best:
        a(f"  A checkpoint already holds {best} formed memories. Resume will")
        a(f"  continue from memory {best + 1}. Nothing to recover.")
    elif s["live"]["verified_count"]:
        a(f"  {s['live']['verified_count']} formed memories are recoverable from the live")
        a("  character state. Run with --adopt to turn them into a checkpoint;")
        a(f"  the next run then resumes from memory {s['live']['verified_count'] + 1}.")
    elif s["runs"]:
        deepest = s["runs"][0]
        a(f"  UNRECOVERABLE. A run log shows {deepest['ingest_steps']} memories were")
        a(f"  ingested (run {deepest['run_id']}), but the character state those")
        a("  memories produced is gone from disk.")
        a("")
        a("  The logs record each step's traits, values and drives, but not the")
        a("  memory store, the embeddings or the formed beliefs — so the character")
        a("  itself cannot be rebuilt from them, only described. Re-forming those")
        a("  memories is the only way back, and this time it will checkpoint.")
    else:
        a("  Nothing found. This character has not been formed on this machine.")
    a("=" * W)
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--character", default="CHAR_01")
    ap.add_argument("--adopt", action="store_true",
                    help="write a checkpoint from the verified live state")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    require_heart_bench()

    s = survey(args.character)
    if args.json:
        print(json.dumps(s, indent=2, ensure_ascii=False, default=str))
        return
    print(render(s))

    if args.adopt:
        char = heartdata.load_characters()[args.character]
        memories = heartdata.ingestible_memories(char)
        name = _bench_name(args.character)
        existing = int((checkpoint.read(args.character) or {}).get(
            "completed_memory_count") or 0)
        if s["live"]["verified_count"] <= existing:
            print(f"\nNothing to adopt: the checkpoint already covers "
                  f"{existing} memories.")
            return
        ck, info = checkpoint.adopt_live_state(args.character, name, memories)
        if not ck:
            print("\nNothing to adopt: no verifiable memories in the live state.")
            return
        print(f"\nAdopted {ck['completed_memory_count']} memories into a checkpoint at")
        print(f"  {checkpoint.dir_for(args.character)}")
        print(f"The next full-protocol run resumes from memory "
              f"{ck['completed_memory_count'] + 1}.")


if __name__ == "__main__":
    main()
