"""Prove from the logs that resuming a formation neither repeated nor corrupted it.

    python -m bench.heart.verify_resume --character CHAR_01

Reads every run log on disk and checks two things about a character that was
formed across more than one process:

  no repeats     every memory appears exactly once, in chronological order, even
                 though several runs contributed to it;
  no drift       memory N+1's traits_before equals memory N's traits_after, at
                 every step and across every process boundary.

The second is the one that matters. A checkpoint that restored a character even
slightly wrong would still produce a plausible-looking run -- the traits would
simply jump at the boundary. Bit-exact carry-over is what says the resumed
character is the same character.
"""

import argparse
import glob
import json
import os
import sys

from bench.heart.config import RESULTS_ROOT


def load_steps(character_id):
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_ROOT, "*", "events.jsonl"))):
        run = os.path.basename(os.path.dirname(path))
        try:
            fh = open(path, encoding="utf-8")
        except OSError:
            continue
        with fh:
            for line in fh:
                if '"ingest_step"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") != "ingest_step":
                    continue
                if character_id not in str(rec.get("character") or ""):
                    continue
                rows.append((run, rec))
    rows.sort(key=lambda x: x[1]["chrono_position"])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--character", default="CHAR_01")
    ap.add_argument("--quiet", action="store_true", help="verdict only")
    args = ap.parse_args()

    rows = load_steps(args.character)
    if not rows:
        print(f"No ingestion logs found for {args.character}.")
        return 1

    runs = {r for r, _ in rows}
    print(f"{args.character}: {len(rows)} memories ingested across {len(runs)} runs\n")

    seen = {}
    dupes = []
    for run, rec in rows:
        if rec["anon_id"] in seen:
            dupes.append((rec["anon_id"], seen[rec["anon_id"]], run))
        seen[rec["anon_id"]] = run

    drift = []
    if not args.quiet:
        print(f"{'pos':>5}  {'memory':<20}{'run':<34}{'':<12}continuity")
    for i, (run, rec) in enumerate(rows):
        boundary = i > 0 and rows[i - 1][0] != run
        if i == 0:
            note = "first"
        else:
            prev = rows[i - 1][1]
            d = max(abs(rec["traits_before"][k] - prev["traits_after"][k])
                    for k in rec["traits_before"])
            note = "exact" if d == 0.0 else f"DRIFT {d:.4f}"
            if d != 0.0:
                drift.append((rec["anon_id"], d))
        if not args.quiet:
            print(f"{rec['chrono_position']:>5}  {rec['anon_id']:<20}{run:<34}"
                  f"{'<-- RESUMED' if boundary else '':<12}{note}")

    crossings = sum(1 for i in range(1, len(rows)) if rows[i - 1][0] != rows[i][0])
    print()
    ok = True
    if dupes:
        ok = False
        print(f"FAIL: {len(dupes)} memories were ingested more than once:")
        for aid, first, again in dupes[:10]:
            print(f"  {aid}  first in {first}, again in {again}")
    else:
        print(f"PASS: every memory ingested exactly once "
              f"(positions {rows[0][1]['chrono_position']}"
              f"..{rows[-1][1]['chrono_position']}).")
    if drift:
        ok = False
        print(f"FAIL: formation state jumped at {len(drift)} steps — resume is "
              f"not restoring the character faithfully:")
        for aid, d in drift[:10]:
            print(f"  {aid}  max trait drift {d:.4f}")
    else:
        print(f"PASS: traits carry over exactly at every step, across all "
              f"{crossings} process boundaries.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
