"""Form a HEART character from all its memories, with checkpoints and resume.

    python -m bench.heart.prepare --character CHAR_01           # start or resume
    python -m bench.heart.prepare --character CHAR_01 --status  # look, change nothing
    python -m bench.heart.prepare --character CHAR_01 --start-over

The same formation the live UI's Full Protocol Benchmark runs, without needing a
browser window open for nine hours. Progress is checkpointed after every memory,
so this can be stopped with Ctrl+C and restarted later -- by this command or by
the UI -- and it picks up at the next memory.
"""

import argparse
import sys
import time

from bench.heart import checkpoint, formation, heartdata, mfadapter
from bench.heart.config import RESULTS_ROOT, require_heart_bench
from bench.heart.logbook import Logbook, new_run_id


def _fmt(s):
    if s is None:
        return "—"
    s = int(s)
    if s < 90:
        return f"{s}s"
    h, m = divmod(s // 60, 60)
    return f"{h}h {m}m" if h else f"{m}m"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--character", default="CHAR_01")
    ap.add_argument("--status", action="store_true", help="report and exit")
    ap.add_argument("--start-over", action="store_true",
                    help="DISCARD the checkpoint and form from memory 1")
    ap.add_argument("--limit", type=int,
                    help="stop after this many total memories (for testing resume)")
    args = ap.parse_args()
    require_heart_bench()

    cid = args.character
    char = heartdata.load_characters()[cid]
    # NOTE: --limit never truncates this list. The checkpoint's ordering hash
    # covers the whole chronological set, so a run stopped early stays compatible
    # with the full run that continues it -- truncating here would make the two
    # look like different characters.
    memories = heartdata.ingestible_memories(char)
    name = f"heartbench {cid}"
    total = len(memories)

    decision = formation.plan(cid, name, memories)
    print(f"\n{cid}: {total} memories, {decision['action'].upper()} — {decision['reason']}")
    if decision["action"] in ("resume", "adopt", "complete"):
        print(f"Found checkpoint for {cid}")
        print(f"Completed: {decision['done']} / {total} memories")
        ck = decision.get("checkpoint") or {}
        if ck.get("last_completed_memory_id"):
            print(f"Last completed memory: {ck['last_completed_memory_id']} "
                  f"(saved {ck.get('updated_at')})")
        if decision["action"] != "complete":
            print(f"Resuming from memory {decision['done'] + 1}")
    if decision["action"] == "blocked":
        print(f"\nThis checkpoint cannot be resumed: {decision['reason']}")
        print(f"Re-run with --start-over to discard {decision['done']} formed memories.")
        return 2
    if args.status:
        return 0
    if args.start_over and decision.get("done"):
        print(f"\nAbout to DELETE {decision['done']} formed memories for {cid}.")
        if input("Type the character id to confirm: ").strip() != cid:
            print("Not confirmed; nothing changed.")
            return 1

    run_id = new_run_id(f"prepare-{cid.lower()}")
    log = Logbook(run_id, RESULTS_ROOT)
    monitor = formation.Monitor(cid, total, done_at_start=decision.get("done", 0),
                                echo=True)
    stop = {"flag": False}

    def cancel():
        # --limit exercises the real cancellation path rather than a shorter
        # benchmark: the run is asked to stop the same way the Stop button asks.
        if args.limit and monitor.done >= args.limit:
            return True
        return stop["flag"]

    import signal

    def on_sigint(signum, frame):
        if stop["flag"]:
            print("\nSecond interrupt — exiting now.")
            sys.exit(130)
        stop["flag"] = True
        print("\n\nInterrupt received.\n\nSaving checkpoint… "
              "(finishing the current model call first)")

    signal.signal(signal.SIGINT, on_sigint)

    t0 = time.time()
    try:
        with formation.FormationLock(cid):
            formation.form(cid, char, memories, name, log, monitor, cancel=cancel,
                           decision=decision, fresh=args.start_over)
    except formation.FormationPaused as exc:
        print(f"\n{exc}")
        print(f"Checkpoint: {checkpoint.dir_for(cid)}")
        print(f"Fix the problem, then re-run this command to resume from memory "
              f"{monitor.done + 1}.")
        return 3
    except (mfadapter.Cancelled, KeyboardInterrupt):
        done = monitor.done
        print(f"\n{cid}: {done} / {total} memories safely completed.")
        print(f"Next run will resume from memory {done + 1}.")
        print(f"Checkpoint: {checkpoint.dir_for(cid)}")
        return 130

    sid, _ = mfadapter.freeze(name, log.dir, "frozen")
    print(f"\nFormed {cid} from all {total} memories in {_fmt(time.time() - t0)}.")
    print(f"Snapshot {sid[:16]}… frozen in {log.dir}")
    print("The live UI will now offer Quick Test and Character Test against the "
          "full snapshot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
