"""Checkpointed, cancellable, watchable character formation.

One place that knows how to form a HEART character from its memories and survive
being interrupted. Both front ends use it -- the live UI's worker thread and the
headless CLI -- so "resume" means exactly the same thing in both, and a run
started in the browser can be finished from the terminal or the other way round.

Three things it guarantees:

  * a memory that has been committed is never ingested twice (the checkpoint is
    written after the engine saves, so the count on disk is always true);
  * a run is never silently restarted from zero (the caller must ask for a fresh
    start explicitly, and a fresh start deletes the old checkpoint on purpose);
  * two processes never form the same character at once (a lock file, checked
    against the running pid).
"""

import collections
import datetime
import json
import os
import time

from bench.heart import checkpoint, mfadapter

# ETA from one or two samples is noise, and noise printed as "52,439 minutes
# remaining" is worse than printing nothing. Nothing is shown until there are
# enough completed memories to average, and the average only ever looks at the
# recent past so a rate change shows up quickly.
ETA_MIN_SAMPLES = 5
ETA_WINDOW = 20
FEED_MAX = 300

# Every formation node falls back to a deterministic heuristic when its LLM call
# fails, so an outage does not stop a run -- it quietly changes what the run is
# measuring, and nine hours later the character was formed by fallback code. A
# short streak of failing memories is treated as an outage and halts the run with
# its checkpoint intact, rather than forming 700 memories nobody would trust.
ERROR_STREAK_LIMIT = 3


class FormationPaused(Exception):
    """Formation stopped itself because something upstream is broken."""


# ---------------------------------------------------------------------------
# deciding what to do
# ---------------------------------------------------------------------------
def plan(character_id, bench_name, memories, allow_adopt=True):
    """Decide, without changing anything, how a formation run should proceed.

    Returns a dict with ``action`` one of:
      complete  -- the checkpoint already covers every memory
      resume    -- a valid checkpoint exists; carry on from ``done``
      adopt     -- no checkpoint, but a verifiable half-formed character is on
                   disk from a pre-checkpoint run and can be turned into one
      fresh     -- nothing usable; form from memory 1
    plus ``blocked`` when a checkpoint exists but must not be used silently.
    """
    total = len(memories)
    ck = checkpoint.read(character_id)
    if ck:
        ok, detail = checkpoint.verify(character_id, ck)
        compat_ok, reason = checkpoint.compatible(character_id, memories, ck)
        done = int(ck.get("completed_memory_count") or 0)
        if ok and compat_ok and done >= total:
            return {"action": "complete", "done": done, "total": total,
                    "checkpoint": ck, "reason": "all memories already formed"}
        if ok and compat_ok:
            return {"action": "resume", "done": done, "total": total,
                    "checkpoint": ck,
                    "reason": f"checkpoint at {done}/{total}"}
        return {"action": "blocked", "done": done, "total": total, "checkpoint": ck,
                "reason": detail if not ok else reason,
                "blocked_kind": "corrupt" if not ok else "stale"}

    if allow_adopt:
        info = checkpoint.inspect_live_state(bench_name, memories)
        if info["verified_count"] > 0:
            return {"action": "adopt", "done": info["verified_count"], "total": total,
                    "live": info,
                    "reason": (f"{info['verified_count']} memories already on disk from "
                               f"an earlier run, verified against the benchmark ordering")}
    return {"action": "fresh", "done": 0, "total": total,
            "reason": "no checkpoint and no recoverable state"}


# ---------------------------------------------------------------------------
# one-process-at-a-time
# ---------------------------------------------------------------------------
def _lock_path(character_id):
    return os.path.join(checkpoint.dir_for(character_id), "forming.lock")


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError, TypeError):
        return False
    return True


class FormationLock:
    """Refuse to form one character in two processes at once.

    Two writers would interleave their memories into one character file and the
    checkpoint count would describe neither of them. A stale lock left by a
    killed process is detected by its pid and taken over.
    """

    def __init__(self, character_id):
        self.character_id = character_id
        self.path = _lock_path(character_id)

    def acquire(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as fh:
                    held = json.load(fh)
            except (json.JSONDecodeError, OSError):
                held = {}
            pid = held.get("pid")
            if pid and pid != os.getpid() and _pid_alive(pid):
                raise RuntimeError(
                    f"{self.character_id} is already being formed by process {pid} "
                    f"(started {held.get('started_at')}). Stop that run first.")
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"pid": os.getpid(),
                       "started_at": datetime.datetime.now().isoformat(timespec="seconds")},
                      fh)
        return self

    def release(self):
        try:
            os.remove(self.path)
        except OSError:
            pass

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False


# ---------------------------------------------------------------------------
# live progress
# ---------------------------------------------------------------------------
class Monitor:
    """Everything a watcher needs to know about a formation in flight.

    Deliberately passive: it is told what happened and computes rates. The UI
    polls ``payload()``; the CLI prints from the same callbacks.
    """

    def __init__(self, character_id, total, done_at_start=0, echo=False):
        self.character_id = character_id
        self.total = total
        self.done = done_at_start
        self.started_done = done_at_start
        self.started_at = time.time()
        self.current_index = None          # 0-based index of the memory in flight
        self.current_memory = None
        self.stage = None
        self.stage_started_at = None
        self.stages_done = []              # [(label, seconds)] for the memory in flight
        self.memory_started_at = None
        self.durations = collections.deque(maxlen=ETA_WINDOW)
        self.llm_calls = 0
        self.retries = 0
        self.errors = 0
        self.last_checkpoint = None
        self.feed = collections.deque(maxlen=FEED_MAX)
        self.echo = echo
        self.paused_reason = None

    # -- events ------------------------------------------------------------
    def note(self, text, icon="·"):
        entry = {"t": datetime.datetime.now().strftime("%H:%M:%S"),
                 "icon": icon, "text": text}
        self.feed.append(entry)
        if self.echo:
            print(f"{entry['t']}  {icon} {text}", flush=True)
        return entry

    def memory_started(self, i, total, mem):
        self.current_index = i
        self.current_memory = {"anon_id": mem["anon_id"], "timeline": mem["timeline"],
                               "chrono_position": mem["chrono_position"]}
        self.memory_started_at = time.time()
        self.stages_done = []
        self.stage = None
        self.stage_started_at = None
        self.note(f"memory {i + 1}/{total} started — {mem['anon_id']}", "→")

    def on_stage(self, kind, label, seconds, error):
        if kind == "stage_start":
            self.stage = label
            self.stage_started_at = time.time()
            self.llm_calls += 1
        elif kind == "stage_done":
            self.stages_done.append({"label": label, "seconds": round(seconds, 1)})
            self.stage = None
            self.note(f"{label}  {seconds:.1f}s", "✓")
        elif kind == "stage_failed":
            self.retries += 1
            self.stage = None
            self.stages_done.append({"label": label, "seconds": round(seconds or 0, 1),
                                     "error": error})
            self.note(f"{label} failed after {seconds or 0:.1f}s — {error} "
                      f"(node falls back to its heuristic)", "!")

    def memory_committed(self, done, mem, rec):
        took = time.time() - (self.memory_started_at or time.time())
        self.durations.append(took)
        self.done = done
        if rec and rec.get("error"):
            self.errors += 1
        self.note(f"memory {done}/{self.total} complete — {took:.1f}s total", "✓")

    def checkpointed(self, done):
        self.last_checkpoint = {"done": done,
                                "at": datetime.datetime.now().strftime("%H:%M:%S")}
        self.note(f"checkpoint saved — {done}/{self.total}", "✓")

    def paused(self, reason):
        self.paused_reason = reason
        self.note(reason, "✗")

    # -- derived -----------------------------------------------------------
    def rate(self):
        """Seconds per memory over the recent window, or None if too early."""
        if len(self.durations) < ETA_MIN_SAMPLES:
            return None
        return sum(self.durations) / len(self.durations)

    def rate_last10(self):
        if len(self.durations) < ETA_MIN_SAMPLES:
            return None
        recent = list(self.durations)[-10:]
        return sum(recent) / len(recent)

    def eta_s(self):
        r = self.rate_last10() or self.rate()
        if r is None:
            return None
        return max(0, (self.total - self.done)) * r

    def payload(self):
        return {
            "character": self.character_id,
            "done": self.done, "total": self.total,
            "percent": round(100.0 * self.done / self.total, 1) if self.total else 0.0,
            "resumed_from": self.started_done,
            "current_memory": self.current_memory,
            "stage": self.stage,
            "stage_elapsed_s": (round(time.time() - self.stage_started_at, 1)
                                if self.stage_started_at and self.stage else None),
            "stages_done": list(self.stages_done),
            "memory_elapsed_s": (round(time.time() - self.memory_started_at, 1)
                                 if self.memory_started_at else None),
            "elapsed_s": round(time.time() - self.started_at, 1),
            "avg_s": round(self.rate(), 1) if self.rate() else None,
            "avg_last10_s": round(self.rate_last10(), 1) if self.rate_last10() else None,
            "eta_s": round(self.eta_s()) if self.eta_s() is not None else None,
            "llm_calls": self.llm_calls,
            "retries": self.retries,
            "errors": self.errors,
            "last_checkpoint": self.last_checkpoint,
            "paused_reason": self.paused_reason,
            "feed": list(self.feed)[-60:],
        }


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
def form(character_id, char, memories, bench_name, logbook, monitor,
         cancel=None, decision=None, fresh=False):
    """Form a character to completion, checkpointing after every memory.

    ``fresh=True`` is the destructive path: it deletes any existing checkpoint
    and starts from memory 1. It is never chosen here -- the caller has to ask
    for it, because on a 1,000-memory character it throws away hours of work.
    """
    total = len(memories)
    order_h = checkpoint.order_hash(memories)
    decision = decision or plan(character_id, bench_name, memories)

    if fresh:
        checkpoint.clear(character_id)
        decision = {"action": "fresh", "done": 0, "total": total,
                    "reason": "explicit start-over requested"}

    if decision["action"] == "blocked":
        raise RuntimeError(
            f"{character_id} has a checkpoint that cannot be resumed: "
            f"{decision['reason']}. Start over (which deletes {decision['done']} "
            f"formed memories) or restore a compatible build.")

    start_at = 0
    if decision["action"] == "complete":
        monitor.note(f"checkpoint already covers all {total} memories", "✓")
        checkpoint.restore(character_id, bench_name, decision["checkpoint"])
        return total, decision

    if decision["action"] == "resume":
        checkpoint.restore(character_id, bench_name, decision["checkpoint"])
        start_at = decision["done"]
        # Seed the panel's "last checkpoint" from the one being resumed, so it
        # shows real progress immediately rather than a dash until the first
        # memory of this session happens to finish.
        monitor.last_checkpoint = {
            "done": start_at,
            "at": (decision["checkpoint"].get("updated_at") or "").split("T")[-1]}
        monitor.note(f"found checkpoint for {character_id} — {start_at}/{total} "
                     f"complete, resuming from memory {start_at + 1}", "✓")
        logbook.event("formation_resumed", character=character_id, already=start_at,
                      remaining=total - start_at,
                      last_memory=decision["checkpoint"].get("last_completed_memory_id"))
    elif decision["action"] == "adopt":
        ck, info = checkpoint.adopt_live_state(character_id, bench_name, memories)
        if not ck:
            decision = {"action": "fresh", "done": 0, "total": total,
                        "reason": "nothing on disk could be verified"}
        else:
            start_at = info["verified_count"]
            monitor.note(f"adopted {start_at} memories left behind by an earlier run "
                         f"(verified against the benchmark ordering), resuming from "
                         f"memory {start_at + 1}", "✓")
            logbook.event("formation_adopted", character=character_id,
                          adopted=start_at, claimed=info.get("experience_count"))

    if decision["action"] == "fresh":
        monitor.note(f"forming {character_id} from memory 1 of {total}", "→")
        mfadapter.create_neutral(bench_name, char.get("occupation", "N/A"))
        logbook.event("formation_fresh", character=character_id, memories=total)

    monitor.done = start_at
    monitor.started_done = start_at
    mfadapter.register_id_map(memories)

    streak = {"n": 0}

    def on_commit(done, mem, rec):
        # Checkpoint FIRST, so whatever is decided below, the work is already safe.
        checkpoint.save(character_id, bench_name, done, mem, order_h,
                        stats={"llm_calls": monitor.llm_calls,
                               "retries": monitor.retries,
                               "errors": monitor.errors})
        monitor.memory_committed(done, mem, rec)
        monitor.checkpointed(done)
        streak["n"] = streak["n"] + 1 if (rec or {}).get("error") else 0
        if streak["n"] >= ERROR_STREAK_LIMIT:
            reason = (f"ERROR: {streak['n']} memories in a row failed to form "
                      f"({(rec or {}).get('error')}). The model or network looks "
                      f"unavailable — formation paused, checkpoint preserved at "
                      f"{done}/{total}.")
            monitor.paused(reason)
            logbook.event("formation_paused", character=character_id, completed=done,
                          reason=reason)
            raise FormationPaused(reason)

    mfadapter.ingest_all(bench_name, memories, logbook,
                         start_at=start_at,
                         on_commit=on_commit,
                         cancel=cancel,
                         on_stage=monitor.on_stage,
                         on_memory_start=monitor.memory_started)
    return total, decision
