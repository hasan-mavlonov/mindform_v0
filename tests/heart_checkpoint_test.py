import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Checkpoint and resume mechanics for the HEART-Bench formation runner.

Run with: python tests/heart_checkpoint_test.py   (no LLM calls, no network -- every
check here is about what is written to disk and what is refused, not about forming
a character).

These cover the two failures that lost nine hours of formation: a run that could
not say how far it had got, and a checkpoint that could be resumed into a
character built by different rules.

Needs a HEART-Bench checkout; point HEART_BENCH_PATH at it.
"""

import json
from bench.heart import checkpoint, formation, heartdata

fails = []
def check(label, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + (f"   {detail}" if detail and not cond else ""))
    if not cond: fails.append(label)

from bench.heart.config import require_heart_bench
require_heart_bench()

char = heartdata.load_characters()["CHAR_01"]
mems = heartdata.ingestible_memories(char)

print("\n1. ordering hash is stable and order-sensitive")
h1 = checkpoint.order_hash(mems)
check("same list -> same hash", h1 == checkpoint.order_hash(mems))
swapped = list(mems); swapped[3], swapped[4] = swapped[4], swapped[3]
check("swapping two memories changes the hash", h1 != checkpoint.order_hash(swapped))
check("a prefix does not collide with the full list",
      h1 != checkpoint.order_hash(mems[:10]))

print("\n2. formation fingerprint reacts to formation code")
f1 = checkpoint.formation_config_hash()
check("stable across calls", f1 == checkpoint.formation_config_hash())
src = open("nodes/moral.py", "rb").read()
try:
    open("nodes/moral.py", "ab").write(b"\n# touched by the checkpoint test\n")
    check("changing a formation node changes it", f1 != checkpoint.formation_config_hash())
finally:
    open("nodes/moral.py", "wb").write(src)
check("restored after the test", f1 == checkpoint.formation_config_hash())

print("\n3. a tampered checkpoint is refused, not silently resumed")
CID = "TEST_CKPT"
checkpoint.clear(CID)
name = "heartbench CHAR_01"          # borrow real state files to copy
ck = checkpoint.save(CID, name, 7, mems[6], h1)
ok, detail = checkpoint.verify(CID, ck)
check("a freshly written checkpoint verifies", ok, detail)
slot = os.path.join(checkpoint.dir_for(CID), ck["slot"])
victim = os.path.join(slot, os.listdir(slot)[0])
with open(victim, "ab") as fh: fh.write(b"x")
ok, detail = checkpoint.verify(CID, ck)
check("a modified state file fails verification", not ok, detail)
check("describe() reports it as corrupt, not resumable",
      checkpoint.describe(CID, 1000)["state"] == "corrupt")
check("plan() blocks instead of resuming",
      formation.plan(CID, name, mems)["action"] == "blocked")
try:
    checkpoint.restore(CID, name, ck); raised = False
except RuntimeError:
    raised = True
check("restore() raises on a corrupt checkpoint", raised)

print("\n4. an incompatible checkpoint is refused")
checkpoint.clear(CID)
ck = checkpoint.save(CID, name, 7, mems[6], "a-different-ordering-hash")
ok, why = checkpoint.compatible(CID, mems, ck)
check("a changed ordering makes it incompatible", not ok, why)
check("plan() blocks on it", formation.plan(CID, name, mems)["action"] == "blocked")

print("\n5. the pointer only ever names a complete slot")
checkpoint.clear(CID)
for n in range(1, 6):
    ck = checkpoint.save(CID, name, n, mems[n-1], h1)
    ok, detail = checkpoint.verify(CID, ck)
    if not ok: check(f"save {n} verifies", False, detail); break
    live = json.load(open(os.path.join(checkpoint.dir_for(CID), "checkpoint.json")))
    slots = [d for d in os.listdir(checkpoint.dir_for(CID)) if d.startswith("slot-")]
    if live["slot"] not in slots:
        check(f"save {n}: pointer names an existing slot", False, str(slots)); break
    if len(slots) != 1:
        check(f"save {n}: exactly one slot is kept", False, str(slots)); break
else:
    check("five successive saves each verify, one slot kept", True)
check("the count is what was last written", live["completed_memory_count"] == 5)

print("\n6. the formation lock keeps two processes apart")
checkpoint.clear(CID)
lock = formation.FormationLock(CID).acquire()
lock_path = lock.path
held = json.load(open(lock_path))
held["pid"] = 1                      # pid 1 is alive and is not us
json.dump(held, open(lock_path, "w"))
try:
    formation.FormationLock(CID).acquire(); raised = False
except RuntimeError:
    raised = True
check("a live foreign lock is refused", raised)
held["pid"] = 999999                 # a pid that cannot exist
json.dump(held, open(lock_path, "w"))
try:
    formation.FormationLock(CID).acquire(); raised = False
except RuntimeError:
    raised = True
check("a stale lock from a dead process is taken over", not raised)
formation.FormationLock(CID).release()
checkpoint.clear(CID)

print("\n7. adoption only trusts what it can verify")
info = checkpoint.inspect_live_state("heartbench CHAR_01", mems)
if not info["exists"]:
    print("  SKIP  no live CHAR_01 state on this machine")
else:
    n = info["verified_count"]
    check("the live state verifies against the benchmark ordering",
          n > 0 and n == info["logged_memories"], f"{n} of {info['logged_memories']}")
    wrong = checkpoint.inspect_live_state("heartbench CHAR_01",
                                          heartdata.ingestible_memories(
                                              heartdata.load_characters()["CHAR_08"]))
    check("verified against the WRONG character's ordering it adopts nothing",
          wrong["verified_count"] == 0, str(wrong["verified_count"]))

print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
