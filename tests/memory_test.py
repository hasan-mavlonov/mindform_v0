import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Memory storage test (Slice 1): a complete text log + a compact .npy embedding sidecar.

Requires numpy (the embedding sidecar) and skips cleanly when it is absent, so it is safe
to run in the dependency-free sandbox. Run: python tests/memory_test.py
"""

import importlib.util
import json
import tempfile

if importlib.util.find_spec("numpy") is None:
    print("memory_test SKIPPED -- numpy not installed")
    raise SystemExit(0)

import core.memory as M
from core.config import BASIS

results = []


def check(name, ok):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


def person(name="Tester"):
    return {"identity": {"name": name}, "traits": {d: 0.0 for d in BASIS}}


tmp = tempfile.mkdtemp()
M.CHARACTERS_DIR = tmp                                   # redirect storage into a temp dir
M.DEFAULT_MEMORY_FILE = os.path.join(tmp, "memories.json")

NAME = "Aisha"
e1 = [1.0, 0.0, 0.0, 0.0]
e2 = [1.0, 0.0, 0.0, 0.0]      # identical to e1 -> should recur
e3 = [0.0, 1.0, 0.0, 0.0]      # orthogonal -> should not

check("empty store -> recurrence 0", M.recurrence(e1, name=NAME) == 0)

M.create_memory("I went to a party.", e1, {"valence": 1.0}, {"E": 0.3}, person(), name=NAME)

log = json.load(open(M._memory_path(NAME)))
check("log keeps the text record", len(log) == 1 and log[0]["text"] == "I went to a party.")
check("embedding is NOT in the JSON log", "embedding" not in log[0])
check("record keeps appraisal / push / traits_after",
      all(k in log[0] for k in ("appraisal", "push", "traits_after")))
check("embedding sidecar exists", os.path.exists(M._embeddings_path(NAME)))
check("sidecar is 1 x D", M.load_embeddings(NAME).shape == (1, len(e1)))

check("identical experience recurs (>=1)", M.recurrence(e2, name=NAME) >= 1)
check("orthogonal experience does not recur", M.recurrence(e3, name=NAME) == 0)

M.create_memory("Another party.", e2, {}, {}, person(), name=NAME)
check("log + sidecar stay index-aligned (2 each)",
      len(json.load(open(M._memory_path(NAME)))) == 2 and M.load_embeddings(NAME).shape[0] == 2)

# --- legacy migration: an old log with inline embeddings and no sidecar -------
LEG = "Marcus"
legacy = [
    {"text": "x", "embedding": [1.0, 0.0, 0.0, 0.0], "appraisal": {}, "push": {}, "traits_after": {}},
    {"text": "y", "embedding": [0.0, 1.0, 0.0, 0.0], "appraisal": {}, "push": {}, "traits_after": {}},
]
json.dump(legacy, open(M._memory_path(LEG), "w"))
check("legacy migrated -> sidecar built (2 x D)", M.load_embeddings(LEG).shape == (2, 4))
migrated = json.load(open(M._memory_path(LEG)))
check("legacy log slimmed (embedding removed, text kept)",
      all("embedding" not in m for m in migrated) and migrated[0]["text"] == "x")
check("recurrence works after migration", M.recurrence([1.0, 0.0, 0.0, 0.0], name=LEG) >= 1)

# --- recall: top-k most relevant past memories (Slice 2) ----------------------
RC = "Sage"
M.create_memory("I love hiking in the mountains.", [1.0, 0.0, 0.0, 0.0], {}, {}, person(), name=RC)
M.create_memory("My boss criticized my report.",    [0.0, 1.0, 0.0, 0.0], {}, {}, person(), name=RC)
M.create_memory("I went hiking again this weekend.", [0.95, 0.05, 0.0, 0.0], {}, {}, person(), name=RC)

hits = M.recall([1.0, 0.0, 0.0, 0.0], name=RC, k=2)
check("recall returns at most k", len(hits) == 2)
check("recall surfaces the relevant memories (hiking, not the boss)",
      all("hiking" in h["text"] for h in hits))
check("recall attaches a relevance score", "score" in hits[0] and hits[0]["score"] > 0.9)
check("recall drops irrelevant memories (below min_score)",
      M.recall([0.0, 0.0, 1.0, 0.0], name=RC, k=3) == [])
check("recall on empty history is []", M.recall([1.0, 0.0, 0.0, 0.0], name="Nobody") == [])

import shutil
shutil.rmtree(tmp, ignore_errors=True)

passed = sum(1 for ok in results if ok)
print(f"\n{passed}/{len(results)} checks passed.")
if passed != len(results):
    raise SystemExit(1)
print("ALL CHECKS PASSED -- full text log + compact embedding sidecar, nothing lost.")
