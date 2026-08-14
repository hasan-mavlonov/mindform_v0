import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Belief-memory storage test: SEMANTIC memory made recallable.

Mirrors tests/memory_test.py's shape and discipline exactly (same class of desync risk,
same fix). Requires numpy (the embedding sidecar) and skips cleanly when it is absent, so
it is safe to run in the dependency-free sandbox. Run: python tests/belief_memory_test.py
"""

import importlib.util
import tempfile

if importlib.util.find_spec("numpy") is None:
    print("belief_memory_test SKIPPED -- numpy not installed")
    raise SystemExit(0)

import core.belief_memory as BM

results = []


def check(name, ok):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


def character(beliefs):
    return {"beliefs": [{"statement": s, "confidence": c, "count": 1} for s, c in beliefs]}


tmp = tempfile.mkdtemp()
# core.belief_memory imports CHARACTERS_DIR by VALUE (`from core.personality import
# CHARACTERS_DIR`), binding it into its OWN module namespace at import time -- so it must
# be redirected on THIS module (BM.CHARACTERS_DIR), not on core.personality, exactly like
# tests/memory_test.py redirects core.memory's own binding, not core.personality's.
BM.CHARACTERS_DIR = tmp

NAME = "Aisha"
check("empty store -> recall is []",
      BM.recall_beliefs([1.0, 0.0, 0.0, 0.0], character([]), name=NAME) == [])

BM.append_belief_embeddings([[1.0, 0.0, 0.0, 0.0]], name=NAME)
check("sidecar exists after one append", BM.load_belief_embeddings(NAME).shape == (1, 4))
check("appending an empty list is a no-op", (BM.append_belief_embeddings([], name=NAME),
      BM.load_belief_embeddings(NAME).shape)[1] == (1, 4))

BM.append_belief_embeddings([[0.0, 1.0, 0.0, 0.0]], name=NAME)
check("sidecar grows by exactly one row per genuinely-new belief",
      BM.load_belief_embeddings(NAME).shape == (2, 4))

ch = character([("hard work pays off", 0.6), ("people can be trusted", 0.3)])
hits = BM.recall_beliefs([1.0, 0.0, 0.0, 0.0], ch, name=NAME, k=2)
check("recall_beliefs returns index-aligned records (the first belief, first embedding)",
      len(hits) == 1 and hits[0]["statement"] == "hard work pays off")
check("recall_beliefs attaches the honest cosine score", abs(hits[0]["score"] - 1.0) < 1e-9)

check("recall_beliefs drops irrelevant beliefs (below min_score)",
      BM.recall_beliefs([0.0, 0.0, 1.0, 0.0], ch, name=NAME, min_score=0.5) == [])
check("recall_beliefs on a character with no belief history is []",
      BM.recall_beliefs([1.0, 0.0, 0.0, 0.0], character([]), name="Nobody") == [])

# a belief added to the character but not yet embedded (sidecar hasn't caught up) is
# simply not yet recallable -- never an IndexError
ch_ahead = character([("hard work pays off", 0.6), ("people can be trusted", 0.3),
                      ("a third belief with no embedding yet", 0.1)])
check("a belief ahead of the sidecar is safely excluded, not an error",
      all(h["statement"] != "a third belief with no embedding yet"
          for h in BM.recall_beliefs([1.0, 0.0, 0.0, 0.0], ch_ahead, name=NAME, k=5)))

# --- conviction re-ranks ties: a firmly-held belief outranks a shaky one at equal cosine -
CONV = "Conviction"
BM.append_belief_embeddings([[1.0, 0.0, 0.0, 0.0], [0.9, 0.436, 0.0, 0.0]], name=CONV)
ch_conv = character([("shaky belief, same relevance", 0.05), ("firm belief, close relevance", 0.9)])
hits = BM.recall_beliefs([1.0, 0.0, 0.0, 0.0], ch_conv, name=CONV, k=2, min_score=0.5)
check("a firmly-held, slightly-less-similar belief can outrank a shaky, closer one",
      hits[0]["statement"] == "firm belief, close relevance")
check("conviction reranking never touches the displayed score (still the honest cosine)",
      abs(hits[0]["score"] - 0.9) < 1e-3)

# --- concurrent appends cannot tear the sidecar apart -----------------------------------
import threading
CONC, N = "Concurrent", 16
vectors = [[1.0 if j == i else 0.0 for j in range(N)] for i in range(N)]
threads = [threading.Thread(target=BM.append_belief_embeddings, args=([v],), kwargs={"name": CONC})
          for v in vectors]
for t in threads:
    t.start()
for t in threads:
    t.join()
final = BM.load_belief_embeddings(CONC)
check("concurrent belief-embedding appends: the sidecar ends up with exactly N rows",
      final.shape == (N, N))
rows = {tuple(row) for row in final}
check("concurrent belief-embedding appends: every row survived intact (none lost or corrupted)",
      rows == {tuple(v) for v in vectors})

import shutil
shutil.rmtree(tmp, ignore_errors=True)

passed = sum(1 for ok in results if ok)
print(f"\n{passed}/{len(results)} checks passed.")
if passed != len(results):
    raise SystemExit(1)
print("ALL CHECKS PASSED -- beliefs are formed AND recalled: semantic memory, made real.")
