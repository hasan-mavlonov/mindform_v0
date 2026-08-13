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


def person(name="Tester", turn=None):
    p = {"identity": {"name": name}, "traits": {d: 0.0 for d in BASIS}}
    if turn is not None:
        p["experience_count"] = turn
    return p


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

# --- recall: emotional memory -- a vivid memory can outrank a flatter, closer one ------
VIVID = "Vivid"
M.create_memory("I aced the presentation everyone was watching.", [1.0, 0.0, 0.0, 0.0],
                {"intensity": 0.0}, {}, person(), name=VIVID)          # closest, flat
M.create_memory("I choked in front of everyone and it still haunts me.", [0.9, 0.436, 0.0, 0.0],
                {"intensity": 1.0}, {}, person(), name=VIVID)          # a bit further, searing
M.create_memory("I mentioned it in passing once.", [0.3, 1.0, 0.0, 0.0],
                {"intensity": 1.0}, {}, person(), name=VIVID)          # vivid but below the floor

hits = M.recall([1.0, 0.0, 0.0, 0.0], name=VIVID, k=3, min_score=0.5)
check("intensity cannot surface a memory the cosine floor rejects",
      all("mentioned it in passing" not in h["text"] for h in hits))
check("a vivid, slightly-less-similar memory outranks a flat, closer one",
      hits[0]["text"].startswith("I choked"))
check("the displayed score stays the honest cosine, not the intensity-boosted one",
      abs(hits[0]["score"] - 0.9) < 1e-3)

# --- recall: emotional memory, part two -- retrieval DECAY (never deletion) ------------
DECAY = "Decay"
M.create_memory("an old, unremarkable Tuesday.", [1.0, 0.0, 0.0, 0.0],
                {"intensity": 0.0}, {}, person(turn=1), name=DECAY)     # stored turn 1
M.create_memory("a fresh, equally unremarkable day.", [1.0, 0.0, 0.0, 0.0],
                {"intensity": 0.0}, {}, person(turn=19), name=DECAY)    # stored turn 19, same cosine

undecayed = M.recall([1.0, 0.0, 0.0, 0.0], name=DECAY, k=2)      # current_turn omitted -> no decay
check("without current_turn, decay is off -- equally-similar memories score identically",
      len(undecayed) == 2 and abs(undecayed[0]["score"] - undecayed[1]["score"]) < 1e-9)

decayed = M.recall([1.0, 0.0, 0.0, 0.0], name=DECAY, k=2, current_turn=20)
check("at equal similarity, a fresh memory outranks a stale one once decay is asked for",
      decayed[0]["text"].startswith("a fresh"))
check("decay never touches the displayed score -- still the honest, undecayed cosine",
      abs(decayed[0]["score"] - 1.0) < 1e-9 and abs(decayed[1]["score"] - 1.0) < 1e-9)

# _decay_factor in isolation: the bounded arithmetic, independent of recall's ranking
old_flat = {"turn": 1, "appraisal": {"intensity": 0.0}}
old_vivid = {"turn": 1, "appraisal": {"intensity": 1.0}}
legacy_no_stamp = {"appraisal": {"intensity": 0.0}}                # predates the "turn" field
check("no current_turn -> no decay (factor 1.0)", M._decay_factor(old_flat, None) == 1.0)
check("no turn stamp on the record -> no decay (factor 1.0)",
      M._decay_factor(legacy_no_stamp, 1000) == 1.0)
check("a fresh memory (age 0) has no decay yet",
      M._decay_factor({"turn": 20, "appraisal": {}}, 20) == 1.0)
check("at the same age, a vivid memory decays less than a flat one (flashbulb protection)",
      M._decay_factor(old_vivid, 101) > M._decay_factor(old_flat, 101))
check("decay is bounded -- never below the floor, even at extreme age",
      M._decay_factor(old_flat, 10 ** 6) >= 0.35 - 1e-9)

# --- create_memory: the log and its embedding row cannot be torn apart by concurrency --
import threading
CONC, N = "Concurrent", 24
vectors = [[1.0 if j == i else 0.0 for j in range(N)] for i in range(N)]
threads = [threading.Thread(target=M.create_memory,
                            args=(f"memory {i}", vectors[i], {}, {}, person()),
                            kwargs={"name": CONC}) for i in range(N)]
for t in threads:
    t.start()
for t in threads:
    t.join()
conc_log = json.load(open(M._memory_path(CONC)))
conc_emb = M.load_embeddings(CONC)
check("concurrent writes: log and sidecar stay the same length",
      len(conc_log) == N and conc_emb.shape[0] == N)
by_text = {m["text"]: i for i, m in enumerate(conc_log)}
check("concurrent writes: every sidecar row still matches its own log record",
      all(by_text[f"memory {i}"] < conc_emb.shape[0]
          and list(conc_emb[by_text[f"memory {i}"]]) == vectors[i] for i in range(N)))

import shutil
shutil.rmtree(tmp, ignore_errors=True)

passed = sum(1 for ok in results if ok)
print(f"\n{passed}/{len(results)} checks passed.")
if passed != len(results):
    raise SystemExit(1)
print("ALL CHECKS PASSED -- full text log + compact embedding sidecar, nothing lost.")
