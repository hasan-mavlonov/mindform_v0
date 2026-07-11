"""PERCEPTION distillation: the LLM appraises online; the offline head learns from it.

Dependency-free (no torch / numpy / network). Proves the three claims of the slice:

  * FALLBACK  -- with no key, appraise_from_text returns the offline reading, labelled so.
  * VALIDATION -- LLM replies are range-clamped per dim and rejected when malformed.
  * THE IRON RULE -- only LLM-labelled rows enter the distillation log (never the head's
                     or the lexicon's own outputs), in the exact JSONL shape the trainer
                     reads, and logging failures never break a turn.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.llm as llm_mod
llm_mod.LLM_API_KEY = ""           # never touch the network

from core.config import APPRAISAL_DIMS, APPRAISAL_SCHEMA, LLM_LABEL
from core.appraisal import appraise
import nodes.llm_appraisal as la
from core.appraisal_log import log_appraisal, maybe_log

_checks = 0
_failed = 0


def check(label, cond):
    global _checks, _failed
    _checks += 1
    if not cond:
        _failed += 1
    print(("PASS" if cond else "FAIL") + " -- " + label)


TEXT = "I failed the exam I studied months for."

# --- fallback: perception never hard-depends on the network -----------------------
reading, source = la.appraise_from_text(TEXT)
check("with no key, appraise_from_text falls back to the offline appraiser",
      source == "offline" and reading == appraise(TEXT))
check("the fallback reading carries every appraisal dim",
      all(k in reading for k in APPRAISAL_DIMS))

# --- validation: the LLM reply is clamped per dim, rejected when malformed --------
check("signed dims clamp to [-1, 1] and unit dims to [0, 1]",
      la._clamp_dim("valence", -3.0) == -1.0
      and la._clamp_dim("intensity", -0.5) == 0.0
      and la._clamp_dim("novelty", 7.0) == 1.0
      and la._clamp_dim("threat_challenge", 0.4) == 0.4)


def fake_llm(data):
    """Route _llm_appraise through a canned reply instead of the network."""
    original = la.complete_json
    la.complete_json = lambda *a, **kw: data
    try:
        return la._llm_appraise(TEXT)
    finally:
        la.complete_json = original


full = {k: 0.5 for k in APPRAISAL_DIMS}
check("a well-formed LLM reply passes through with all dims",
      fake_llm(dict(full)) == {k: 0.5 for k in APPRAISAL_DIMS})
wild = dict(full, valence=-9.0, intensity=4.0)
clamped = fake_llm(wild)
check("out-of-range LLM values are clamped, not trusted",
      clamped["valence"] == -1.0 and clamped["intensity"] == 1.0)
try:
    fake_llm({"valence": 0.2})                     # 7 dims missing
    ok = False
except Exception:
    ok = True
check("a reply missing dims is rejected (falls back rather than half-reads)", ok)
try:
    fake_llm(dict(full, agency="strong"))          # non-numeric
    ok = False
except Exception:
    ok = True
check("a non-numeric dim is rejected", ok)

# --- the distillation log ----------------------------------------------------------
tmp = os.path.join(tempfile.mkdtemp(), "rows.jsonl")
check("an LLM-labelled reading is logged", maybe_log(TEXT, full, LLM_LABEL, tmp) is True)
check("an offline reading is NEVER logged (the iron rule)",
      maybe_log(TEXT, reading, "offline", tmp) is False)
check("a lexicon/head label is never logged either",
      maybe_log(TEXT, reading, "heuristic", tmp) is False)
rows = [json.loads(line) for line in open(tmp) if line.strip()]
check("exactly the teacher rows landed, in the trainer's exact shape",
      len(rows) == 1 and rows[0]["text"] == TEXT
      and set(rows[0]["appraisal"]) == set(APPRAISAL_DIMS))
check("appending accumulates rows",
      log_appraisal("Another day at the lake.", full, tmp) is True
      and sum(1 for _ in open(tmp)) == 2)
check("blank text or empty appraisal is refused",
      log_appraisal("   ", full, tmp) is False and log_appraisal(TEXT, {}, tmp) is False)
check("an unwritable path fails soft (a turn never breaks over logging)",
      log_appraisal(TEXT, full, "/proc/nope/rows.jsonl") is False)

# --- schema hygiene ------------------------------------------------------------------
check("the prompt promises exactly the engine's appraisal dims",
      all(k in la.APPRAISAL_SYSTEM_PROMPT for k in APPRAISAL_DIMS)
      and all(k in APPRAISAL_SCHEMA for k in APPRAISAL_DIMS))


print()
print("%d/%d checks passed." % (_checks - _failed, _checks))
if _failed:
    print("SOME CHECKS FAILED.")
    sys.exit(1)
print("ALL CHECKS PASSED -- the online model teaches; the offline appraiser learns with use.")
