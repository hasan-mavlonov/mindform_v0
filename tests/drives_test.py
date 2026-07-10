"""MOTIVATION & DRIVES: the SDT need-homeostat seeded by Schwartz values.

Dependency-free (pure arithmetic -- no torch / numpy / network), like the acceptance and
cognition suites. Proves the three claims of the slice:

  * SEED    -- the formed Schwartz values decide which of the three SDT needs matter.
  * HOMEOSTAT -- tension relaxes toward its resting weight, falls when an event feeds the
                 need and rises when it is denied, and stays bounded.
  * WANTING -- an ACTIVE (loud) need bends the appraisal: the same event reads as more
               goal-relevant and more positive when the character is starved for it, and a
               frustrating event reads as more threatening -- and a louder need bends harder.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.llm as llm_mod
llm_mod.LLM_API_KEY = ""           # never touch the network

from core.config import DRIVES, DRIVE_WEIGHT_FLOOR
import nodes.drives as drives
from nodes.cognition import interpret, read_lens
from core.personality import default_personality, migrate

_checks = 0
_failed = 0


def check(label, cond):
    global _checks, _failed
    _checks += 1
    if not cond:
        _failed += 1
    print(("PASS" if cond else "FAIL") + " -- " + label)


def person(tensions=None, values=None):
    """A minimal personality: neutral traits, given values, given drive tensions."""
    tensions = tensions or {}
    return {
        "traits": {"O": 0.0, "C": 0.0, "E": 0.0, "A": 0.0, "N": 0.0},
        "character": {"values": values or {}},
        "drives": {d: {"tension": tensions.get(d, 0.0)} for d in DRIVES},
    }


# --- SEED: values decide which needs matter ------------------------------------
w_blank = drives.seed_weights({})
check("blank values -> every need rests at the floor",
      all(abs(w_blank[d] - DRIVE_WEIGHT_FLOOR) < 1e-9 for d in DRIVES))
check("all seed weights lie in [0, 1]", all(0.0 <= w_blank[d] <= 1.0 for d in DRIVES))

w_sd = drives.seed_weights({"SD": 1.0})
check("self-direction seeds autonomy above the other needs",
      w_sd["autonomy"] > w_sd["competence"] and w_sd["autonomy"] > w_sd["relatedness"])
w_ac = drives.seed_weights({"AC": 1.0})
check("achievement seeds competence highest",
      w_ac["competence"] > w_ac["autonomy"] and w_ac["competence"] > w_ac["relatedness"])
w_be = drives.seed_weights({"BE": 1.0})
check("benevolence seeds relatedness highest",
      w_be["relatedness"] > w_be["autonomy"] and w_be["relatedness"] > w_be["competence"])
w_max = drives.seed_weights({v: 1.0 for v in
                             ["SD", "ST", "HE", "AC", "PO", "SE", "CO", "TR", "BE", "UN"]})
check("weights stay clamped in [0, 1] even with every value pinned at 1",
      all(0.0 <= w_max[d] <= 1.0 for d in DRIVES))


# --- HOMEOSTAT: tension rises toward weight, falls when fed ---------------------
lo = drives.refresh(person({"competence": 0.0}, {"AC": 1.0}))["drives"]["competence"]["tension"]
hi = drives.refresh(person({"competence": 1.0}, {"AC": 1.0}))["drives"]["competence"]["tension"]
check("an unmet need's tension rises toward its (high) resting weight", 0.0 < lo < 0.78)
check("an over-relieved need's tension falls back toward its weight", 0.78 < hi < 1.0)

good = {"outcome": 0.8, "agency": 0.4}          # feeds competence
bad = {"outcome": -0.8, "agency": -0.4}         # frustrates competence
fed = drives.apply_event(person({"competence": 0.6})["drives"], good)["competence"]["tension"]
denied = drives.apply_event(person({"competence": 0.6})["drives"], bad)["competence"]["tension"]
check("a satisfying event lowers the need's tension (relief)", fed < 0.6)
check("a frustrating event raises the need's tension", denied > 0.6)

capped = drives.apply_event(person({"competence": 0.95})["drives"], good)["competence"]["tension"]
check("tension stays within [0, 1] under a strong event", 0.0 <= capped <= 1.0)

seq, d = [], person({"competence": 0.9})["drives"]
for _ in range(5):
    d = drives.apply_event(d, good)
    seq.append(d["competence"]["tension"])
check("repeated satisfaction drives the need toward sated (monotone down)",
      all(seq[i] < seq[i - 1] for i in range(1, len(seq))) and seq[-1] < 0.5)

# satisfaction is temporary: after being fed, regeneration brings the need back
after = drives.apply_event(person({"competence": 0.9})["drives"], good)
back = drives.refresh({"character": {"values": {"AC": 1.0}}, "drives": after})
check("after relief, regeneration brings the need back (satisfaction is temporary)",
      back["drives"]["competence"]["tension"] > after["competence"]["tension"])


# --- WANTING: an active need bends the appraisal -------------------------------
mastery = {"valence": 0.5, "intensity": 0.7, "novelty": 0.3, "agency": 0.6, "social": 0.0,
           "outcome": 0.85, "self_relevance": 0.5, "threat_challenge": 0.4}

starved = interpret(mastery, person({"autonomy": 0.3, "competence": 0.9, "relatedness": 0.3}))
sated = interpret(mastery, person({"autonomy": 0.3, "competence": 0.1, "relatedness": 0.3}))
check("a starved need makes the same event read as more self-relevant than when sated",
      starved["self_relevance"] > sated["self_relevance"])
check("a starved need makes a fulfilling event read more positively (valence)",
      starved["valence"] > sated["valence"])
check("a goal-relevant event raises self_relevance above the raw reading",
      starved["self_relevance"] > mastery["self_relevance"])

quiet = interpret(mastery, person())     # all tensions 0 -> no active need
check("no active need -> the drive lens leaves the appraisal unchanged",
      quiet["self_relevance"] == mastery["self_relevance"]
      and quiet["valence"] == mastery["valence"]
      and quiet["threat_challenge"] == mastery["threat_challenge"])

# a LOUDER need bends harder (the 'starved hits harder' dynamic): single active need
loud = interpret(mastery, person({"competence": 0.9}))
faint = interpret(mastery, person({"competence": 0.2}))
check("a louder need bends self_relevance harder than a faint one",
      loud["self_relevance"] - mastery["self_relevance"]
      > faint["self_relevance"] - mastery["self_relevance"] > 0.0)

# frustrating an ACTIVE need adds threat (threat_challenge more negative)
frustrating = {"valence": -0.3, "intensity": 0.6, "novelty": 0.2, "agency": -0.7, "social": 0.0,
               "outcome": -0.4, "self_relevance": 0.5, "threat_challenge": -0.1}
fr_loud = interpret(frustrating, person({"autonomy": 0.9}))
fr_quiet = interpret(frustrating, person({"autonomy": 0.1}))
check("frustrating a loud need adds more threat than frustrating a quiet one",
      fr_loud["threat_challenge"] < fr_quiet["threat_challenge"])

# the loud need surfaces in the UI/LLM lens; a quiet one does not
check("a loud need surfaces as a 'needs ...' tag in the lens",
      "needs" in read_lens(person({"competence": 0.9})).lower())
check("no loud need -> no drive tag in the lens",
      read_lens(person({"competence": 0.2})) == "")


# --- Degradation & persistence -------------------------------------------------
pre_drive = {"traits": {"O": 0.0, "C": 0.0, "E": 0.0, "A": 0.0, "N": 0.0}}   # no 'drives' key
try:
    _ = interpret(mastery, pre_drive)
    ok = True
except Exception:
    ok = False
check("interpret() on a personality with no drive state does not crash (offline-safe)", ok)

blank = default_personality()
check("a fresh character is born with all three drives at rest",
      set(blank["drives"]) == set(DRIVES)
      and all(abs(blank["drives"][d]["tension"] - DRIVE_WEIGHT_FLOOR) < 1e-9 for d in DRIVES))

legacy = migrate({"traits": {"O": 0, "C": 0, "E": 0, "A": 0, "N": 0},
                  "character": {"values": {"AC": 1.0}}})
check("migrate backfills drives onto a pre-drives save, at rest = weight",
      "drives" in legacy and legacy["drives"]["competence"]["tension"] > legacy["drives"]["autonomy"]["tension"])


# --- Slice 2: frustration bites harder than satisfaction soothes (SDT asymmetry) ---
sat_event = {"outcome": 0.6, "agency": 0.3}      # feeds competence
frust_event = {"outcome": -0.6, "agency": -0.3}  # frustrates it, equal magnitude
relief = 0.5 - drives.apply_event(person({"competence": 0.5})["drives"], sat_event)["competence"]["tension"]
harm = drives.apply_event(person({"competence": 0.5})["drives"], frust_event)["competence"]["tension"] - 0.5
check("an equal-magnitude frustration moves tension harder than a satisfaction (asymmetry)",
      harm > relief > 0)


# --- Slice 2: motivated retrieval -- what they lack shapes what they remember -----
def mem(text, score, **appraisal):
    base = {"valence": 0.0, "intensity": 0.5, "novelty": 0.3, "agency": 0.0,
            "social": 0.0, "outcome": 0.0, "self_relevance": 0.5, "threat_challenge": 0.0}
    return {"text": text, "score": score, "appraisal": {**base, **appraisal}}

# a warm-social memory and a mastery memory; the mastery one is (slightly) more similar
social_mem = mem("laughing with friends", 0.40, social=0.9, valence=0.6)
mastery_mem = mem("finally winning the contest", 0.45, outcome=0.9, agency=0.6)
neutral_mem = mem("a walk in the rain", 0.44)

lonely = person({"relatedness": 0.9, "competence": 0.1, "autonomy": 0.1})["drives"]
striving = person({"relatedness": 0.1, "competence": 0.9, "autonomy": 0.1})["drives"]

lonely_top = drives.recall_bias([mastery_mem, social_mem, neutral_mem], lonely)[0]
striving_top = drives.recall_bias([mastery_mem, social_mem, neutral_mem], striving)[0]
check("a connection-starved character surfaces the belonging memory first",
      lonely_top["text"] == "laughing with friends")
check("an achievement-starved character surfaces the mastery memory first",
      striving_top["text"] == "finally winning the contest")
check("the pulling need is named on the surfaced memory",
      lonely_top.get("need") == "relatedness" and striving_top.get("need") == "competence")
check("the displayed score stays the honest cosine (selection is biased, familiarity is not)",
      lonely_top["score"] == 0.40 and striving_top["score"] == 0.45)

quiet = person()["drives"]                       # all tensions 0 -> no motive, no re-rank
unbiased = drives.recall_bias([mastery_mem, social_mem, neutral_mem], quiet)
check("with no active need the order is untouched (pure pass-through)",
      [m["text"] for m in unbiased] == ["finally winning the contest",
                                        "laughing with friends", "a walk in the rain"])
check("the pool is cut to k", len(drives.recall_bias([social_mem] * 8, lonely, k=3)) == 3)
check("recall_bias survives empty and missing-appraisal input",
      drives.recall_bias([], lonely) == []
      and drives.recall_bias([{"text": "x", "score": 0.3}], lonely)[0]["text"] == "x")
# the need chooses among related memories; it cannot make a weak match beat a strong one
far = mem("laughing with friends", 0.30, social=0.9, valence=0.6)
near = mem("the contract deadline", 0.80)
check("semantic relevance stays primary (a weak match cannot leapfrog a strong one)",
      drives.recall_bias([near, far], lonely)[0]["text"] == "the contract deadline")


print()
print("%d/%d checks passed." % (_checks - _failed, _checks))
if _failed:
    print("SOME CHECKS FAILED.")
    sys.exit(1)
print("ALL CHECKS PASSED -- values seed the needs; the homeostat wants; perception bends to it.")
