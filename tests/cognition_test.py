import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Cognitive Patterns test: the interpretation lens bends perception by current traits
(Slice 1) and by what the character remembers (Slice 2).

Dependency-free (pure functions, no torch / numpy / network). Recalled episodes are
fabricated directly -- the same slim record shape ``memory.recall`` returns.
Run: python tests/cognition_test.py
"""

from nodes.cognition import interpret, lens, read_lens

results = []


def check(name, ok):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


def person(**traits):
    base = {"O": 0.0, "C": 0.0, "E": 0.0, "A": 0.0, "N": 0.0}
    base.update(traits)
    return {"traits": base}


raw = {"valence": 0.2, "threat_challenge": 0.0, "novelty": 0.3, "social": 1.0, "intensity": 0.5}

# a near-neutral character perceives the event as-is
neutral = interpret(raw, person())
check("neutral character leaves the appraisal unchanged",
      neutral["valence"] == raw["valence"]
      and neutral["threat_challenge"] == raw["threat_challenge"]
      and neutral["novelty"] == raw["novelty"])

# neuroticism: more perceived threat, darker valence. threat_challenge is signed
# -1 = threat/loss, +1 = challenge/growth, so "more threat" = MORE NEGATIVE.
anxious = interpret(raw, person(N=0.8))
check("anxious reads more threat than neutral", anxious["threat_challenge"] < neutral["threat_challenge"])
check("anxious reads a darker valence than neutral", anxious["valence"] < neutral["valence"])
calm = interpret(raw, person(N=-0.8))
check("calm reads less threat than neutral", calm["threat_challenge"] > neutral["threat_challenge"])

# openness: more perceived novelty
openp = interpret(raw, person(O=0.8))
check("open perceives more novelty than neutral", openp["novelty"] > neutral["novelty"])

# agreeableness: reads a social event warmer
warm = interpret(raw, person(A=0.8))
check("agreeable reads a social event warmer", warm["valence"] > neutral["valence"])

# the whole point: the SAME event lands differently on different people
check("same event -> different interpretation per character",
      anxious["threat_challenge"] != openp["threat_challenge"]
      or anxious["valence"] != warm["valence"])

# stays in range even at the extremes
extreme = interpret({"valence": -0.95, "threat_challenge": 0.95, "novelty": 0.98}, person(N=1.0, O=1.0))
check("interpretation stays clamped in range",
      -1.0 <= extreme["valence"] <= 1.0 and -1.0 <= extreme["threat_challenge"] <= 1.0
      and 0.0 <= extreme["novelty"] <= 1.0)

# lens: empty when neutral, descriptive when strong
check("lens is empty for a near-neutral character", lens(person()) == "" and read_lens(person()) == "")
check("anxious lens mentions threat", "threat" in lens(person(N=0.8)).lower())
check("read_lens gives a short non-empty phrase for a strong character",
      read_lens(person(N=0.8, O=0.8)) != "")


# --- Slice 2: memory bends perception too (a learned expectation / schema) ----
def mem(valence=0.0, threat=0.0, novelty=0.0, score=0.8):
    """A recalled episode in the slim shape memory.recall returns (appraisal + score)."""
    return {"appraisal": {"valence": valence, "threat_challenge": threat, "novelty": novelty},
            "score": score}


raw2 = {"valence": 0.2, "threat_challenge": 0.0, "novelty": 0.3, "social": 0.0, "intensity": 0.5}

# no relevant memory -> identical to the trait-only lens (backward compatible, offline-safe)
check("empty recall == trait-only interpretation",
      interpret(raw2, person(N=0.8), recalled=[]) == interpret(raw2, person(N=0.8)))

base2 = interpret(raw2, person())                       # neutral person, no memory: as-is
bad = [mem(valence=-0.8, score=0.9), mem(valence=-0.7, score=0.85)]
good = [mem(valence=0.8, score=0.9), mem(valence=0.7, score=0.85)]

dark = interpret(raw2, person(), recalled=bad)
bright = interpret(raw2, person(), recalled=good)
check("memory of bad outcomes darkens the reading", dark["valence"] < base2["valence"])
check("memory of good outcomes brightens the reading", bright["valence"] > base2["valence"])
check("same person + same event -> read differently by their past",
      dark["valence"] != bright["valence"])

# the pull is bounded: it moves toward the remembered tone, never past it
check("memory pull stays between the raw reading and the memory (bounded)",
      -0.8 < dark["valence"] < base2["valence"])

# threatening memories raise perceived threat (toward the negative threat/loss pole)
scary = interpret(raw2, person(), recalled=[mem(threat=-0.8, score=0.9)])
check("threatening memories raise perceived threat",
      scary["threat_challenge"] < base2["threat_challenge"])

# familiarity damps novelty: the more it resembles lived experience, the less new it reads
familiar = interpret(raw2, person(), recalled=[mem(score=0.95)])
check("familiarity damps perceived novelty", familiar["novelty"] < base2["novelty"])

# a close match colours more than a distant one
strong = interpret(raw2, person(), recalled=[mem(valence=-0.8, score=0.95)])
weak = interpret(raw2, person(), recalled=[mem(valence=-0.8, score=0.30)])
check("a close memory colours more than a distant one", strong["valence"] < weak["valence"])

# the memory lens surfaces in the brief (LLM) and the tag (UI)
check("lens brief reflects a remembered bad tone",
      "badly" in lens(person(), recalled=bad).lower())
check("read_lens tag reflects a remembered bad tone",
      "memory" in read_lens(person(), recalled=bad).lower())
check("a threatening memory reads as 'braced' in the UI tag",
      "braced" in read_lens(person(), recalled=[mem(threat=-0.8, score=0.9)]).lower())

# a faint, neutral memory injects nothing (no noise in the prompt or the UI)
faint = [mem(score=0.30)]
check("a faint neutral memory adds no brief or tag",
      lens(person(), recalled=faint) == "" and read_lens(person(), recalled=faint) == "")

print("\n" + f"{sum(results)}/{len(results)} checks passed.")
if sum(results) != len(results):
    raise SystemExit(1)
print("ALL CHECKS PASSED -- perception is bent by the character's cognitive lens.")
