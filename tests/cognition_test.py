import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Cognitive Patterns test: the interpretation lens bends perception by current traits.

Dependency-free (pure functions, no torch / numpy / network).
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

# neuroticism: more perceived threat, darker valence
anxious = interpret(raw, person(N=0.8))
check("anxious reads more threat than neutral", anxious["threat_challenge"] > neutral["threat_challenge"])
check("anxious reads a darker valence than neutral", anxious["valence"] < neutral["valence"])
calm = interpret(raw, person(N=-0.8))
check("calm reads less threat than neutral", calm["threat_challenge"] < neutral["threat_challenge"])

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

print("\n" + f"{sum(results)}/{len(results)} checks passed.")
if sum(results) != len(results):
    raise SystemExit(1)
print("ALL CHECKS PASSED -- perception is bent by the character's cognitive lens.")
