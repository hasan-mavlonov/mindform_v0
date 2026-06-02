"""Acceptance test: text changes the five traits with diminishing returns.

Dependency-free (no torch / numpy / sentence-transformers). Demonstrates the model:

  * a vivid experience moves a trait visibly on the FIRST occurrence (party: E ~0 -> ~0.3)
  * repeating it raises the trait by LESS each time (diminishing returns), bounded by 1
  * the push is signed: an experience can lower a trait, not only raise it
  * identical "fear" content moves N up (helpless) or down (mastery) via appraisal

Run: python3 acceptance_test.py
"""

from personality import default_personality
from appraisal import appraise
from impact import impact
from updater import update_personality


def trait(personality, dim):
    return personality["traits"][dim]


# --- diminishing returns: repeat a vivid party, watch E rise by less each time ---
p = default_personality()
party = appraise("I went to a party and had fun.")
es = [trait(p, "E")]
for _ in range(5):
    p = update_personality(p, impact(party))
    es.append(trait(p, "E"))
steps = [round(es[i + 1] - es[i], 3) for i in range(len(es) - 1)]

# --- signed: a lonely / isolating experience lowers E ---
iso = update_personality(default_personality(),
                         impact(appraise("I spent the whole week alone and sad.")))

# --- mastery vs terror move N in opposite directions ---
terror = update_personality(default_personality(),
                            impact(appraise("I am terrified of everything.")))
mastery = update_personality(default_personality(),
                             impact(appraise("I faced my fear at the party and it went fine.")))

print("E after each party:", [round(e, 3) for e in es])
print("per-party increase:", steps)
print(f"E after isolation : {trait(iso, 'E'):+.3f}")
print(f"N terror={trait(terror, 'N'):+.3f}   N mastery={trait(mastery, 'N'):+.3f}")

checks = {
    "first party moves E visibly (>0.15)": es[1] > 0.15,
    "diminishing returns (each step smaller)": all(steps[i] > steps[i + 1]
                                                   for i in range(len(steps) - 1)),
    "bounded (E stays < 1)": es[-1] < 1.0,
    "signed (isolation lowers E)": trait(iso, "E") < 0,
    "terror raises N": trait(terror, "N") > 0,
    "mastery lowers N": trait(mastery, "N") < 0,
}

print("\nRESULTS:")
for name, ok in checks.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

assert all(checks.values()), "acceptance test FAILED"
print("\nALL CHECKS PASSED -- text changes the five traits with diminishing returns.")
