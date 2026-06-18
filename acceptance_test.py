"""Acceptance test: experience forms the traits, and temperament reins them in.

Dependency-free (no torch / numpy / sentence-transformers). Demonstrates the model:

  * a vivid experience moves a trait visibly on the FIRST occurrence (party: E ~0 -> ~0.2)
  * repeating it raises the trait by LESS each time (diminishing returns)
  * the push is signed: an experience can lower a trait, not only raise it
  * identical "fear" content moves N up (helpless) or down (mastery) via appraisal
  * TEMPERAMENT (Slice 2): the baseline pulls a trait back, so a repeated experience
    settles at a fixed point BETWEEN baseline and the extreme -- two characters with the
    same experiences but different temperaments diverge -- and a sustained shift slowly
    drifts the baseline itself.

Run: python3 acceptance_test.py
"""

from personality import default_personality
from appraisal import appraise
from impact import impact
from updater import update_personality


def trait(personality, dim):
    return personality["traits"][dim]


party = appraise("I went to a party and had fun.")

# --- diminishing returns: repeat a vivid party, watch E rise by less each time ---
p = default_personality()
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


# --- temperament (Slice 2): baseline pull + slow drift ---
def settle(mu_E, tau_E, occurrences=20):
    """Born at an E-baseline with stickiness tau_E, then party `occurrences` times."""
    person = default_personality()
    person["temperament"]["mu"]["E"] = mu_E
    person["temperament"]["tau"]["E"] = tau_E
    person["traits"]["E"] = mu_E              # born at baseline
    for _ in range(occurrences):
        person = update_personality(person, impact(party))
    return person


neutral = settle(0.0, 0.30)       # average baseline, ordinary stickiness
sticky = settle(-0.7, 0.60)       # resilient introvert: low E baseline, high stickiness
ne, se = trait(neutral, "E"), trait(sticky, "E")
drifted_mu = sticky["temperament"]["mu"]["E"]

print("E after each party:", [round(e, 3) for e in es])
print("per-party increase:", steps)
print(f"E after isolation : {trait(iso, 'E'):+.3f}")
print(f"N terror={trait(terror, 'N'):+.3f}   N mastery={trait(mastery, 'N'):+.3f}")
print(f"after 20 parties -> neutral E={ne:+.3f}   sticky-introvert E={se:+.3f} "
      f"(its baseline drifted -0.70 -> {drifted_mu:+.2f})")

checks = {
    "first party moves E visibly (>0.15)": es[1] > 0.15,
    "diminishing returns (each step smaller)": all(steps[i] > steps[i + 1]
                                                   for i in range(len(steps) - 1)),
    "bounded (E stays < 1)": es[-1] < 1.0,
    "signed (isolation lowers E)": trait(iso, "E") < 0,
    "terror raises N": trait(terror, "N") > 0,
    "mastery lowers N": trait(mastery, "N") < 0,
    "temperament: trait settles below the +/-1 wall": ne < 0.9,
    "temperament: same parties, sticky introvert ends far below neutral": se < ne - 0.3,
    "temperament: sustained experience drifts the baseline up": drifted_mu > -0.7,
}

print("\nRESULTS:")
for name, ok in checks.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

assert all(checks.values()), "acceptance test FAILED"
print("\nALL CHECKS PASSED -- experience forms the traits; temperament reins them in.")
