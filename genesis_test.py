"""Genesis / temperament acceptance test (Slice 1) -- dependency-free.

Demonstrates the temperament seed:

  * genesis seeds an OCEAN baseline (mu) and per-trait stickiness (tau)
  * the current traits are born AT the baseline (x == mu)
  * two contrasting bios yield distinct baselines -- a character is distinguishable
    from birth, not only after a long conversation
  * create_character keeps explicit identity fields verbatim and seeds temperament
    from the free-text background
  * Slice 1 has no temperament dynamics yet: an experience still moves the traits
    by the existing diminishing-returns rule, and identity/temperament ride through

Run: python3 genesis_test.py
"""

from config import BASIS
from temperament import genesis, create_character
from personality import default_personality
from appraisal import appraise
from impact import impact
from updater import update_personality


anxious_bio = "Aisha, a shy, anxious, sensitive poet, easily overwhelmed."
bold_bio = "Marcus, a bold, outgoing, disciplined, calm athlete."

aisha, aisha_src, _ = genesis(anxious_bio)
marcus, marcus_src, _ = genesis(bold_bio)

# born at baseline: every current trait equals its mu
born_at_baseline = all(
    aisha["traits"][d] == aisha["temperament"]["mu"][d] for d in BASIS
)

# distinguishable from birth
distinct = aisha["temperament"]["mu"] != marcus["temperament"]["mu"]
anxious_more_N = aisha["temperament"]["mu"]["N"] > marcus["temperament"]["mu"]["N"]
bold_more_E = marcus["temperament"]["mu"]["E"] > aisha["temperament"]["mu"]["E"]

identity_captured = bool(aisha["identity"])
tau_in_range = all(0.0 <= aisha["temperament"]["tau"][d] <= 1.0 for d in BASIS)

# create_character: explicit immutable fields + a free-text background
created, created_src, _ = create_character({
    "name": "Aisha",
    "age": "24",
    "origin": "Tashkent",
    "religion": "Muslim",
    "language": "Uzbek",
    "background": "a shy, anxious, sensitive poet",
})
fields = created["identity"]
fields_verbatim = (fields.get("name") == "Aisha"
                   and fields.get("origin") == "Tashkent"
                   and fields.get("religion") == "Muslim")
seeded_from_background = created["temperament"]["mu"]["N"] > 0   # "anxious" -> +N
created_born_at_baseline = all(
    created["traits"][d] == created["temperament"]["mu"][d] for d in BASIS
)

# Slice 1: dynamics off -- the existing trait update still works and preserves
# the new identity/temperament fields.
blank = default_personality()
after = update_personality(blank, impact(appraise("I went to a party and had fun.")))
update_preserves = after["traits"]["E"] > 0 and "temperament" in after and "identity" in after

print("Aisha   baseline:", {d: round(aisha["temperament"]["mu"][d], 2) for d in BASIS}, f"({aisha_src})")
print("Marcus  baseline:", {d: round(marcus["temperament"]["mu"][d], 2) for d in BASIS}, f"({marcus_src})")
print("Created identity:", fields, f"({created_src})")

checks = {
    "born at baseline (x == mu)": born_at_baseline,
    "distinguishable from birth": distinct,
    "anxious bio -> higher N baseline": anxious_more_N,
    "bold bio -> higher E baseline": bold_more_E,
    "identity captured": identity_captured,
    "tau in [0, 1]": tau_in_range,
    "create_character keeps identity fields verbatim": fields_verbatim,
    "create_character seeds temperament from background": seeded_from_background,
    "created character born at baseline": created_born_at_baseline,
    "trait update still works + preserves temperament/identity": update_preserves,
}

print("\nRESULTS:")
for name, ok in checks.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

assert all(checks.values()), "genesis test FAILED"
print("\nALL CHECKS PASSED -- characters are born with a distinct temperament.")
