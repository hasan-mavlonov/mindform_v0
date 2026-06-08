"""Genesis / temperament acceptance test (Slice 1) -- dependency-free.

Demonstrates the temperament seed and character authoring:

  * genesis(bio) seeds an OCEAN baseline (mu) and per-trait stickiness (tau)
  * the current traits are born AT the baseline (x == mu)
  * two contrasting bios yield distinct baselines (distinguishable from birth)
  * create_character(fields) keeps explicit identity fields verbatim and seeds
    temperament from the free-text background
  * build_character(identity, mu) uses an explicitly chosen baseline, no LLM
    (the manual questionnaire path)
  * the roster saves / lists / reloads multiple named characters
  * Slice 1 has no temperament dynamics yet: an experience still moves the traits
    by the existing diminishing-returns rule, and identity/temperament ride through

Run: python3 genesis_test.py
"""

import tempfile

import personality as P
from config import BASIS, DEFAULT_TAU
from temperament import genesis, create_character, build_character
from personality import default_personality
from appraisal import appraise
from impact import impact
from updater import update_personality


anxious_bio = "Aisha, a shy, anxious, sensitive poet, easily overwhelmed."
bold_bio = "Marcus, a bold, outgoing, disciplined, calm athlete."

aisha, aisha_src, _ = genesis(anxious_bio)
marcus, marcus_src, _ = genesis(bold_bio)

born_at_baseline = all(aisha["traits"][d] == aisha["temperament"]["mu"][d] for d in BASIS)
distinct = aisha["temperament"]["mu"] != marcus["temperament"]["mu"]
anxious_more_N = aisha["temperament"]["mu"]["N"] > marcus["temperament"]["mu"]["N"]
bold_more_E = marcus["temperament"]["mu"]["E"] > aisha["temperament"]["mu"]["E"]
identity_captured = bool(aisha["identity"])
tau_in_range = all(0.0 <= aisha["temperament"]["tau"][d] <= 1.0 for d in BASIS)

# create_character: explicit immutable fields + a free-text background
created, created_src, _ = create_character({
    "name": "Aisha", "age": "24", "origin": "Tashkent",
    "religion": "Muslim", "language": "Uzbek",
    "background": "a shy, anxious, sensitive poet",
})
fields = created["identity"]
fields_verbatim = (fields.get("name") == "Aisha"
                   and fields.get("origin") == "Tashkent"
                   and fields.get("religion") == "Muslim")
seeded_from_background = created["temperament"]["mu"]["N"] > 0   # "anxious" -> +N

# build_character: explicit identity + explicitly chosen OCEAN baseline (no LLM)
chosen_mu = {"O": 0.4, "C": 0.8, "E": -0.4, "A": 0.0, "N": -0.8}
manual, manual_src, _ = build_character({"name": "Bordi", "origin": "Bukhara"}, chosen_mu)
manual_uses_chosen = (manual["identity"]["name"] == "Bordi"
                      and manual["temperament"]["mu"] == chosen_mu
                      and manual["traits"]["C"] == 0.8
                      and manual_src == "manual"
                      and manual["temperament"]["tau"]["C"] == DEFAULT_TAU)

# roster: save / list / reload multiple characters (temp dir -> no side effects)
with tempfile.TemporaryDirectory() as tmp:
    P.CHARACTERS_DIR = tmp
    P.save_character(created)   # Aisha
    P.save_character(manual)    # Bordi
    names = sorted((c["identity"].get("name") for c in P.list_characters()))
    roster_lists = names == ["Aisha", "Bordi"]
    roster_reloads = P.load_character("Bordi")["temperament"]["mu"]["C"] == 0.8

# Slice 1: dynamics off -- the trait update still works and preserves new fields
after = update_personality(default_personality(), impact(appraise("I went to a party and had fun.")))
update_preserves = after["traits"]["E"] > 0 and "temperament" in after and "identity" in after

print("Aisha   baseline:", {d: round(aisha["temperament"]["mu"][d], 2) for d in BASIS}, f"({aisha_src})")
print("Marcus  baseline:", {d: round(marcus["temperament"]["mu"][d], 2) for d in BASIS}, f"({marcus_src})")
print("Created identity:", fields, f"({created_src})")
print("Manual  baseline:", chosen_mu, f"({manual_src})")

checks = {
    "born at baseline (x == mu)": born_at_baseline,
    "distinguishable from birth": distinct,
    "anxious bio -> higher N baseline": anxious_more_N,
    "bold bio -> higher E baseline": bold_more_E,
    "identity captured": identity_captured,
    "tau in [0, 1]": tau_in_range,
    "create_character keeps identity fields verbatim": fields_verbatim,
    "create_character seeds temperament from background": seeded_from_background,
    "build_character uses chosen mu (no LLM)": manual_uses_chosen,
    "roster saves + lists multiple characters": roster_lists,
    "roster reloads a character by name": roster_reloads,
    "trait update still works + preserves temperament/identity": update_preserves,
}

print("\nRESULTS:")
for name, ok in checks.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

assert all(checks.values()), "genesis test FAILED"
print("\nALL CHECKS PASSED -- characters: born, authored by fields, chosen by hand, and rostered.")
