import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Genesis / character-creation acceptance test -- dependency-free.

Demonstrates the temperament seed and character authoring:

  * genesis(bio) seeds an OCEAN baseline (mu) and per-trait stickiness (tau)
  * the current traits are born AT the baseline (x == mu)
  * two contrasting bios yield distinct baselines (distinguishable from birth)
  * create_character(fields) keeps explicit identity fields verbatim and seeds
    temperament from the free-text background
  * build_character(identity, mu) uses an explicitly chosen baseline, no LLM
    (the manual questionnaire path)
  * the roster saves / lists / reloads multiple named characters
  * an experience still moves a trait and preserves the identity/temperament fields
    (the temperament pull-back + drift now live in updater.py; see acceptance_test.py)

Run: python3 genesis_test.py
"""

import tempfile

import core.personality as P
from core.config import BASIS, DEFAULT_TAU
from nodes.temperament import genesis, create_character, build_character
from core.personality import default_personality
from core.appraisal import appraise
from core.impact import impact
from core.updater import update_personality

import core.llm as llm_mod
llm_mod.LLM_API_KEY = ""        # force the heuristic seed -- no network in tests


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

    # unique_name: creation must never silently overwrite an existing save (the bug --
    # two characters resolving to the same slug, most commonly two "unnamed" ones from
    # bios with no extractable name, or two authored people who share a name).
    free_name_unchanged = P.unique_name("Zarina") == "Zarina"
    P.save_character(build_character({"name": "Zarina"}, chosen_mu)[0])
    taken_name_disambiguated = P.unique_name("Zarina") == "Zarina 2"
    P.save_character(build_character({"name": "Zarina 2"}, chosen_mu)[0])
    third_collision_disambiguated = P.unique_name("Zarina") == "Zarina 3"
    blank_name_falls_back = P.unique_name(None) == "unnamed" or P.unique_name("") == "unnamed"

    # end to end: two DIFFERENT people whose names can't be told apart (both blank, as a
    # bio with no capitalized name would seed) must both survive on the roster distinctly,
    # not collapse into one overwritten "unnamed" record.
    p1, _, _ = genesis("a shy, anxious, sensitive poet, easily overwhelmed.")   # no name in text
    p1["identity"]["name"] = P.unique_name(p1["identity"].get("name"))
    P.save_character(p1)
    p2, _, _ = genesis("a bold, outgoing, disciplined, calm athlete.")         # also no name
    p2["identity"]["name"] = P.unique_name(p2["identity"].get("name"))
    P.save_character(p2)
    unnamed_both_survive = (
        p1["identity"]["name"] != p2["identity"]["name"]
        and P.load_character(p1["identity"]["name"])["temperament"]["mu"]["N"] > 0
        and P.load_character(p2["identity"]["name"])["temperament"]["mu"]["E"] > 0
    )

# the trait update still moves a trait and preserves the identity/temperament fields
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
    "unique_name leaves a free name unchanged": free_name_unchanged,
    "unique_name disambiguates a taken name": taken_name_disambiguated,
    "unique_name disambiguates past a second collision": third_collision_disambiguated,
    "unique_name falls back to 'unnamed' for a blank name": blank_name_falls_back,
    "two nameless creations both survive the roster, not one overwritten": unnamed_both_survive,
    "trait update still works + preserves temperament/identity": update_preserves,
}

print("\nRESULTS:")
for name, ok in checks.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

assert all(checks.values()), "genesis test FAILED"
print("\nALL CHECKS PASSED -- characters: born, authored by fields, chosen by hand, and rostered.")
