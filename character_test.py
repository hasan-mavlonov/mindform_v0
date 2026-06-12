"""Character formation checks: Schwartz values form from experience; habits recur.

Run with: python character_test.py   (no network -- the LLM key is forced off so the
deterministic heuristic path runs).
"""

from config import VALUES, BASIS, HABIT_MIN_RECURRENCE
import values as values_mod
from character import (
    default_character, update_values, note_habit, higher_order,
    read_values, dominant_value,
)
import personality as P
from temperament import build_character

values_mod.LLM_API_KEY = ""        # force the heuristic path -- no network in tests

results = []


def check(name, ok):
    results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


def approx(a, b, eps=1e-6):
    return abs(a - b) < eps


# --- 1. A blank character is ten neutral values and no habits -----------------
blank = default_character()
check("blank character has all ten values at 0",
      set(blank["values"]) == set(VALUES) and all(v == 0.0 for v in blank["values"].values()))
check("blank character has no habits", blank["habits"] == [])


# --- 2. Values form with diminishing returns, signed, independent -------------
ch = default_character()
seq = [ch["values"]["SD"]]
for _ in range(3):
    ch = update_values(ch, {"SD": 0.3})
    seq.append(ch["values"]["SD"])
# 0 -> 0.30 -> 0.51 -> 0.657
check("first push moves the value by its full amount", approx(seq[1], 0.30))
check("diminishing returns (0.30, 0.51, 0.657)", approx(seq[2], 0.51) and approx(seq[3], 0.657))
incs = [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]
check("each step is smaller than the last", all(incs[i] > incs[i + 1] for i in range(len(incs) - 1)))
check("a push touches only its own value (others stay 0)",
      all(ch["values"][v] == 0.0 for v in VALUES if v != "SD"))

down = update_values(default_character(), {"SD": -0.4})
check("negative push lowers a value", down["values"]["SD"] < 0)

bounded = default_character()
for _ in range(60):
    bounded = update_values(bounded, {"AC": 0.9})
check("values stay bounded in [-1, 1]", -1.0 <= bounded["values"]["AC"] <= 1.0)


# --- 3. Higher-order roll-up onto Schwartz's four poles -----------------------
vals = {v: 0.0 for v in VALUES}
vals.update({"SD": 1.0, "ST": 1.0, "HE": 0.5})
ho = higher_order(vals)
# openness_to_change = (1*1 + 1*1 + 0.5*0.5) / (1+1+0.5) = 2.25 / 2.5 = 0.9
check("openness_to_change rolls up correctly", approx(ho["openness_to_change"], 0.9))
# self_enhancement = (0 + 0 + 0.5*0.5) / 2.5 = 0.1  (hedonism is split across two poles)
check("hedonism is split across two higher-order poles", approx(ho["self_enhancement"], 0.1))
check("empty poles roll up to 0", approx(ho["conservation"], 0.0) and approx(ho["self_transcendence"], 0.0))


# --- 4. Habits: a recurring experience crosses the threshold ------------------
ch = default_character()
below = note_habit(ch, "I go for a run", HABIT_MIN_RECURRENCE - 1)
check("below the recurrence threshold, no habit forms", below["habits"] == [])

at = note_habit(ch, "I go for a run", HABIT_MIN_RECURRENCE)
check("at the threshold a habit is recorded", len(at["habits"]) == 1 and at["habits"][0]["count"] == HABIT_MIN_RECURRENCE)

again = note_habit(at, "I go for a run", HABIT_MIN_RECURRENCE + 4)
check("seeing the same habit again dedupes and keeps the higher count",
      len(again["habits"]) == 1 and again["habits"][0]["count"] == HABIT_MIN_RECURRENCE + 4)

two = note_habit(again, "I call my mother", HABIT_MIN_RECURRENCE)
check("a different recurring experience is a separate habit", len(two["habits"]) == 2)


# --- 5. Heuristic formation (no network): valid shape + sane direction --------
push_a, source_a, _ = values_mod.values_push_from_text("I started a fun new project and made it work.")
check("heuristic push is labelled 'heuristic'", source_a == "heuristic")
check("heuristic push covers all ten values, each in [-1, 1]",
      set(push_a) == set(VALUES) and all(-1.0 <= push_a[v] <= 1.0 for v in VALUES))
check("agentic, successful experience raises self-direction and achievement",
      push_a["SD"] > 0 and push_a["AC"] > 0)

push_b, _, _ = values_mod.values_push_from_text("I was terrified and scared and afraid and alone.")
check("a threatening experience raises the value placed on security", push_b["SE"] > 0)


# --- 6. Persistence: migration backfills character onto older saves -----------
legacy = {                                   # a pre-character save (traits + temperament only)
    "identity": {"name": "Legacy"},
    "temperament": {"mu": {d: 0.0 for d in BASIS}, "tau": {d: 0.3 for d in BASIS}},
    "traits": {d: 0.1 for d in BASIS},
    "experience_count": 5,
}
migrated = P.migrate(legacy)
check("migration backfills a full character onto a pre-character save",
      isinstance(migrated.get("character"), dict)
      and set(migrated["character"]["values"]) == set(VALUES)
      and migrated["character"]["habits"] == [])

partial = {
    "traits": {d: 0.0 for d in BASIS},
    "temperament": {"mu": {d: 0.0 for d in BASIS}, "tau": {d: 0.3 for d in BASIS}},
    "character": {"values": {"SD": 0.5}, "habits": []},
}
fixed = P.migrate(partial)
check("migration seeds missing value keys while preserving existing ones",
      set(fixed["character"]["values"]) == set(VALUES)
      and approx(fixed["character"]["values"]["SD"], 0.5)
      and fixed["character"]["values"]["UN"] == 0.0)


# --- 7. Newborn characters carry the substrate --------------------------------
born, _, _ = build_character({"name": "Newborn"}, {d: 0.0 for d in BASIS})
check("a freshly built character has the character layer",
      set(born["character"]["values"]) == set(VALUES)
      and all(v == 0.0 for v in born["character"]["values"].values()))
check("default_personality includes the character layer",
      set(P.default_personality()["character"]["values"]) == set(VALUES))


# --- 8. Read-outs -------------------------------------------------------------
ch = default_character()
ch = update_values(ch, {"BE": 0.6, "PO": -0.4})
ordered = list(read_values(ch).items())
check("read_values orders by strength (strongest first)", ordered[0][0] == "benevolence")
dom = dominant_value(ch)
check("dominant_value names the strongest value", dom["key"] == "BE" and dom["value"] > 0)


# --- 9. Bridge snapshot exposes the character block ---------------------------
from web import engine_bridge as bridge   # noqa: E402  (after path is set up by running at root)

snap = bridge.snapshot(born)
check("snapshot carries a character block with ten value rows",
      "character" in snap and len(snap["character"]["values"]) == len(VALUES))
check("snapshot character block includes the higher-order roll-up",
      set(snap["character"]["higher_order"]) ==
      {"openness_to_change", "self_enhancement", "conservation", "self_transcendence"})


print("\nRESULTS:")
passed = sum(1 for _, ok in results if ok)
for name, ok in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
print(f"\n{passed}/{len(results)} checks passed.")
if passed == len(results):
    print("ALL CHECKS PASSED -- character forms values from experience and settles into habits.")
else:
    raise SystemExit(1)
