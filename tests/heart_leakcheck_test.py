import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Leak checker: MindForm's own formed state is allowed, HEART's labels are not.

Run with: python tests/heart_leakcheck_test.py   (no LLM calls, no network).

The full CHAR_01 run aborted on its first MCQ with
``big_five:conscientiousness=0.5``. That was a false positive: D2 exists to state
MindForm's formed OCEAN numbers, and the old check flagged any number sitting
near a trait word if it merely looked like HEART's hidden value -- unescaped, so
"0.5" matched the first three characters of 0.53, and unsigned, so -0.53 matched
too. These checks pin both halves: the derived state passes, and every genuinely
withheld label is still caught.

Needs a HEART-Bench checkout; point HEART_BENCH_PATH at it.
"""

from bench.heart import arms, heartdata
from bench.heart.config import require_heart_bench

require_heart_bench()
fails = []


def check(label, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        fails.append(label)


char = heartdata.load_characters()["CHAR_01"]
gt = char["big_five"]
questions = heartdata.questions_for("CHAR_01")
forbidden = heartdata.forbidden_strings(char, questions)
scen = heartdata.load_scenarios()
q0 = questions[0]
s0 = scen[q0["scenario_id"]]
pub = heartdata.character_public(char)
opts = heartdata.public_options(q0)


def state_with(**traits):
    base = {"O": 0.11, "C": 0.22, "E": -0.33, "A": 0.44, "N": -0.05}
    base.update(traits)
    return {"traits": base, "top_values": [("SE", 1.0)], "top_moral": [("care", 0.4)],
            "top_drives": [("belonging", 0.7)], "esteem": 0.12,
            "self_image": {d: 0.1 for d in "OCEAN"}, "beliefs": ["I finish what I start."]}


def prompt_with(state):
    return arms.build_prompt(pub, s0, [], opts, state=state)


print("\n1. the exact failure that aborted the CHAR_01 run")
for c in (0.53, 0.50, -0.50, -0.53, 0.5, 0.59, -0.59):
    st = state_with(C=c)
    ok, detail = arms.leak_check(prompt_with(st), forbidden, char, state=st,
                                 live_traits=st["traits"])
    check(f"MindForm-derived conscientiousness {c:+.2f} passes "
          f"(HEART's hidden value is {gt['conscientiousness']})", ok, detail)

print("\n2. every trait, at the value HEART hides, when the engine formed it")
letter = {"openness": "O", "conscientiousness": "C", "extraversion": "E",
          "agreeableness": "A", "neuroticism": "N"}
for trait, val in gt.items():
    st = state_with(**{letter[trait]: float(val)})
    ok, detail = arms.leak_check(prompt_with(st), forbidden, char, state=st,
                                 live_traits=st["traits"])
    check(f"derived {trait}={val} passes on provenance", ok, detail)

print("\n3. the same numbers are still caught OUTSIDE the state block")
for trait, val in gt.items():
    st = state_with()
    p = prompt_with(st) + f"\n\nAside: this person's {trait} is {val}."
    ok, detail = arms.leak_check(p, forbidden, char, state=st, live_traits=st["traits"])
    check(f"leaked {trait}={val} in the prompt body is caught", not ok, "not flagged")
    check(f"  ...and is named in the detail", f"big_five:{trait}" in detail, detail)

print("\n4. a near-miss outside the block is NOT flagged (no prefix matching)")
st = state_with()
p = prompt_with(st) + "\n\nAside: this person's conscientiousness is 0.53."
ok, _ = arms.leak_check(p, forbidden, char, state=st, live_traits=st["traits"])
check("0.53 is not HEART's 0.5 and passes", ok)
p = prompt_with(st) + "\n\nAside: this person's conscientiousness is -0.5."
ok, _ = arms.leak_check(p, forbidden, char, state=st, live_traits=st["traits"])
check("-0.5 is not HEART's +0.5 and passes", ok)

print("\n5. withheld strings are still caught, inside the state block too")
st = state_with()
cases = [
    ("character description", (char.get("description") or "")[:80]),
    ("name parenthetical", (char.get("name") or "").split("(")[-1].rstrip(")")),
    ("answer-key token", "correct_answer"),
    ("source_character token", "source_character"),
    ("trait tag", "_N_HIGH_"),
]
for label, payload in cases:
    if not payload or len(payload) < 4:
        print(f"  SKIP  {label} (not present in this character)")
        continue
    st2 = state_with()
    st2["beliefs"] = [payload]                    # smuggle it INTO the D2 block
    p = arms.build_prompt(pub, s0, [], opts, state=st2)
    ok, detail = arms.leak_check(p, forbidden, char, state=st2,
                                 live_traits=st2["traits"])
    check(f"{label} hidden inside the D2 block is caught", not ok, "not flagged")

other = next((o.get("source_character") for q in questions for o in q.get("options", [])
              if o.get("source_character") and o["source_character"] != char["id"]), None)
if other:
    p = prompt_with(st) + f"\n\nThis option came from {other}."
    ok, _ = arms.leak_check(p, forbidden, char, state=st, live_traits=st["traits"])
    check("another character's id is caught", not ok)

print("\n6. the state block itself cannot be forged")
st = state_with(C=0.22)
p = prompt_with(st).replace("Conscientiousness +0.22", "Conscientiousness +0.99")
ok, detail = arms.leak_check(p, forbidden, char, state=st, live_traits=st["traits"])
check("a block edited after rendering is caught", not ok, "not flagged")
check("  ...reported as a provenance failure", "state:block" in detail, detail)

p = prompt_with(st)
ok, detail = arms.leak_check(p, forbidden, char, state=None, live_traits=None)
check("a block with no engine state behind it is caught", not ok, "not flagged")

ok, detail = arms.leak_check(p, forbidden, char, state=st,
                             live_traits={**st["traits"], "C": 0.77})
check("state disagreeing with the character on disk is caught", not ok, "not flagged")
check("  ...names the trait that disagrees", "state:C" in detail, detail)

print("\n7. a character seeded from HEART's labels is caught")
seeded = state_with(**{letter[k]: float(v) for k, v in gt.items()})
ok, detail = arms.leak_check(prompt_with(seeded), forbidden, char, state=seeded,
                             live_traits=seeded["traits"])
check("all five traits equal to the withheld big_five is caught", not ok, "not flagged")
check("  ...reported as seeding", "seeded" in detail, detail)

print("\n8. D1 and naive RAG prompts carry no state block")
p = arms.build_prompt(pub, s0, [], opts, state=None)
check("no Formed Disposition section", arms.STATE_BLOCK_MARKER not in p)
ok, detail = arms.leak_check(p, forbidden, char)
check("a stateless prompt passes", ok, detail)

print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILURES ({len(fails)}): {fails}"))
sys.exit(1 if fails else 0)
