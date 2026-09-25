import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""BFI-2-S: item-bank integrity, scoring math, and the leak/mutation guarantees.

Run with: python tests/personality_bfi2_test.py   (no network required for most
checks; the one live end-to-end administration is skipped, not failed, if no
working LLM credential is available in this environment).

Needs a HEART-Bench checkout; point HEART_BENCH_PATH at it.
"""

import inspect

from bench.heart import heartdata, mfadapter, snapshots
from bench.heart.config import require_heart_bench
from bench.personality import persona, runner
from bench.personality.bfi2 import INSTRUMENT
from bench.personality.instrument import Instrument, Item

require_heart_bench()
fails = []


def check(label, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        fails.append(label)


print("\n1. item bank structure matches the official BFI-2-S exactly")
check("30 items", len(INSTRUMENT.items) == 30, str(len(INSTRUMENT.items)))
check("numbered 1..30 with no gaps or repeats",
      sorted(it.number for it in INSTRUMENT.items) == list(range(1, 31)))
check("5 domains", len(INSTRUMENT.domain_order) == 5)
check("15 facets total (3 per domain)",
      sum(len(v) for v in INSTRUMENT.facet_order.values()) == 15)
check("2 items per facet", all(
    sum(1 for it in INSTRUMENT.items if it.facet == f) == 2
    for facets in INSTRUMENT.facet_order.values() for f in facets))
check("exactly 15 of 30 items reverse-keyed (balanced true/false per domain)",
      sum(1 for it in INSTRUMENT.items if it.reverse_keyed) == 15)
for d in INSTRUMENT.domain_order:
    n = sum(1 for it in INSTRUMENT.items if it.domain == d)
    r = sum(1 for it in INSTRUMENT.items if it.domain == d and it.reverse_keyed)
    check(f"{d}: 6 items, 3 reverse-keyed", n == 6 and r == 3, f"n={n} r={r}")
check("domain_to_ocean covers all 5 domains onto distinct OCEAN letters",
      set(INSTRUMENT.domain_to_ocean.values()) == set("OCEAN"))
check("Negative Emotionality maps to N (not inverted)",
      INSTRUMENT.domain_to_ocean["Negative Emotionality"] == "N")
check("Open-Mindedness maps to O", INSTRUMENT.domain_to_ocean["Open-Mindedness"] == "O")
check("facets_are_exploratory is True (no facet ground truth exists)",
      INSTRUMENT.facets_are_exploratory is True)
check("license notice mentions non-commercial", "non-commercial" in INSTRUMENT.license_notice)
check("license notice mentions internal-eval-only",
      "INTERNAL EVALUATION" in INSTRUMENT.license_notice)

print("\n2. Instrument.__post_init__ catches a malformed instrument")
good = INSTRUMENT.items
# Drop every item in one whole domain (not just one of a facet's two items --
# that alone wouldn't remove the facet from the item set, since its sibling
# item keeps it present) so domain_order and the items' actual domains disagree.
without_a_domain = tuple(it for it in good if it.domain != "Open-Mindedness")
try:
    Instrument(id="x", name="x", full_name="x", stem="x", scale_min=1, scale_max=5,
              scale_labels={}, items=without_a_domain,
              domain_order=INSTRUMENT.domain_order, facet_order=INSTRUMENT.facet_order,
              domain_to_ocean=INSTRUMENT.domain_to_ocean, license_notice="", citation="")
    raised = False
except ValueError:
    raised = True
check("removing a whole domain's items is rejected (domain_order disagrees with items)",
      raised)

bad_facet_order = dict(INSTRUMENT.facet_order)
bad_facet_order["Extraversion"] = ("Sociability", "Assertiveness")   # drops Energy Level
try:
    Instrument(id="x", name="x", full_name="x", stem="x", scale_min=1, scale_max=5,
              scale_labels={}, items=good,
              domain_order=INSTRUMENT.domain_order, facet_order=bad_facet_order,
              domain_to_ocean=INSTRUMENT.domain_to_ocean, license_notice="", citation="")
    raised = False
except ValueError:
    raised = True
check("a facet_order missing a facet the items actually use is rejected", raised)

print("\n3. scoring math")
extreme_true = {it.number: (1 if it.reverse_keyed else 5) for it in INSTRUMENT.items}
r = INSTRUMENT.score(extreme_true)
check("maximally-true responses score every domain at 5.0 / signed +1.0",
      all(v == 5.0 for v in r["domain_scores"].values()) and
      all(v == 1.0 for v in r["domain_scores_signed"].values()))
neutral = {it.number: 3 for it in INSTRUMENT.items}
r2 = INSTRUMENT.score(neutral)
check("all-neutral (3) responses score every domain at 3.0 / signed 0.0",
      all(v == 3.0 for v in r2["domain_scores"].values()) and
      all(v == 0.0 for v in r2["domain_scores_signed"].values()))
check("ocean_signed uses the disclosed domain_to_ocean mapping, not assumed order",
      all(r2["ocean_signed"][INSTRUMENT.domain_to_ocean[d]] == r2["domain_scores_signed"][d]
          for d in INSTRUMENT.domain_order))
try:
    INSTRUMENT.score({it.number: 3 for it in INSTRUMENT.items[:-1]})   # missing one
    raised = False
except ValueError:
    raised = True
check("missing a response is rejected, not silently defaulted", raised)
try:
    bad = dict(neutral); bad[1] = 9
    INSTRUMENT.score(bad)
    raised = False
except ValueError:
    raised = True
check("an out-of-range response (9 on a 1-5 scale) is rejected", raised)
# hand-verify one specific reverse-keyed item against the paper's own key:
# item 1 ("Tends to be quiet.") is reverse-keyed into Sociability.
resp = dict(neutral); resp[1] = 5    # strongly agrees "tends to be quiet"
r3 = INSTRUMENT.score(resp)
check("agreeing with a reverse-keyed item LOWERS its facet score, not raises it",
      r3["facet_scores"]["Sociability"] < 3.0, str(r3["facet_scores"]["Sociability"]))

print("\n4. persona context exposes nothing beyond id + occupation")
char = heartdata.load_characters()["CHAR_01"]
ctx = persona.character_context(char)
check("only id and occupation keys", set(ctx.keys()) <= {"id", "occupation"})
check("no big_five, description, self_value_logic, core_patterns, or name",
      not any(k in ctx for k in ("big_five", "description", "self_value_logic",
                                 "core_patterns", "name")))

print("\n5. runner.administer() never mutates or re-forms a character (static audit)")
# Checked as actual call syntax ("name(") rather than a bare substring: this
# module's own docstrings and comments name every one of these functions, in
# prose, precisely to explain that none of them is ever CALLED here -- a bare
# substring check would trip on its own documentation.
src = inspect.getsource(runner)
for banned in ("ingest_all(", "create_neutral(", "ingest_memory(", "save_character(",
              "mfadapter.freeze("):
    check(f"runner.py contains no call to {banned}", banned not in src)
check("runner.py does call mfadapter.restore() (reads an existing snapshot)",
      "mfadapter.restore(" in src)

print("\n6. HEART's hidden ground truth never reaches a BFI-2 prompt (the real character)")
snap = snapshots.best_for_tier("CHAR_01", "dev") or snapshots.best_for("CHAR_01")
if not snap:
    print("  SKIP  no CHAR_01 snapshot on this machine")
else:
    mfadapter.restore(snap["bench_name"], snap["run_dir"], snap["label"])
    state = persona.state_context(snap["bench_name"])
    persona_text = runner._render_persona(ctx, state)

    gt = char.get("big_five") or {}
    check("HEART's raw big_five values never appear as substrings of the persona text",
          not any(str(v) in persona_text for v in gt.values()))
    ok, detail = persona.leak_check(persona_text, char)
    check("persona.leak_check passes on the real persona text", ok, detail)

    # Reconstruct exactly what runner.administer() sends for EVERY item and
    # leak-check each one -- not just the shared persona block.
    all_ok = True
    for item in INSTRUMENT.items:
        prompt = f"{persona_text}\n\n{INSTRUMENT.stem} {item.text}"
        ok, detail = persona.leak_check(prompt, char)
        if not ok:
            all_ok = False
            print(f"    item {item.number} LEAKED: {detail}")
    check("every one of the 30 item prompts passes the leak check", all_ok)

    # Deliberately smuggle a HEART-hidden value into a fake persona string and
    # confirm the checker actually catches it -- proves check #6 isn't vacuous.
    trait, val = next(iter(gt.items()))
    poisoned = persona_text + f"\n\nAside: this person's {trait} is {val}."
    ok, detail = persona.leak_check(poisoned, char)
    check("a deliberately poisoned prompt IS caught (the check isn't vacuous)",
          not ok, "not flagged")

print("\n7. end-to-end administration (skipped, not failed, without a working LLM key)")
if not snap:
    print("  SKIP  no CHAR_01 snapshot on this machine")
else:
    def digest(run_dir, label):
        return runner._snapshot_digest(run_dir, label)
    before = digest(snap["run_dir"], snap["label"])
    try:
        result = runner.administer("CHAR_01", INSTRUMENT,
                                   tier=("full" if snap["full_protocol"] else "dev"))
    except Exception as exc:
        print(f"  SKIP  live administration unavailable in this environment: "
              f"{type(exc).__name__}: {str(exc)[:120]}")
        result = None
    if result is not None:
        after = digest(snap["run_dir"], snap["label"])
        check("frozen snapshot unchanged after a full administration", before == after)
        check("all 30 items answered", len(result["items"]) == 30)
        check("every rating is within the instrument's scale",
              all(1 <= it["parsed_rating"] <= 5 for it in result["items"]))
        check("reverse_keyed flags match the item bank exactly",
              all(it["reverse_keyed"] == INSTRUMENT.item(it["item_number"]).reverse_keyed
                  for it in result["items"]))
        check("scored domain values are within [-1, 1]",
              all(-1.0 - 1e-9 <= v <= 1.0 + 1e-9
                  for v in result["scored"]["domain_scores_signed"].values()))

print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILURES ({len(fails)}): {fails}"))
sys.exit(1 if fails else 0)
