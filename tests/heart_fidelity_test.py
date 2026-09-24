import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Personality Fidelity: the math, the discipline, and the "never re-forms" guarantee.

Run with: python tests/heart_fidelity_test.py   (no LLM calls, no network).

Needs a HEART-Bench checkout; point HEART_BENCH_PATH at it. Needs at least one
frozen CHAR_01 snapshot on disk (dev or full) for the end-to-end checks --
those are skipped, not failed, if none exists.
"""

import inspect
import json
import math

from bench.heart import fidelity, heartdata, snapshots
from bench.heart.config import require_heart_bench

require_heart_bench()
fails = []


def check(label, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        fails.append(label)


print("\n1. cosine similarity")
check("identical vectors -> 1.0",
      abs(fidelity.cosine_similarity([1, 0, 0, 0, 0], [1, 0, 0, 0, 0]) - 1.0) < 1e-9)
check("opposite vectors -> -1.0",
      abs(fidelity.cosine_similarity([1, 1, 1, 1, 1], [-1, -1, -1, -1, -1]) - (-1.0)) < 1e-9)
check("orthogonal-ish vectors -> ~0",
      abs(fidelity.cosine_similarity([1, 0, 0, 0, 0], [0, 1, 0, 0, 0])) < 1e-9)
check("scale-invariant (same direction, different magnitude) -> 1.0",
      abs(fidelity.cosine_similarity([0.1, 0.2, 0.3, 0.1, 0.1],
                                     [0.5, 1.0, 1.5, 0.5, 0.5]) - 1.0) < 1e-9)
check("zero vector -> None, not a crash or a fake number",
      fidelity.cosine_similarity([0, 0, 0, 0, 0], [0.5, 0.1, -0.2, 0.3, 0.1]) is None)

print("\n2. mean absolute error")
check("identical vectors -> 0.0",
      fidelity.mean_absolute_error([0.2] * 5, [0.2] * 5) == 0.0)
check("known differences average correctly",
      abs(fidelity.mean_absolute_error([1, 1, 1, 1, 1], [0, 0, 0, 0, 0]) - 1.0) < 1e-9)
check("max possible error (opposite corners) -> 2.0",
      abs(fidelity.mean_absolute_error([1]*5, [-1]*5) - 2.0) < 1e-9)

print("\n3. signed_target rescales HEART's 0..1 to MindForm's -1..+1")
char = heartdata.load_characters()["CHAR_01"]
gt_raw = char["big_five"]
gt_signed = fidelity.signed_target(char)
for k, long in fidelity.LONG.items():
    want = round(gt_raw[long] * 2 - 1, 4)
    check(f"{long}: {gt_raw[long]} -> {want}", gt_signed[k] == want)

print("\n4. direction agreement is deadbanded and never invented")
formed = {"O": 0.5, "C": 0.02, "E": -0.5, "A": -0.02, "N": 0.9}
target = {"O": 0.5, "C": 0.0, "E": 0.5, "A": 0.05, "N": -0.9}
da = fidelity.direction_agreement(formed, target)
check("clear agreement scores True", da["O"] is True)
check("clear disagreement scores False", da["E"] is False)
check("target at exact midpoint scores n/a, not a miss", da["C"] is None)
check("target inside the deadband scores n/a, not a miss", da["A"] is None)
check("disagreement even near saturation is still scored", da["N"] is False)

print("\n5. fidelity.compute() never forms, re-forms, freezes, or ingests")
src = inspect.getsource(fidelity.compute)
for banned in ("ingest_all", "create_neutral", "mfadapter.freeze", "ingest_memory"):
    check(f"compute() contains no call to {banned}()", banned not in src)
check("compute() does call restore() (read an existing snapshot)",
      "mfadapter.restore" in src)

print("\n6. find_behavioral_result never fabricates a number, and excludes aborted runs")
src = inspect.getsource(fidelity.find_behavioral_result)
check("excludes run_invalid", '"run_invalid"' in src)
check("excludes run_error", '"run_error"' in src)
check("excludes leak_detected", '"leak_detected"' in src)
check("counts correctness straight from graded events, not a stored total",
      'g.get("correct")' in src)

print("\n7. end-to-end against whatever CHAR_01 snapshot exists on this machine")
dev = snapshots.best_for_tier("CHAR_01", "dev")
full = snapshots.best_for_tier("CHAR_01", "full")
if not dev and not full:
    print("  SKIP  no CHAR_01 snapshot on this machine")
else:
    tier = "full" if full else "dev"
    fid = fidelity.compute("CHAR_01", tier=tier)
    check("tier matches what was requested", fid["tier"] == tier)
    check("protocol.faithful matches the tier", fid["protocol"]["faithful"] == (tier == "full"))
    check("trait_similarity is a float in [-1, 1] (or None for a zero vector)",
          fid["trait_similarity"] is None or -1.0 - 1e-9 <= fid["trait_similarity"] <= 1.0 + 1e-9)
    check("average_trait_error is non-negative",
          fid["average_trait_error"] >= 0.0)
    check("all five traits present in the per-trait table",
          set(fid["per_trait"].keys()) == set(fidelity.BASIS))
    # recompute independently and check it matches, so the pipeline isn't
    # silently comparing the wrong two vectors
    mf = [fid["per_trait"][k]["mindform"] for k in "OCEAN"]
    ht = [fid["per_trait"][k]["heart_signed"] for k in "OCEAN"]
    recomputed_cos = fidelity.cosine_similarity(mf, ht)
    recomputed_mae = fidelity.mean_absolute_error(mf, ht)
    # fid's values are rounded to 4dp for display; compare at that granularity.
    check("reported Trait Similarity matches an independent recomputation",
          (fid["trait_similarity"] is None and recomputed_cos is None) or
          abs(fid["trait_similarity"] - round(recomputed_cos, 4)) < 1e-9)
    check("reported Average Trait Error matches an independent recomputation",
          abs(fid["average_trait_error"] - round(recomputed_mae, 4)) < 1e-9)
    check("ground_truth_disclosure names the hidden-until-frozen rule",
          "hidden" in fid["ground_truth_disclosure"].lower())

    text = fidelity.render_text(fid, None)
    check("text report renders", "PERSONALITY FIDELITY" in text and "BEHAVIORAL FIDELITY" in text)

print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILURES ({len(fails)}): {fails}"))
sys.exit(1 if fails else 0)
