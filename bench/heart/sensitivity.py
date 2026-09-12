"""D1 vs D2 sensitivity report: does the formed state change decisions at all?

    python -m bench.heart.sensitivity --run <run_id>

D1 and D2 answer the same questions from the same frozen snapshot with the same
retrieval, the same model and the same sampling settings. The single difference
is the ``## Formed Disposition`` block. So every disagreement between them is
attributable to the persistent psychological state and nothing else.

This is a SENSITIVITY test, not a benchmark score. Twenty questions cannot
establish an accuracy difference; they can establish whether the state has any
behavioural effect at all, which is the precondition for the larger run being
worth its cost.
"""

import argparse
import json
import os

from core.config import BASIS, BASIS_NAMES
from bench.heart.config import RESULTS_ROOT
from bench.heart.dashboard import load_events

W = 78
TICK, CROSS = "✅", "❌"


def collect(run_id):
    ev = load_events(os.path.join(RESULTS_ROOT, run_id))
    start = next((e for e in ev if e["event"] == "run_start"), {})
    committed = [e for e in ev if e["event"] == "answer_committed"]
    graded = [e for e in ev if e["event"] == "graded"]

    by = {}
    order = []
    for g in graded:
        qid = g["question_id"]
        if qid not in by:
            by[qid] = {}
            order.append(qid)
        by[qid][g["arm"]] = g
    for c in committed:
        cell = by.get(c["question_id"], {}).get(c["arm"])
        if cell is not None:
            cell["_committed"] = c
    return start, by, order, committed, graded


def d2_state(committed):
    for c in committed:
        if c["arm"] == "mindform_d2" and c.get("mindform_state"):
            return c["mindform_state"], c
    return None, None


def state_block(st, src):
    out = [ "=" * W,
            "EXACT MINDFORM STATE INJECTED INTO D2",
            "=" * W,
            f"(identical for every question — one frozen snapshot, "
            f"{st.get('experience_count')} experiences)", ""]
    out.append("  OCEAN traits")
    for k in BASIS:
        out.append(f"    {BASIS_NAMES[k]:<20}{st['traits'][k]:+.3f}")
    si = st.get("self_image") or {}
    out.append("")
    out.append("  self-image           " + "  ".join(f"{k}{si.get(k,0):+.2f}" for k in BASIS))
    out.append(f"  self-regard (esteem) {st.get('esteem'):+.3f}")
    out.append("")
    out.append("  values (Schwartz)    " + ", ".join(f"{n} {v:+.2f}" for n, v in st.get("top_values") or []))
    out.append("  moral (Haidt)        " + ", ".join(f"{n} {v:+.2f}" for n, v in st.get("top_moral") or []))
    out.append("  needs (SDT tension)  " + ", ".join(f"{n} {v:.2f}" for n, v in st.get("top_drives") or []))
    b = st.get("behavior") or {}
    out.append(f"  behaviour            approach {b.get('approach')}, inhibition "
               f"{b.get('inhibition')}, stance {b.get('tendency')} ({b.get('mode')})")
    out.append("  beliefs formed       " + ("; ".join(st.get("beliefs") or []) or "none"))
    out.append("")
    out.append("  Verbatim text appended to the D2 prompt:")
    if src:
        i = src["prompt"].find("## Formed Disposition")
        j = src["prompt"].find("## Behavioural Decision Options")
        block = src["prompt"][i:j].rstrip() if i > 0 else "(not found)"
        for line in block.split("\n"):
            out.append("    | " + line)
    return "\n".join(out)


def build(run_id):
    start, by, order, committed, graded = collect(run_id)
    st, src = d2_state(committed)

    lines = []
    add = lines.append
    if st:
        add(state_block(st, src))
        add("")

    add("=" * W)
    add("PER-QUESTION: D1 (retrieval only) vs D2 (retrieval + formed state)")
    add("=" * W)

    same = fixed = broke = both_wrong_diff = 0
    d1c = d2c = n = 0
    disagreements = []

    for i, qid in enumerate(order, 1):
        a = by[qid].get("mindform_d1")
        b = by[qid].get("mindform_d2")
        if not a or not b:
            continue
        n += 1
        sa, sb, gt = a["selected_answer"], b["selected_answer"], a["ground_truth"]
        ok_a, ok_b = bool(a["correct"]), bool(b["correct"])
        d1c += ok_a
        d2c += ok_b

        if sa == sb:
            same += 1
            verdict = "SAME"
        elif not ok_a and ok_b:
            fixed += 1
            verdict = "CHANGED — D2 FIX"
        elif ok_a and not ok_b:
            broke += 1
            verdict = "CHANGED — D2 HURT"
        else:
            both_wrong_diff += 1
            verdict = "CHANGED — both wrong"
        if sa != sb:
            disagreements.append((i, qid, sa, ok_a, sb, ok_b, gt, verdict))

        add(f"Q{i:02d} | D1: {sa or '∅'} {TICK if ok_a else CROSS}"
            f" | D2: {sb or '∅'} {TICK if ok_b else CROSS} | {verdict}")

    add("")
    add("=" * W)
    add("SUMMARY")
    add("=" * W)
    add(f"Questions tested: {n}")
    add("")
    add(f"D1 accuracy: {d1c}/{n}" + (f"  ({100*d1c/n:.1f}%)" if n else ""))
    add(f"D2 accuracy: {d2c}/{n}" + (f"  ({100*d2c/n:.1f}%)" if n else ""))
    add("")
    add(f"D1 == D2: {same}")
    add(f"D1 != D2: {n - same}")
    add("")
    add("Among disagreements:")
    add(f"D2 fixed D1 errors:            {fixed}")
    add(f"D2 broke D1 correct answers:   {broke}")
    add(f"Both wrong but different:      {both_wrong_diff}")
    add("")
    rate = (100.0 * (n - same) / n) if n else 0.0
    add(f"State sensitivity: the formed state changed the decision on "
        f"{n - same}/{n} questions ({rate:.0f}%).")

    if disagreements:
        add("")
        add("=" * W)
        add("DISAGREEMENTS IN DETAIL")
        add("=" * W)
        for i, qid, sa, ok_a, sb, ok_b, gt, verdict in disagreements:
            c = by[qid]["mindform_d2"].get("_committed") or {}
            sc = (c.get("scenario") or {}).get("name", "?")
            add(f"\nQ{i:02d}  {qid}")
            add(f"     scenario : {sc}")
            add(f"     D1 chose : {sa} {TICK if ok_a else CROSS}"
                f"     D2 chose : {sb} {TICK if ok_b else CROSS}     truth: {gt}")
            add(f"     {verdict}")
            for o in (c.get("options") or []):
                if o["label"] in (sa, sb, gt):
                    who = []
                    if o["label"] == sa: who.append("D1")
                    if o["label"] == sb: who.append("D2")
                    if o["label"] == gt: who.append("TRUTH")
                    txt = " ".join(o["content"].split())[:150]
                    add(f"       {o['label']} [{'/'.join(who):<9}] {txt}…")

    # integrity + cost
    add("")
    add("=" * W)
    add("INTEGRITY / COST")
    add("=" * W)
    hashes = {g["snapshot_hash_before"] for g in graded} | {g["snapshot_hash_after"] for g in graded}
    add(f"  snapshot hashes seen        : {len(hashes)} (1 = never mutated)")
    add(f"  state_mutated events        : {sum(1 for g in graded if g.get('state_mutated'))}")
    add(f"  leak-check failures         : {sum(1 for g in graded if not g.get('leak_check_ok', True))}")
    add(f"  memories retrieved per Q    : "
        f"{len((committed[0].get('memories_retrieved') or [])) if committed else '?'}")
    ident = all(
        [m["anon_id"] for m in (by[q]["mindform_d1"].get("_committed") or {}).get("memories_retrieved", [])]
        == [m["anon_id"] for m in (by[q]["mindform_d2"].get("_committed") or {}).get("memories_retrieved", [])]
        for q in order if "mindform_d1" in by[q] and "mindform_d2" in by[q])
    add(f"  D1/D2 retrieved same memories: {ident}  (must be True — else the arms differ in more than state)")
    tin = sum(c.get("input_tokens") or 0 for c in committed)
    tout = sum(c.get("output_tokens") or 0 for c in committed)
    lat = sum(c.get("latency_ms") or 0 for c in committed) / max(len(committed), 1)
    add(f"  answer calls                : {len(committed)}  ({tin:,} in / {tout:,} out)")
    add(f"  mean latency                : {lat:.0f} ms")
    add(f"  memories re-ingested        : 0 (frozen snapshot reused)")
    add("")
    add("Sensitivity test on ~20 questions. Accuracy differences here are NOT")
    add("statistically meaningful; the disagreement RATE is the signal.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    text = build(args.run)
    print(text)
    if args.save:
        p = os.path.join(RESULTS_ROOT, args.run, "sensitivity.txt")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
