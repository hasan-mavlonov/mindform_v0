"""Summary report for a HEART-Bench run, built only from the logs.

    python -m bench.heart.report --run <run_id>

No LLM is called and no engine state is touched: every number here is read back
out of ``events.jsonl``.
"""

import argparse
import json
import os

from bench.heart.config import RESULTS_ROOT
from bench.heart.dashboard import load_events, ARM_ORDER, ARM_LABEL

W = 78


def _rows(events, kind):
    return [e for e in events if e["event"] == kind]


def _s(rows, key):
    return sum(r.get(key) or 0 for r in rows)


def build(run_id):
    run_dir = os.path.join(RESULTS_ROOT, run_id)
    ev = load_events(run_dir)
    start = next((e for e in ev if e["event"] == "run_start"), {})
    ingest = _rows(ev, "ingest_step")
    done = next((e for e in ev if e["event"] == "ingest_done"), {})
    committed = _rows(ev, "answer_committed")
    graded = _rows(ev, "graded")
    invalid = next((e for e in ev if e["event"] in ("run_invalid", "leak_detected")), None)

    out = []
    add = out.append
    add("=" * W)
    add(f"HEART-Bench × MindForm — {run_id}")
    add("=" * W)
    add(f"character      {start.get('character')}")
    add(f"model          {start.get('model')}  temp={start.get('temperature')}  "
        f"max_tokens={start.get('max_tokens')}  top_k={start.get('top_k')}")
    add(f"embedder       {start.get('embedder')}")
    add(f"memories       {start.get('n_memories')} ingested chronologically")
    add(f"snapshot       {done.get('snapshot_id','—')}")
    add(f"status         {'INVALID: ' + str(invalid.get('reason') or invalid.get('detail')) if invalid else 'valid'}")
    add("")

    add("-" * W)
    add("PER-ARM RESULTS")
    add("-" * W)
    add(f"{'arm':<26}{'acc':>10}{'in tok':>10}{'out tok':>10}{'calls':>7}"
        f"{'lat ms':>9}{'cost':>9}")
    for arm in ARM_ORDER:
        g = [x for x in graded if x["arm"] == arm]
        c = [x for x in committed if x["arm"] == arm]
        if not g and not c:
            continue
        corr = sum(1 for x in g if x.get("correct"))
        acc = f"{corr}/{len(g)}" if g else "—"
        pct = f" ({100*corr/len(g):.1f}%)" if g else ""
        cost = _s(c, "cost_usd")
        cost_s = f"${cost:.4f}" if any(x.get("cost_usd") is not None for x in c) else "n/a"
        lat = round(_s(c, "latency_ms") / len(c)) if c else 0
        add(f"{ARM_LABEL[arm]:<26}{acc + pct:>10}{_s(c,'input_tokens'):>10,}"
            f"{_s(c,'output_tokens'):>10,}{len(c):>7}{lat:>9}{cost_s:>9}")
    add("")

    add(f"{'':<26}{'state failures':>16}{'leak failures':>16}{'parse retries':>16}")
    for arm in ARM_ORDER:
        g = [x for x in graded if x["arm"] == arm]
        c = [x for x in committed if x["arm"] == arm]
        if not g and not c:
            continue
        add(f"{ARM_LABEL[arm]:<26}"
            f"{sum(1 for x in g if x.get('state_mutated')):>16}"
            f"{sum(1 for x in g if not x.get('leak_check_ok', True)):>16}"
            f"{sum(1 for x in c if x.get('errors')):>16}")
    add("")

    add("-" * W)
    add("CALLS / TOKENS BY TYPE")
    add("-" * W)
    add(f"  memory ingestion      {len(ingest)*5:>7} LLM calls  "
        f"({len(ingest)} memories × 5: appraisal, trait push, values, moral, beliefs)")
    add(f"  embeddings            {len(ingest):>7} local MiniLM encodes — $0, no API")
    add(f"  state updates         {0:>7} calls — deterministic arithmetic")
    add(f"  retrieval             {0:>7} calls — local cosine over the sidecar")
    add(f"  MCQ answering         {len(committed):>7} calls  "
        f"{_s(committed,'input_tokens'):,} in / {_s(committed,'output_tokens'):,} out")
    if done.get("seconds"):
        add(f"  ingestion wall clock  {done['seconds']/60:.1f} min "
            f"({done['seconds']/max(len(ingest),1):.2f} s/memory)")
    add("")

    # ---- per-question comparison ----
    add("-" * W)
    add("PER-QUESTION COMPARISON")
    add("-" * W)
    qids = []
    table = {}
    for g in graded:
        q = g["question_id"]
        if q not in table:
            table[q] = {}
            qids.append(q)
        table[q].setdefault(g["arm"], []).append(g)
    for q in qids:
        short = q.replace("Q_CHAR_", "").replace("SCN_", "")
        cells = []
        truth = None
        for arm in ARM_ORDER:
            rs = table[q].get(arm)
            if not rs:
                cells.append(f"{arm.split('_')[-1].upper()}=—")
                continue
            truth = rs[0].get("ground_truth")
            marks = " ".join(f"{r.get('selected_answer') or '∅'}"
                             f"{'✅' if r.get('correct') else '❌'}" for r in rs)
            label = {"naive_rag": "RAG", "mindform_d1": "D1", "mindform_d2": "D2"}[arm]
            cells.append(f"{label}={marks}")
        add(f"  {short:<34} " + " | ".join(cells) + f"   truth={truth}")
    add("")

    # ---- D2 vs D1 disagreements ----
    add("-" * W)
    add("WHERE STATE CHANGED THE ANSWER (D2 vs D1)")
    add("-" * W)
    add("  Surfaced for inspection, not as evidence of superiority.")
    diffs = 0
    for q in qids:
        d1 = table[q].get("mindform_d1") or []
        d2 = table[q].get("mindform_d2") or []
        for a, b in zip(d1, d2):
            if a.get("selected_answer") != b.get("selected_answer"):
                diffs += 1
                add(f"  {q.replace('Q_CHAR_','')}  rep{a.get('repeat',0)}  "
                    f"D1={a.get('selected_answer')}{'✅' if a.get('correct') else '❌'} → "
                    f"D2={b.get('selected_answer')}{'✅' if b.get('correct') else '❌'}  "
                    f"(truth {a.get('ground_truth')})")
    if not diffs:
        add("  none — D2's state did not change any answer in this run.")
    add("")

    # ---- determinism note ----
    temps = {c.get("temperature") for c in committed}
    reps = {c.get("repeat", 0) for c in committed}
    if temps == {0} and len(reps) > 1:
        add("-" * W)
        add("NOTE ON REPEATS")
        add("-" * W)
        add("  Every answer call ran at temperature 0. Repeats therefore measure provider")
        add("  nondeterminism only, not sampling variance — they are NOT statistical")
        add("  replicates. Check the per-question rows above: identical letters across")
        add("  repeats mean the run was effectively deterministic and n = questions, not")
        add("  questions × repeats.")
        add("")

    add("=" * W)
    add("Internal R&D only — HEART-Bench data is CC-BY-NC and its commercial licence")
    add("is unresolved. Not investor validation.")
    add("=" * W)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    text = build(args.run)
    print(text)
    if args.save:
        path = os.path.join(RESULTS_ROOT, args.run, "report.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
