"""HEART-Bench runner: ingest, freeze, then answer under three arms.

    python -m bench.heart.runner --stage 0
    python -m bench.heart.runner --stage 1
    python -m bench.heart.runner --character CHAR_08 --memories 50 --questions 0

Integrity rules enforced here, not assumed:
  * the frozen snapshot is restored before EVERY question and re-hashed after it;
    any difference marks the run INVALID and stops it;
  * the ground truth is read only after the answer has been committed to the log;
  * every prompt is leak-checked before it is sent, and a hit stops the run.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bench.heart import arms, heartdata, mfadapter
from bench.heart.config import (
    RESULTS_ROOT, MODEL, TEMPERATURE, MAX_TOKENS, TOP_K, TEMPERATURE_NOTE,
    RATE_IN, RATE_OUT, HEART_PATH,
)
from bench.heart.logbook import Logbook, new_run_id

ARMS = ("naive_rag", "mindform_d1", "mindform_d2")
ARM_LABEL = {"naive_rag": "Naive RAG (A)", "mindform_d1": "MindForm D1 (retrieval)",
             "mindform_d2": "MindForm D2 (retrieval + state)"}


class RunInvalid(RuntimeError):
    pass


# --------------------------------------------------------------------------
def _bench_name(character_id):
    return f"heartbench {character_id}"


def ingest_phase(log, char, memories, name, run_dir, resume=True):
    """Ingest chronologically and freeze. Resumes from an existing snapshot."""
    frozen_dir = os.path.join(run_dir, "snapshots", "frozen")
    if resume and os.path.exists(os.path.join(frozen_dir, "manifest.json")):
        with open(os.path.join(frozen_dir, "manifest.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)
        sid = mfadapter.restore(name, run_dir, "frozen")
        log.say(f"Resumed frozen snapshot {sid[:16]}… "
                f"({manifest.get('snapshot_id','?')[:16]}… in manifest)")
        if sid != manifest.get("snapshot_id"):
            raise RunInvalid("restored snapshot does not match its manifest hash")
        return sid, manifest, []

    # Mid-run resume. run_turn calls save_character every turn, so a long
    # ingestion that dies at memory 700 has 700 memories on disk already; only
    # the freeze is missing. Pick up from experience_count rather than starting
    # over -- a 1,000-memory ingestion is hours long and must survive a restart.
    done = 0
    if resume:
        try:
            done = int(mfadapter.load_character(name).get("experience_count") or 0)
        except Exception:
            done = 0
    if done and done < len(memories):
        log.say(f"Partial ingestion found: {done}/{len(memories)} already formed — resuming.")
        log.event("ingest_resume_partial", character=char["id"], already=done,
                  remaining=len(memories) - done)
    elif done >= len(memories) and done:
        log.say(f"All {done} memories already ingested; freezing.")
    else:
        mfadapter.create_neutral(name, char.get("occupation", "N/A"))

    log.banner(f"INGESTION | {char['id']} | {len(memories)} memories, chronological",
               f"neutral start · reply generation skipped · engine path: run_turn"
               + (f" · resuming at {done}" if done else ""))
    log.event("ingest_start", character=char["id"], memories=len(memories),
              neutral_start=(done == 0), big_five_used=False, resumed_from=done)

    t0 = time.time()
    steps = mfadapter.ingest_all(name, memories[done:], log, on_step=log.ingest_tick)
    dt = time.time() - t0

    sid, manifest = mfadapter.freeze(name, run_dir, "frozen")
    errs = [s for s in steps if s.get("error")]
    log.say(f"\nIngested {len(steps)} memories in {dt/60:.1f} min "
            f"({len(errs)} errors). Snapshot {sid[:16]}…")
    final = steps[-1]["traits_after"] if steps else mfadapter.trait_vector(
        mfadapter.load_character(name))
    log.say("Formed traits: " + "  ".join(f"{k}{final.get(k,0):+.3f}" for k in "OCEAN"))
    log.event("ingest_done", character=char["id"], snapshot_id=sid,
              steps=len(steps), errors=len(errs), seconds=round(dt, 1),
              traits_final=final, manifest=manifest)
    return sid, manifest, steps


# --------------------------------------------------------------------------
def retrieve_for(arm, name, query, k):
    if arm == "naive_rag":
        return mfadapter.naive_rag(name, query, k)
    return mfadapter.recall_memories(name, query, k)


def question_query(scenario):
    """The retrieval query: the situation as it would present itself."""
    trig = scenario.get("trigger_event") or {}
    return "\n".join(str(x) for x in [
        scenario.get("name", ""), scenario.get("context_text", ""),
        trig.get("message_content", ""), trig.get("action_required", ""),
    ] if x)


def execute_question(log, arm, char, char_pub, scenario, question, name, run_dir,
                     frozen_id, forbidden, repeat=0, snapshot_label="frozen",
                     on_stage=None):
    """Restore -> retrieve -> build -> leak-check -> answer -> COMMIT.

    Returns the committed record. The ground truth is deliberately NOT read
    here: reveal-after-commit is enforced by this function simply never
    touching the answer key. ``grade_question`` does that, after the caller has
    written this record.

    Both the CLI runner and the live UI go through this one function, so the
    integrity guarantees cannot drift apart between the two front ends.
    ``on_stage`` is an optional callback (stage_name, payload) for a UI that
    wants to show progress mid-question.
    """
    def stage(name_, **payload):
        if on_stage:
            on_stage(name_, payload)

    # 1. restore the frozen state; nothing from a previous question may survive
    stage("restoring")
    restored = mfadapter.restore(name, run_dir, snapshot_label)
    if restored != frozen_id:
        raise RunInvalid(f"restore mismatch: {restored[:16]} != {frozen_id[:16]}")
    hash_before, _ = mfadapter.snapshot_hash(name)

    options = heartdata.public_options(question)
    arms.assert_no_forbidden_fields(options)

    stage("retrieving")
    query = question_query(scenario)
    memories, retr_meta = retrieve_for(arm, name, query, TOP_K)

    state = mfadapter.state_summary(name) if arm == "mindform_d2" else None
    prompt = arms.build_prompt(char_pub, scenario, memories, options, state=state)

    # 2. leak check BEFORE the call
    ok, detail = arms.leak_check(prompt, forbidden, char)
    if not ok:
        log.event("leak_detected", arm=arm, question_id=question["question_id"],
                  detail=detail, prompt_sha256=arms.prompt_sha(prompt))
        raise RunInvalid(f"LEAK in prompt for {question['question_id']} [{arm}]: {detail}")

    stage("answering", options=options, memories=memories, state=state,
          retrieval=retr_meta)

    # 3. answer, then COMMIT before the truth is touched
    result = arms.answer(prompt)

    committed = {
        "arm": arm, "character_id": char["id"], "snapshot_id": frozen_id,
        "question_id": question["question_id"], "scenario_id": question["scenario_id"],
        "repeat": repeat,
        "scenario": {"name": scenario.get("name"),
                     "setting": scenario.get("setting"),
                     "context_text": scenario.get("context_text")},
        "trigger_event": scenario.get("trigger_event"),
        "options": options,
        "memories_retrieved": memories,
        "retrieval": retr_meta,
        "mindform_state": state,
        "prompt": prompt,
        "prompt_sha256": arms.prompt_sha(prompt),
        "model": result["model"], "temperature": result["temperature"],
        "max_tokens": result["max_tokens"],
        "selected_answer": result["choice"], "raw_output": result["raw_output"],
        "parse_method": result["parse_method"],
        "latency_ms": result["latency_ms"],
        "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"],
        "cost_usd": result["cost_usd"],
        "snapshot_hash_before": hash_before,
        "leak_check": {"ok": True, "detail": ""},
        "errors": result["errors"],
    }
    log.event("answer_committed", **committed)
    return committed


def grade_question(log, committed, question, name, tallies):
    """Reveal the answer key and grade. Only ever called AFTER execute_question
    has written its record, so the key cannot influence the answer."""
    arm = committed["arm"]
    truth = heartdata.ground_truth(question)          # first touch of the key
    correct = (committed["selected_answer"] == truth)

    hash_before = committed["snapshot_hash_before"]
    hash_after, _ = mfadapter.snapshot_hash(name)
    mutated = (hash_after != hash_before)

    c, n = tallies.get(arm, (0, 0))
    tallies[arm] = (c + (1 if correct else 0), n + 1)

    log.event("graded", arm=arm, question_id=committed["question_id"],
              repeat=committed.get("repeat", 0),
              selected_answer=committed["selected_answer"], ground_truth=truth,
              correct=correct, snapshot_hash_before=hash_before,
              snapshot_hash_after=hash_after, state_mutated=mutated, leak_check_ok=True,
              input_tokens=committed["input_tokens"],
              output_tokens=committed["output_tokens"],
              cost_usd=committed["cost_usd"], latency_ms=committed["latency_ms"])
    return {"ground_truth": truth, "correct": correct, "state_mutated": mutated,
            "snapshot_hash_after": hash_after}


def run_question(log, arm, char, char_pub, scenario, question, name, run_dir,
                 frozen_id, idx, total, tallies, forbidden, repeat=0):
    """CLI wrapper: the shared path plus console output."""
    def stage(kind, payload):
        if kind == "answering":
            log.question_header(ARM_LABEL[arm], char["id"], idx, total, scenario,
                                scenario.get("trigger_event"), payload["options"])
            log.retrieval_block(payload["memories"], payload["retrieval"]["embedder"])
            log.state_block(payload["state"])

    committed = execute_question(log, arm, char, char_pub, scenario, question, name,
                                 run_dir, frozen_id, forbidden, repeat=repeat,
                                 on_stage=stage)
    log.commit(committed["selected_answer"])

    g = grade_question(log, committed, question, name, tallies)

    log.reveal(g["ground_truth"], g["correct"], {ARM_LABEL[a]: tallies[a] for a in tallies},
               committed["latency_ms"],
               {"input_tokens": committed["input_tokens"],
                "output_tokens": committed["output_tokens"]},
               committed["cost_usd"])
    log.integrity(committed["snapshot_hash_before"], g["snapshot_hash_after"],
                  g["state_mutated"], True)

    if g["state_mutated"]:
        raise RunInvalid(f"state mutated while answering {question['question_id']} [{arm}]")
    return g["correct"]


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, choices=[0, 1], default=0)
    ap.add_argument("--character", default="CHAR_01")
    ap.add_argument("--memories", type=int, default=None, help="0 = all")
    ap.add_argument("--questions", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    if args.stage == 0:
        n_mem = args.memories if args.memories is not None else 50
        n_q = args.questions if args.questions is not None else 3
        repeats = args.repeats if args.repeats is not None else 1
    else:
        n_mem = args.memories if args.memories is not None else 0      # all
        n_q = args.questions if args.questions is not None else 15
        repeats = args.repeats if args.repeats is not None else 3

    selected = [a for a in args.arms.split(",") if a in ARMS]
    run_id = args.run_id or new_run_id(f"stage{args.stage}-{args.character.lower()}")
    log = Logbook(run_id, RESULTS_ROOT)

    log.banner(f"HEART-Bench × MindForm | run {run_id}",
               f"stage {args.stage} · {args.character} · memories={n_mem or 'all'} · "
               f"questions={n_q} · repeats={repeats} · arms={','.join(selected)}")
    log.say(f"model={MODEL}  temperature={TEMPERATURE}  max_tokens={MAX_TOKENS}  top_k={TOP_K}")
    log.say(f"note: {TEMPERATURE_NOTE}")
    if RATE_IN is None or RATE_OUT is None:
        log.say("note: HEART_RATE_IN / HEART_RATE_OUT unset -> cost reported as n/a")

    characters = heartdata.load_characters()
    scenarios = heartdata.load_scenarios()
    char = characters[args.character]
    questions = heartdata.questions_for(args.character, limit=n_q)
    forbidden = heartdata.forbidden_strings(char, questions)

    all_mem = heartdata.ingestible_memories(char)
    memories = all_mem if not n_mem else all_mem[:n_mem]
    mfadapter.register_id_map(all_mem)   # text-sha256 -> HEART anon_id, for traces

    log.event("run_start", stage=args.stage, character=args.character,
              n_memories=len(memories), n_questions=len(questions), repeats=repeats,
              arms=selected, model=MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
              top_k=TOP_K, heart_path=HEART_PATH,
              embedder="all-MiniLM-L6-v2 (both arms — see report for the fairness note)",
              forbidden_string_count=len(forbidden))

    name = _bench_name(args.character)
    tallies = {}
    invalid = None
    try:
        frozen_id, manifest, _ = ingest_phase(log, char, memories, name, run_dir=log.dir,
                                              resume=not args.no_resume)
        total_cells = len(questions) * len(selected) * repeats
        i = 0
        for rep in range(repeats):
            for q in questions:
                scenario = scenarios[q["scenario_id"]]
                for arm in selected:
                    i += 1
                    run_question(log, arm, char, heartdata.character_public(char),
                                 scenario, q, name, log.dir, frozen_id,
                                 i, total_cells, tallies, forbidden, repeat=rep)
    except RunInvalid as exc:
        invalid = str(exc)
        log.say(f"\n*** RUN INVALID: {exc}", level="ERROR")
        log.event("run_invalid", reason=str(exc))
    except KeyboardInterrupt:
        log.say("\ninterrupted — snapshot and events preserved; rerun with --run-id "
                f"{run_id} to resume", level="WARN")
        log.event("run_interrupted")

    log.banner("SUMMARY", run_id)
    for arm in selected:
        c, n = tallies.get(arm, (0, 0))
        pct = (100.0 * c / n) if n else 0.0
        log.say(f"  {ARM_LABEL[arm]:<34} {c}/{n} = {pct:5.1f}%")
    log.event("run_end", tallies={a: list(tallies.get(a, (0, 0))) for a in selected},
              invalid=invalid)
    log.say(f"\nevents : {log.events_path}")
    log.say(f"log    : {log.log_path}")
    log.say(f"report : python -m bench.heart.report --run {run_id}")
    log.say(f"dash   : python -m bench.heart.dashboard --run {run_id}")
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
