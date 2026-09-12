# HEART-Bench × MindForm

Internal R&D harness for running MindForm against
[HEART-Bench](https://github.com/peng-weihan/HEART-BENCH)'s MCQ track.

> **Not investor validation.** HEART-Bench's code is Apache 2.0 but its
> **dataset is CC-BY-NC-4.0**, and the two disagree about the same files. Until
> the authors confirm commercial terms in writing, results from this harness stay
> internal.

Nothing in `core/`, `nodes/` or `web/` is modified. Two runtime patches live in
this package and are documented under *Compromises* below.

## Arms

| Arm | Retrieval | Prompt | Question it answers |
|---|---|---|---|
| **A** `naive_rag` | plain cosine top-k | HEART's template, verbatim | the baseline |
| **D1** `mindform_d1` | MindForm `recall` (cosine × vividness × recency × drive bias) | **byte-identical to A** | is MindForm's *ranking* better? |
| **D2** `mindform_d2` | same as D1 | A's template **plus** a `## Formed Disposition` block | does the *persistent state* add behavioural information beyond retrieval? |

D2 is deliberately **not** protocol-identical to naive RAG. A and D1 are. The two
are never merged into one number: D1 isolates retrieval, D2 − D1 isolates state.

## Protocol

1. Neutral MindForm character — `default_personality()`, every trait `0.0`.
   HEART's `big_five` is **never** used to initialise (that is the withheld label).
2. Only `content_full` is ingested; ids are anonymised (`MEM_CHAR_01_N_HIGH_0049`
   → `MEM_CHAR_01_0049`) because every one of a character's ids carries the same
   trait tag.
3. Ingestion is **chronological by `timeline`**, not file order — file order is one
   solid block of a single trait tag, which would feed the engine a thousand
   consecutive trait-consistent experiences.
4. Formation uses `run_turn`, the normal experience path.
5. After ingestion the state is frozen and hashed (personality JSON + memory log +
   embedding sidecars).
6. Every question restores that snapshot first and re-hashes after answering.
   A mismatch marks the run **INVALID** and stops it.
7. Ground truth is read **only after** the answer is written to the log.

### Leak control

Every prompt is checked before it is sent. A hit stops the run. Blocked:
`big_five`, `description`, `self_value_logic`, `core_patterns`, `name` (its
parenthetical encodes trait + direction), other characters' ids, `correct_answer`,
`is_correct`, `source_character`, raw trait-bearing memory ids, and
`activated_memories_*`. Exposed, matching HEART's own `build_basic_info`:
character **id** and **occupation** only.

## Running it as a live benchmark (start here)

```bash
python -m bench.heart.live          # then open http://127.0.0.1:8500
```

Pick a mode, pick a character, pick which systems to test, click **Run
Benchmark**, and watch it go question by question: scenario, the four options,
the answer as it commits, then the official ground truth revealed after that,
then the running score. Click any row in the history to expand it (retrieved
memories, the state injected into D2, the raw model output); tick **Debug** to
include the exact prompt.

| Mode | What it does | Tier |
|---|---|---|
| **Quick Test** | ~20 official MCQs against an already-prepared snapshot | Development |
| **Character Test** | every official MCQ for one character, same snapshot | Development |
| **Full Protocol Benchmark** | forms the character from all 1,000 memories, then runs every question | Full protocol |

**The tier label is not decoration.** A run against a 50-memory snapshot uses
official questions and official scoring, but the character was formed from 5% of
its memories — it is a development test and the UI says so on every screen and in
the export. Only a run whose snapshot has the full memory set is labelled
FULL PROTOCOL BENCHMARK. At the measured ~33 s/memory that preparation is roughly
9 hours per character, which is why it is a separate, explicitly-confirmed mode.

The UI is a front end, not a second implementation: it calls
`runner.execute_question` and `runner.grade_question`, the same two functions the
CLI uses, so snapshot restore-and-rehash, the pre-send leak check and
reveal-after-commit behave identically. Every run still writes a full
`events.jsonl`.

## Running from the CLI

```bash
python -m bench.heart.runner --stage 0                 # 50 memories, 3 questions
python -m bench.heart.traitdiff --characters CHAR_01,CHAR_08 --memories 50
python -m bench.heart.runner --stage 1                 # all memories, 15 questions
python -m bench.heart.report    --run <run_id> --save
python -m bench.heart.dashboard --run <run_id>         # http://127.0.0.1:8420
```

Runs resume: rerun with the same `--run-id` and the frozen snapshot is restored
instead of re-ingesting.

Environment: `GEMINI_API_KEY` (or `LLM_API_KEY`), optional `HEART_MODEL`,
`HEART_TEMPERATURE`, `HEART_TOP_K`, `HEART_REASONING_EFFORT`,
`HEART_RATE_IN` / `HEART_RATE_OUT` (unset → cost reported as `n/a`),
`HEART_BENCH_PATH`.

## Output

```
data/benchmark/heart/<run_id>/
  events.jsonl            every event, machine-readable
  run.log                 the same story in prose
  report.txt              --save
  snapshots/frozen/       the frozen character + manifest.json with hashes
```

`events.jsonl` carries, per question: arm, character, snapshot id, question,
scenario, trigger, options, every retrieved memory with its score, the MindForm
state used, the **exact prompt** and its sha256, model/temperature/max_tokens,
the selected letter, raw model output, ground truth, correctness, latency,
token counts, cost, snapshot hash before and after, `state_mutated`, the leak
check, and any parser errors. "Why did MindForm answer C on question 12?" is
answerable from this file alone.

## Compromises

Each of these is a real deviation, logged in every event rather than buried.

1. **`generate_reply` patched out during ingestion.** `run_turn` calls it *after*
   `save_character`, and its output only reaches the returned read-out — so it
   cannot affect formation. Saves one LLM call per memory.
2. **`recall(min_score=0.0)`** instead of the product's `0.25` relevance floor, so
   MindForm returns a full top-k the way naive RAG does. Without this the arms
   would differ in *how many* memories reach the model, not just which.
3. **`recall(k=30)`** instead of MindForm's default `k=3`. At the default, D1 would
   send ~2,100 tokens of memory against A's ~21,400 and lose for reasons unrelated
   to personality.
4. **`reasoning_effort="none"` on answer calls.** The default Gemini model is a
   thinking model and `max_tokens` caps thinking + output together: measured,
   `max_tokens=1600` yielded **62** visible output tokens and truncated JSON.
   Applied identically to all three arms. HEART's own runners do not set it, so
   this is a deviation from their published setup — but not between our arms.
5. **Embedder is MiniLM for both A and D1.** HEART's published naive RAG uses
   `qwen3-embedding-4b`. Holding the embedder constant across arms is what makes
   A-vs-D1 a fair retrieval comparison; it also means our A is **not** their
   published A, and our absolute numbers are not comparable to their table.
   MiniLM truncates at 256 tokens while these memories average ~714, so both arms
   retrieve on roughly the first third of each memory.
6. **Answering is not MindForm's shipped inference path.** `generate_reply`
   produces one- or two-sentence spoken replies, not MCQ selections, so the answer
   call uses HEART's own prompt and JSON schema. The engine supplies the state and
   the retrieval; it does not supply the decision format.
7. **Ingestion LLM calls still run with thinking enabled**, because that path lives
   in `core/llm.py` and is out of scope for this harness. Measured cost: ~28 s and
   5 calls per memory.

## What a good result here would and would not mean

It would mean MindForm's retrieval and/or formed state improves single-shot,
in-character behavioural prediction over plain cosine retrieval on the same
memories, with the same model.

It would **not** mean anything about persistence or evolution. Every HEART
question is independent, over a fixed memory set, with no state carried between
questions. The benchmark's own results also show base-model choice moving
accuracy ~30 points while memory systems move it 1–3 — so any effect here is
small relative to model choice, and a 1-point difference is noise.
