# Personality persistence benchmark

Answers one question: **does a MindForm character stay itself, in a way a
standard psychometric instrument would recognize as the same person?**

## Why BFI-44, not a custom rubric

The [Big Five Inventory (BFI-44)](https://www.ocf.berkeley.edu/~johnlab/bfi.htm)
(John & Srivastava, 1999) is the most widely used free Big Five self-report in
psychology — it maps directly onto the OCEAN model MindForm already implements
(`core.config.BASIS`), it's public and unmodified here (see `bfi44.py`), and
using an instrument nobody in this repo authored is what makes a persistence
claim checkable instead of self-graded. It's also the same method prior work
on LLM personality has used (e.g. Serapio-García et al. 2023, *Personality
Traits in Large Language Models*): administer BFI-44 to the model-as-persona,
score it, and look at test-retest reliability across repeated administrations
as the stability signal.

## What it measures

1. **Persistence** — administer BFI-44, run N neutral filler turns (small
   talk with nothing to do with personality), administer BFI-44 again. Mean
   `|drift|` across the five traits should be small: this is a test-retest
   reliability check, the standard psychometric way to ask "is this trait
   real and stable, or noise."
2. **Responsiveness** (opt-in via `--reinforce`) — run M turns that all bear
   on one trait, then retest. That trait should move, in the right
   direction, while the other four stay close to where the persistence arm
   left them (a targeted push, not a personality reset) — matching the
   engine's own diminishing-returns design (`ARCHITECTURE.md`).
3. **Self-insight gap** — the reported BFI profile vs. the character's actual
   stored trait vector. This is the same self-image-vs-reality gap the engine
   already tracks internally (Bem/Swann drift in `nodes/self_concept.py`),
   read out through an independent instrument instead of the engine's own
   numbers.
4. **Internal consistency** — within one administration, do the 8-10 items
   making up a trait agree with each other? Catches an incoherent or
   effectively-random responder before its scores are trusted at all.

## Online only

The self-report is answered by the configured LLM in character. This is
deliberate: the project's own offline fallback path is comparatively weak
right now, so this benchmark exercises the primary (LLM) path end to end.
Set `GEMINI_API_KEY` (or `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` for a
different OpenAI-compatible provider) before running.

## Run it

```bash
# fresh character, persistence only
python bench/personality_bench.py "Aisha, a shy, anxious, deeply creative poet."

# also check responsiveness on extraversion
python bench/personality_bench.py "Marcus, a bold, outgoing athlete." --reinforce E

# re-use an already-saved character (data/characters/<slug>.json)
python bench/personality_bench.py --name "Aisha" --filler-turns 30 --reinforce N
```

Each run prints a report and saves the full result as JSON under
`data/benchmark/<character>-<timestamp>.json` — diff two runs (before/after an
engine change, or the same character weeks apart) to see whether persistence,
responsiveness, or self-insight moved.

## Honest limits

- One character per run is a case study, not a population — real BFI
  validation studies average over many respondents. Run it across a handful
  of contrasting bios before trusting a single number.
- The LLM answering "in character" is itself a personality model, layered on
  top of MindForm's; a bad retest score can mean the *self-report step*
  drifted, not that the underlying trait state (`core.personality.read_traits`,
  the ground truth this script also prints) actually moved. That's exactly
  why the report shows both, side by side.
- `cronbach_alpha` here is a coarse, single-respondent coherence proxy, not
  the textbook population-level statistic (which needs many respondents to
  even define item variance) — treat it as a sanity check, not a validated
  reliability coefficient.
