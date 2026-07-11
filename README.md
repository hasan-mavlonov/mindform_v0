# MindForm

**A personality-formation engine.** You don't prompt a character into existence — you *raise*
one. A character is born with a temperament, and then every experience you tell it forms who
it becomes: its traits, what it prizes, what it believes is right, what it needs, who it
thinks it is, how it speaks, and what it dares to do — each a live, visible faculty, each
changing by bounded psychological dynamics rather than by rewriting a prompt.

```
you: "I failed the exam I studied months for."

the character reads it through who it already is (anxious? starved for approval?
still thinking of itself as the capable one?), forms a little differently because
of it, remembers it, wants differently, carries itself differently -- and answers
in a voice those changes shaped:

"Maybe it's just me, but I'm still working out how it sits with me -- and honestly,
 I just don't want to feel alone in it. For now I'd rather keep my distance."
```

## Quickstart (zero dependencies)

```bash
python console.py            # open http://localhost:8000
```

That's it — the cockpit runs on the Python standard library alone, fully offline. Create a
character from a one-line biography, from explicit fields, or from a 5-question sliders form,
then tell it what happens to it and watch every faculty move.

**Optional upgrades:**
```bash
cp .env.example .env                 # set GEMINI_API_KEY -> the LLM-primary path (richer reading + voice)
pip install -r requirements.txt     # numpy + sentence-transformers -> episodic memory & recall
```
Everything degrades gracefully: no key, no packages, no network — the whole product still works.

## The ten faculties

| Faculty | What it does |
|---|---|
| **Temperament** | the innate OCEAN baseline a character is born with — traits are pulled back toward it |
| **Traits** | the current OCEAN, formed by experience with diminishing returns |
| **Memory** | the hub: every experience logged + embedded; recall colors the next reading — and **what the character lacks re-ranks what it remembers** |
| **Character** | Schwartz values, moral foundations, open beliefs, habits — earned, never innate |
| **Cognitive patterns** | the lens: the same event reads differently through who they are, what they remember, what they want, who they think they are |
| **Motivation & drives** | three SDT needs that starve and get fed (frustration bites harder than satisfaction soothes) |
| **Self-concept** | a self-image that *lags* who they've become, self-regard with an impostor gap, and the ideal/ought selves they measure themselves against |
| **Social expression** | a formed manner of speaking — entrenched by which ways of speaking kept working |
| **Behavior** | the enacted stance: leaning in or holding back gates how hard life lands (the shy spiral, bounded), trained by consequences |
| **The loop** | behavior's outcomes become new lived experience — the ring closes |

Every faculty is visible live in the cockpit — ghost ticks show set-points and aspirations,
flashes show what this experience just did, and the meters honestly disclose their source
(`via gemma` / `offline`).

## The offline appraiser learns with use

Perception is LLM-primary with a trained-head fallback — and the online model is the
teacher: every LLM-read experience is logged as training data, and you can manufacture a
full corpus without any external dataset:

```bash
python bootstrap/distill_appraisal_corpus.py 1000   # the LLM writes + labels diverse experiences
python bootstrap/train_appraisal_head.py            # train the offline head (needs torch)
# core.appraisal now automatically prefers the trained head offline
```

## Tests

```bash
for t in acceptance cognition character genesis drives self_concept expression memory behavior appraisal_distill; do
  python tests/${t}_test.py
done
```
Ten suites, ~300 checks, dependency-free by default (`memory` needs numpy and skips cleanly
without it). Every feedback loop ships with a stability argument and a no-runaway test.

## Docs & layout

Read **[ARCHITECTURE.md](ARCHITECTURE.md)** for the full per-turn pipeline, the layer table,
the math, and the design principles (LLM-primary/offline-complete, bounded dynamics,
derived-not-cached read-outs, one `personality` dict for all state).

```
core/    shared kernel      nodes/   the ten faculties
web/     the cockpit        tests/   the behaviour suites
```
