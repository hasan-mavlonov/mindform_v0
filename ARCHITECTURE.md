# MindForm Architecture

> The design rationale — and why Pandora was removed — is in
> [`docs/RESEARCH_REVIEW.md`](docs/RESEARCH_REVIEW.md). This file is the
> "what it is now."

## What it does
A piece of text (an experience) changes a personality made of five OCEAN traits,
each in **[-1, 1]**, on **three layers**: a fast **mood** that every experience
moves and that decays back toward 0; a slow **disposition** that only forms when
mood is sustained and that relaxes back toward a set-point when it is not; and a
**dispersion** that tracks how variable the expression is. The slow disposition is
"the personality"; together the layers make each trait a *distribution* of states,
not a point (Whole Trait Theory).

```
text -> MiniLM embedding (encoder.py)
     -> appraisal vector  (appraisal.py: heuristic now / learned head later)
     -> signed push        (impact.py)
     -> three-layer update: fast mood + slow disposition + dispersion (updater.py)
     -> persist + memory   (personality.py, memory.py)
```

There is **no `text -> trait` model in the loop.** The formed personality is a
state variable we set and *observe directly* (`evaluation.py`); it is never
estimated from text. (The old Pandora `text -> OCEAN` model was removed — it learned
Reddit author attribution, not personality state, was never validated, and
estimating an observed state is backwards. See the review doc.)

## Representation
- **Personality** (`personality.py`): per trait axis, three values —
  `{"traits": {k: {"state", "trait", "trait_var"}}, "experience_count"}`.
  `state` is fast mood (every experience moves it, it decays); `trait` is the slow
  disposition / the distribution's mean (integrates sustained state, relaxes toward
  a set-point); `trait_var` is the within-person variability of expression (a slow
  EWMA of squared mood swings, MSSD). The slow `trait` is "the personality"; its
  dispersion (`sqrt(trait_var)`) distinguishes a steady trait from a volatile one
  with the same mean.
- **Experience** (`config.APPRAISAL_SCHEMA`): an appraisal vector —
  `valence, intensity, novelty, agency, social, outcome, self_relevance, threat_challenge`
  — the causal ingredients of change, not a trait-expression reading.
- **Basis**: Big Five is the interpretable *coordinate system* we project
  appraisals onto and read trajectories out of — not the mechanism. `config.BASIS`
  is swappable; `config.M` is the only basis-specific component.

## Push + update (three layers)
```
salience = intensity*(0.5+0.5*self_relevance)*(0.5+0.5*novelty)
pull     = M . appraisal                          # which traits move, signed (config.M)
push[k]  = clamp(FORMATION_RATE * salience * pull[k])

# fast mood: every experience moves it; it decays back toward 0
state[k] <- clamp(state[k]*(1 - STATE_DECAY) + push[k])
# slow disposition: integrates SUSTAINED mood, with diminishing returns ...
trait[k] <- clamp(trait[k] + CONSOLIDATION_RATE * state[k] * (1 - |trait[k]|))
# ... and relaxes back toward a set-point when unreinforced (homeostasis)
trait[k] <- clamp(trait[k] - HOMEOSTASIS * (trait[k] - SETPOINT))
# dispersion: within-person variability of expression (drift-invariant, MSSD-style)
trait_var[k] <- (1-VARIABILITY_RATE)*trait_var[k] + VARIABILITY_RATE*(state[k]-prev_state[k])^2
```
A single vivid party moves *mood* ~0.2 but *disposition* ~0.01; only repetition
graduates mood into a lasting trait (E disposition: 0.01 -> 0.10 -> 0.23 -> 0.63
over 40 parties, bounded), and an unreinforced trait partially relaxes back toward
SETPOINT. The push is signed, so experiences can lower a trait too (helpless terror
raises N; fear faced with mastery lowers it). A consistent life keeps dispersion
small; a volatile one widens it. See `acceptance_test.py` for the stylized facts
this reproduces.

Recurrence (`memory.py`) modulates formation: a recurring experience stirs **less**
mood (habituation) but consolidates **more** into disposition (chronicity), and a
new experience is read partly through the appraisals of similar remembered ones
(retrieval-conditioned appraisal).

## Deterministic vs learned — every component is replaceable
| Component | Today | Later (same interface) |
|---|---|---|
| Encoder | MiniLM (frozen) | — |
| Appraisal extractor | heuristic -> head trained on affect corpora | larger/fine-tuned head |
| Push matrix `M` | theory rules | learned `appraisal -> push`; add trait covariance |
| Expression probe | none (state is observed) | validated inventory on a real generator |

## Data (no longitudinal dataset required)
- Appraisal head: existing cross-sectional affect/appraisal corpora (EmoBank VAD,
  GoEmotions, appraisal-annotated event sets) via `bootstrap/`. Today only 3 of 8
  dims (valence/intensity/agency) get real labels; the rest are weak heuristic
  labels — bring in appraisal-annotated event corpora to supervise the others.
- No `experience -> trait-change` labels exist anywhere, so `M` stays rules until
  such data does. This — not the model — is the real bottleneck.

## Run
```
python acceptance_test.py                                 # dependency-free behaviour check
python evaluation.py                                      # dependency-free formation report
pip install -r requirements.txt && python simulation.py   # full pipeline (encoder in the loop)
python bootstrap/build_affect_dataset.py && python bootstrap/train_appraisal_head.py  # train head (local)
```
