# MindForm Architecture

## What it does
A piece of text (an experience) changes a personality made of five OCEAN traits,
each in **[-1, 1]**, on **two timescales**: a fast **mood** that every experience
moves and that decays back toward 0, and a slow **disposition** that only forms
when mood is sustained and that relaxes back toward a set-point when it is not.
The slow disposition is "the personality".

```
text -> MiniLM embedding (encoder.py)
     -> appraisal vector  (appraisal.py: heuristic now / learned head later)
     -> signed push        (impact.py)
     -> two-timescale update: fast mood + slow disposition (updater.py)
     -> persist + memory   (personality.py, memory.py)
```

The Pandora text->OCEAN model (`trait_model.py`) is used only as a read-only readout
(`readout.py`) -- never to encode experiences or compute the push.

## Representation
- **Personality** (`personality.py`): two timescales per trait --
  `{"traits": {k: {"state", "trait"} in [-1,1]}, "experience_count"}`. `state` is
  fast mood (every experience moves it, it decays); `trait` is the slow disposition
  (integrates sustained state, relaxes toward a set-point). The slow `trait` is
  "the personality".
- **Experience** (`config.APPRAISAL_SCHEMA`): an appraisal vector --
  `valence, intensity, novelty, agency, social, outcome, self_relevance, threat_challenge`
  -- the causal ingredients of change, not a trait-expression reading.

## Push + update (two timescales)
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
```
A single vivid party moves *mood* ~0.2 but *disposition* ~0.01; only repetition
graduates mood into a lasting trait (E disposition: 0.01 -> 0.10 -> 0.23 -> 0.63
over 40 parties, bounded), and an unreinforced trait partially relaxes back toward
SETPOINT. The push is signed, so experiences can lower a trait too (helpless terror
raises N; fear faced with mastery lowers it). See `acceptance_test.py` for the
stylized facts this reproduces.

`FORMATION_RATE` scales how hard an experience hits mood; `CONSOLIDATION_RATE`,
`STATE_DECAY` and `HOMEOSTASIS` set how fast mood graduates into disposition and how
strongly an unreinforced disposition returns to baseline.

## Deterministic vs learned -- every component is replaceable
| Component | Today | Later (same interface) |
|---|---|---|
| Encoder | MiniLM (frozen) | — |
| Appraisal extractor | heuristic -> head trained on affect corpora | larger/fine-tuned head |
| Push matrix `M` | theory rules | learned `appraisal -> push` |
| Pandora OCEAN model | read-only readout | richer probe, still read-only |

## Data (no longitudinal dataset required)
- Appraisal head: existing cross-sectional affect/appraisal corpora (EmoBank VAD,
  GoEmotions, appraisal-annotated event sets) via `bootstrap/`.
- Pandora readout: `train_trait_model.py` (`text -> OCEAN`).
- No `experience -> trait-change` labels exist anywhere, so `M` stays rules until
  such data does.

## Run
```
python acceptance_test.py                                 # dependency-free behaviour check
pip install -r requirements.txt && python simulation.py   # full pipeline (encoder in the loop)
python bootstrap/build_affect_dataset.py && python bootstrap/train_appraisal_head.py  # train head (local)
```
