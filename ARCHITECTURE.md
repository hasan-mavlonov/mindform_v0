# MindForm Architecture

## What it does
A piece of text (an experience) changes a personality made of five OCEAN traits,
each in **[-1, 1]**. Experiences push the traits, and the push is applied with
**diminishing returns**: a trait moves fast while near 0 and ever more slowly as it
approaches +/-1.

```
text -> MiniLM embedding (encoder.py)
     -> appraisal vector  (appraisal.py: heuristic now / learned head later)
     -> signed push        (impact.py)
     -> diminishing-returns update of the five traits (updater.py)
     -> persist + memory   (personality.py, memory.py)
```

The Pandora text->OCEAN model (`trait_model.py`) is used only as a read-only readout
(`readout.py`) -- never to encode experiences or compute the push.

## Representation
- **Personality** (`personality.py`): `{"traits": {O,C,E,A,N in [-1,1]}, "experience_count"}`.
- **Experience** (`config.APPRAISAL_SCHEMA`): an appraisal vector --
  `valence, intensity, novelty, agency, social, outcome, self_relevance, threat_challenge`
  -- the causal ingredients of change, not a trait-expression reading.

## Push + update
```
salience = intensity*(0.5+0.5*self_relevance)*(0.5+0.5*novelty)
pull     = M . appraisal                          # which traits move, signed (config.M)
push[k]  = clamp(FORMATION_RATE * salience * pull[k])

trait[k] <- clamp(trait[k] + push[k] * (1 - |trait[k]|))   # diminishing returns
```
Worked example (a vivid party, push_E ~ 0.3):
`E: 0.00 -> 0.30 -> 0.51 -> 0.66 -> ...` -- each repetition adds less, bounded by 1.

Because the push is signed, experiences can lower a trait too: the `M` neuroticism
row `(-valence, -agency, -outcome, -threat_challenge, +intensity)` makes helpless
terror raise N while fear faced with mastery lowers it. (See `acceptance_test.py`.)

`FORMATION_RATE` is the responsiveness knob: ~0.3 per vivid experience is fast;
lower it for slower, more gradual formation.

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
