# MindForm Architecture

## What it does
A piece of text (an experience) changes a personality made of five OCEAN traits,
each in **[-1, 1]**. Experiences push the traits, and the push is applied with
**diminishing returns**: a trait moves fast while near 0 and ever more slowly as it
approaches +/-1.

```
text -> MiniLM embedding (encoder.py)                  # recurrence + memory
     -> signed push:
          DeepSeek LLM  (llm_impact.py): text -> OCEAN delta x LLM_FORMATION_RATE  [primary]
          heuristic     (appraisal.py -> impact.py): appraisal vector -> push      [fallback]
     -> diminishing-returns update of the five traits (updater.py)
     -> persist + memory   (personality.py, memory.py)
```

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

## LLM push (DeepSeek) + heuristic fallback
The primary push comes from DeepSeek (`llm_impact.py`): the model reads the experience
and returns a signed OCEAN delta in [-1, 1] -- the pressure of a **single** occurrence --
and `push = clamp(LLM_FORMATION_RATE * delta)`. Repetition is not baked into the delta;
the engine accumulates repeated experiences and the diminishing-returns update supplies
the asymptote. DeepSeek is OpenAI-compatible (cheap, China-reachable), so any compatible
endpoint works via `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL`.

Copy `.env.example` to `.env` and set `DEEPSEEK_API_KEY`. If the key is missing, the
`openai` package is absent, the network fails, or the reply is unparseable,
`push_from_text` falls back to the deterministic heuristic `impact(appraise(text))` --
so there is no hard network dependency and the dependency-free `acceptance_test.py`
path is unchanged.

## Deterministic vs learned -- every component is replaceable
| Component | Today | Later (same interface) |
|---|---|---|
| Encoder | MiniLM (frozen) | — |
| Push source | DeepSeek LLM (primary), heuristic fallback | learned `appraisal -> push` |
| Appraisal extractor | heuristic -> head trained on affect corpora | larger/fine-tuned head |
| Push matrix `M` | theory rules | learned `appraisal -> push` |

## Data (no longitudinal dataset required)
- Appraisal head: existing cross-sectional affect/appraisal corpora (EmoBank VAD,
  GoEmotions, appraisal-annotated event sets) via `bootstrap/`.
- No `experience -> trait-change` labels exist anywhere, so `M` stays rules until
  such data does.

## Run
```
python acceptance_test.py                                 # dependency-free behaviour check
pip install -r requirements.txt                           # encoder + DeepSeek client
cp .env.example .env                                      # set DEEPSEEK_API_KEY (optional; heuristic runs without it)
python simulation.py                                      # full pipeline (encoder in the loop)
python bootstrap/build_affect_dataset.py && python bootstrap/train_appraisal_head.py  # train head (local)
```
