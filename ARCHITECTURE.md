# MindForm Architecture

## What it does
A character is **born** from a short biography (genesis): it gets immutable identity
facts plus a temperamental OCEAN baseline. Then a piece of text (an experience)
changes its personality -- five OCEAN traits, each in **[-1, 1]** -- by pushing the
traits with **diminishing returns**: a trait moves fast while near 0 and ever more
slowly as it approaches +/-1.

```
bio  -> genesis (temperament.py): identity + OCEAN baseline mu + stickiness tau,
        born at baseline (traits x = mu)

text -> MiniLM embedding (encoder.py)                  # recurrence + memory
     -> signed push:
          DeepSeek LLM  (llm_impact.py): text -> OCEAN delta x LLM_FORMATION_RATE  [primary]
          heuristic     (appraisal.py -> impact.py): appraisal vector -> push      [fallback]
     -> diminishing-returns update of the five traits (updater.py)
     -> CHARACTER: the same experience pushes the ten Schwartz values
          (values.py: LLM delta x LLM_FORMATION_RATE [primary]; impact + VALUES_M [fallback])
          -> same diminishing-returns update (character.py); recurring experiences
          settle into habits (memory.recurrence -> character.note_habit)
     -> persist + memory   (personality.py, memory.py)
```

## Representation
- **Personality** (`personality.py`): `{"identity", "temperament": {"mu","tau"}, "traits": {O,C,E,A,N in [-1,1]}, "character": {"values","habits"}, "experience_count"}`.
- **Temperament** (`temperament.py`): per-trait baseline `mu in [-1,1]` and stickiness
  `tau in [0,1]`, seeded at genesis; `identity` holds immutable facts (name, origin,
  religion-raised, ...). The current `traits` are born at the baseline (x = mu).
- **Character** (`character.py`): what a person *becomes* from experience -- the ten
  Schwartz **values** (`config.VALUES`, each signed [-1,1]) plus the **habits** they
  fall into. Unlike temperament, values are **not** innate: they start at 0 and form.
- **Experience** (`config.APPRAISAL_SCHEMA`): an appraisal vector --
  `valence, intensity, novelty, agency, social, outcome, self_relevance, threat_challenge`
  -- the causal ingredients of change, not a trait-expression reading.

## Temperament & genesis (the baseline you're born with)
A character is born from a biography via `temperament.genesis(bio)`, which seeds:
- `identity` -- immutable facts (name, origin, religion-raised, ...): the lens, never drifts.
- `temperament.mu` -- the per-trait OCEAN **baseline** (the biological set-point).
- `temperament.tau` -- per-trait **stickiness** in [0, 1]: how hard biology anchors the trait.

The current `traits` start AT the baseline (`x = mu`), so two different bios yield two
distinguishable characters from birth instead of identical blank slates. Genesis uses
DeepSeek with the same heuristic fallback discipline as the push, so it runs with no network.

Three authoring paths build a character:
- `genesis(bio)` -- a one-line free-text biography; LLM/heuristic seeds `mu`/`tau`.
- `create_character(fields)` -- explicit identity fields + a free-text `background` that seeds `mu`/`tau`.
- `build_character(identity, mu)` -- explicit identity + an explicitly chosen OCEAN baseline,
  no LLM. This backs the short **questionnaire** the interactive shell uses
  (`config.TRAIT_QUESTIONS`: one 1-5 question per trait -> baseline `mu`).

Characters live in a roster (`data/characters/<slug>.json`, keyed by name; per-character
memories alongside as `<slug>.memories.json`). `interactive.py` opens with **Use existing**
(lists the roster, pick one) or **Create new** (identity + trait questionnaire), then drops
into the talk loop, autosaving the active character each turn.

**Slice 1 (today)** only *seeds* the baseline. **Slice 2** turns on the temperament
dynamics in the update -- the current trait pulled back toward its baseline, and the
baseline drifting slowly after a sustained shift:
```
x[k]  <- x[k] + tau[k] * (mu[k] - x[k])   # prior pulls current toward baseline
mu[k] <- mu[k] + eta    * (x[k] - mu[k])  # baseline drifts slowly (eta << tau)
```
With the pull active, repeated experience settles a trait at a fixed point *between* its
baseline and the extreme -- so identical lives with different temperaments end up as
different stable people. (Bayesian point-estimate now; per-trait distributions later.)

## Character (what you become from experience)
Temperament is the baseline you're *born* with; **character** is what you grow into by
living. Where the OCEAN traits describe *how* a person is, the Schwartz **values**
describe *what they prize* -- and they form by the very same dynamics:

```
text -> Schwartz delta (values.py)        # LLM [primary] / impact + VALUES_M [fallback]
push[v]    = clamp(LLM_FORMATION_RATE * delta[v])           # one occurrence's pressure
value[v]  <- clamp(value[v] + push[v] * (1 - |value[v]|))   # same diminishing returns
```
The ten values (`SD, ST, HE, AC, PO, SE, CO, TR, BE, UN`) trade off against one another,
so one experience can raise some and lower others (quitting a stable job to travel raises
Stimulation/Self-Direction while lowering Security). `character.higher_order` rolls them up
onto Schwartz's four poles (openness-to-change, self-enhancement, conservation,
self-transcendence) for plain-language read-outs. Values start at **0** -- earned, not
innate -- which is exactly what separates character (experience) from temperament (biology).

**Habits** are the behavioral residue of repetition: when an experience recurs
(`memory.recurrence`) past `config.HABIT_MIN_RECURRENCE`, `character.note_habit` records
it. The trait push and the values push share one appraisal of the experience and one
update rule (`updater.apply_diminishing`); they are kept as separate nodes so each can be
swapped independently. (See `character_test.py`.)

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
| Push source | LLM (primary), heuristic fallback | learned `appraisal -> push` |
| Values push | LLM Schwartz delta (primary), `impact + VALUES_M` fallback | learned `appraisal -> values` |
| Values matrix `VALUES_M` | theory rules | learned `appraisal -> values` |
| Appraisal extractor | heuristic -> head trained on affect corpora | larger/fine-tuned head |
| Push matrix `M` | theory rules | learned `appraisal -> push` |
| Temperament seed | DeepSeek genesis (heuristic fallback) | learned `bio -> (mu, tau)` |
| Temperament math | point estimate (mu, tau) | per-trait distributions (mu, sigma) |

## Data (no longitudinal dataset required)
- Appraisal head: existing cross-sectional affect/appraisal corpora (EmoBank VAD,
  GoEmotions, appraisal-annotated event sets) via `bootstrap/`.
- No `experience -> trait-change` labels exist anywhere, so `M` stays rules until
  such data does.

## Run
```
python acceptance_test.py                                 # dependency-free: experience -> trait change
python genesis_test.py                                    # dependency-free: bio -> distinct temperament
python character_test.py                                  # dependency-free: experience -> values + habits
pip install -r requirements.txt                           # encoder + DeepSeek client
cp .env.example .env                                      # set DEEPSEEK_API_KEY (optional; heuristic runs without it)
python genesis.py "Aisha, an anxious, creative, sheltered poet."   # birth a character (CLI)
python interactive.py                                     # roster menu (use existing / create new) + talk
python simulation.py                                      # full pipeline (encoder in the loop)
python bootstrap/build_affect_dataset.py && python bootstrap/train_appraisal_head.py  # train head (local)
```
