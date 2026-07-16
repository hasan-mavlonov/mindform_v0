# MindForm Architecture

## What it does
A character is **born** from a short biography (genesis): immutable identity facts plus a
temperamental OCEAN baseline. Every experience you tell it then **forms** the whole person:
traits, values, moral outlook, beliefs, habits, needs, a self-image, a manner of speaking,
and a behavioral stance -- each its own node, each visible live in the cockpit, each moving
by bounded dynamics (pushes with **diminishing returns**, pulls back to set-points), so
formation is fast near zero and asymptotes at the extremes without ever running away.

The engine is **LLM-primary, offline-complete**: with a key set, an OpenAI-compatible model
(Gemini 3.5 Flash via the Gemini API by default) reads each experience; with no key, no
`openai` package, or no network, every node falls back to deterministic heuristics and the
whole product still runs -- including the UI.

## One turn, end to end (`web/engine_bridge.run_turn`)
```
text -> REFRESH the carried state (each x <- x + rate*(set_point - x)):
          drives      needs regenerate toward their value-seeded weights
          self        esteem relaxes toward its (aspiration-depressed) baseline
          behavior    sensitivities relax toward trait-anchored set-points
          expression  the formed manner relaxes toward what the inner state calls for

     -> RECALL (memory.py; needs numpy + encoder, skipped cleanly without):
          cosine top-k over the embedding sidecar, then MOTIVATED RETRIEVAL --
          the active needs re-rank the pool (drives.recall_bias): what you lack
          shapes what you remember

     -> APPRAISE (perception): llm_appraisal.appraise_from_text
          LLM 8-dim appraisal [primary] -> trained head -> lexicon   [fallback chain]
          every LLM-labelled reading is logged as training data (appraisal_log) --
          the offline head learns from the online model with use (distillation)

     -> INTERPRET (cognition.interpret) -- the lens, five tilts in order:
          behavior gate   the carried stance scales INTENSITY (lean in = life lands harder)
          trait tilt      anxious -> more threat, darker valence; open -> more novelty
          memory tilt     recalled episodes pull valence/threat toward how they felt
          drive tilt      events bearing on loud needs read more relevant/warm/threatening
          self tilt       self-discrepant events read as threat; esteem buffers; the
                          ought-gap adds vigilance (agitation)

     -> PUSH x3 (traits / values / moral): LLM delta x LLM_FORMATION_RATE [primary]
          (the prompt carries cognition.lens() -- the same tilts as words), or
          impact(appraisal) x M / VALUES_M / MORAL_M [fallback]; the LLM pushes are
          mirrored by the realized intake ratio so both paths land identically gated

     -> UPDATE: traits (diminishing returns + temperament pull/drift), values, moral,
          habits (recurrence), beliefs (LLM extraction, offline-deferred)

     -> STORE memory: the INTERPRETED appraisal + the stance it was met with

     -> APPLY the events back onto the carried state:
          drives      satisfaction/frustration (frustration bites harder)
          self        sociometer esteem + Bem/Swann self-image drift
          behavior    operant credit (rewarded OWN action; threat trains inhibition
                      2x faster than mastery extinguishes) + reception of the last act
          expression  the manner actually spoken with is entrenched/extinguished by
                      how the reply landed (reception shaping)

     -> NOTE the act (stance + mode + style frozen for next turn's credit) -> SAVE
     -> REPLY (reply.py): speaks from the SELF-VIEW in the formed voice, sees the
          interpreted appraisal; deterministic styled compositor offline
     -> SNAPSHOT for the cockpit
```
The reply's **text** never re-enters formation; what feeds forward is the deterministic
stance recorded before the reply, resolved against the *next* user message as its outcome.

## The layers (all in one `personality` dict, all persisted as JSON)
| Layer | What it is | Dynamics | Anchor / set-point |
|---|---|---|---|
| `identity` | immutable facts | never moves | — |
| `temperament` (mu, tau) | innate OCEAN baseline | mu drifts only under sustained change | — |
| `traits` (OCEAN) | who they are now | push + diminishing returns | pulled toward `mu` by `tau` |
| `character.values` (Schwartz 10) | what they prize | same push dynamics, from 0 | none -- earned |
| `character.moral` (Haidt 6) | what's right/wrong to them | same | none -- earned |
| `character.beliefs` | open propositions held | LLM-extracted, conviction accumulates | none |
| `character.habits` | recurring experiences | recurrence count threshold | none |
| `drives` (SDT 3) | need tensions: how starved | fast satisfy/frustrate (asymmetric) | weights seeded from values |
| `self.image` (OCEAN) | who they THINK they are | Bem drift toward traits, Swann-damped | lags reality by design |
| `self.esteem` | self-regard | sociometer (rides the needs) | dispositional base − aspiration gap |
| `expression.style` | the formed manner | reception shaping (operant) | derived target from self-view |
| `behavior` (approach/inhibition) | the enacted stance | operant credit from outcomes | trait-anchored set-points |

Derived each turn, never stored: drive weights, esteem baselines, the ideal/ought selves
(Higgins: values -> who they want to be, moral -> who they should be; falling short
depresses regard and adds vigilance), the style target, behavior set-points, the intake
gate. Read-outs cannot fall out of sync with their sources.

## Perception -> formation (the numbers)
```
appraisal = {valence, intensity, novelty, agency, social, outcome,
             self_relevance, threat_challenge}          # threat_challenge: -1 threat, +1 challenge
salience  = intensity*(0.5+0.5*self_relevance)*(0.5+0.5*novelty)
push[k]   = clamp(FORMATION_RATE * salience * (M . appraisal)[k])
trait[k] <- clamp(trait[k] + push[k]*(1-|trait[k]|))    # diminishing returns
x[k]     <- x[k] + tau[k]*(mu[k]-x[k]);  mu[k] <- mu[k] + DRIFT*(x[k]-mu[k])
```
A vivid party moves E `0.00 -> 0.30 -> 0.51 -> 0.66 ...`; helpless terror raises N while
fear faced with mastery lowers it (`M`'s N row is signed). Values and moral share the same
update through `VALUES_M` / `MORAL_M`. Every feedback loop added on top (perceive -> form ->
perceive; behavior's intake gate; expression's reception shaping) is bounded by the same two
forces -- clamped small gains and pull-to-set-point -- with stability arguments and tests per
node.

## Perception distillation (the offline appraiser learns with use)
The dataset problem is solved by manufacture: the online model is the teacher.
`appraise_from_text` (LLM) serves the live reading and logs `{"text", "appraisal"}` rows to
`data/appraisal_dataset.jsonl`; `bootstrap/distill_appraisal_corpus.py` batch-writes and
labels synthetic experiences (15 domains x 7 tones) into the same file;
`bootstrap/train_appraisal_head.py` fits the MiniLM-embedding head; `core.appraisal`
automatically prefers the trained head offline. **The iron rule:** only LLM-labelled rows
enter the log -- the head never trains on its own (or the lexicon's) outputs.

## Deterministic vs learned -- every component is replaceable
| Component | Today | Later (same interface) |
|---|---|---|
| Appraisal | LLM [primary] -> trained head -> lexicon | head keeps improving via distillation |
| Push matrices `M`/`VALUES_M`/`MORAL_M` | theory rules (fallback path) | learned from the same distillation log |
| Drive/self/behavior/style priors | theory-authored (SDT, sociometer, BIS/BAS, circumplex) | learned |
| Temperament seed | LLM genesis (lexicon fallback) | learned `bio -> (mu, tau)` |
| Temperament math | point estimates | per-trait distributions |

## Run
```
python console.py                        # the cockpit at http://localhost:8000 -- zero deps, works offline
cp .env.example .env                     # set GEMINI_API_KEY for the LLM-primary path (optional)
pip install -r requirements.txt          # optional: encoder + memory/recall (numpy, sentence-transformers)

for t in acceptance cognition character genesis drives self_concept expression memory behavior appraisal_distill; \
  do python tests/${t}_test.py; done     # ten suites; all but memory are dependency-free

python bootstrap/distill_appraisal_corpus.py 1000   # manufacture appraisal training data (needs key)
python bootstrap/train_appraisal_head.py            # train the offline head (needs torch)
python interactive.py                    # terminal shell; genesis.py / simulation.py: CLI entry points
```

## Layout
```
core/    shared kernel: config (all knobs + priors), llm, encoder, appraisal (+_head, _log),
         impact, updater, memory, personality
nodes/   the faculties: temperament, llm_impact, llm_appraisal, cognition, character,
         values, moral, beliefs, drives, self_concept, expression, behavior
web/     the cockpit:   server (stdlib), engine_bridge (run_turn + snapshot), reply, static/
tests/   ten behaviour suites, dependency-free by default
root     entry scripts: console.py, interactive.py, simulation.py, genesis.py
```
Modules import across packages by path (`from core.config import ...`); entry scripts stay
at the root. The split is by *layer* (shared kernel vs. faculties), and each faculty is one
file with the same shape: seed/defaults, `refresh`, an interpret-tilt or push, `apply_event`,
read-outs -- so the diagram box and the module map one-to-one.
