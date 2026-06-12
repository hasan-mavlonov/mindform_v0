# MindForm Console — the web UI

A browser cockpit for the MindForm personality engine. Pick or create a
character, then talk to it: every message is an **experience** that nudges its
five **OCEAN** traits, and you watch them move live.

```
python console.py            # then open http://127.0.0.1:8000/
PORT=9000 python console.py  # choose a port
```

No build step, no extra dependencies — it runs on the Python standard library and
plain JavaScript. With a `GEMINI_API_KEY` in `.env` the pushes and the
character's replies come from the LLM (Google Gemma 4 by default); without one,
everything still works on the engine's heuristic fallback (you just won't get
LLM-quality prose).

> The personality engine is **not modified** by any of this. This package only
> *calls* the existing modules (the same ones `interactive.py` uses) and draws the
> result. All trait math stays in `config.py`, `temperament.py`, `appraisal.py`,
> `llm_impact.py`, `impact.py`, `updater.py`, `personality.py`, `memory.py`.

## What you see

It is one screen.

- **Start screen** — *Use existing* (your roster), *From a story* (`genesis`: a
  one-line bio seeds the OCEAN baseline), or *Build one* (identity fields + a 1–5
  slider per trait → `build_character`). These are the same three authoring paths
  as the interactive shell.
- **Chat** — talk to the character. Each message is an experience.
- **The push** (magenta, *this experience*) — the signed pressure the last message
  put on each trait, straight from `llm_impact.push_from_text`.
- **Personality** (violet, *who they are · OCEAN*) — the five current traits in
  [-1, 1], center-zero, with a ghost tick at each **baseline** (`temperament.mu`).
  They ease to new values every turn — the thing you came to watch.
- **The identity field** — a canvas orb: five trait spokes (ghost tick at
  baseline), a nucleus that breathes with Extraversion and runs hot with
  Neuroticism, and sparks that fly in when a message arrives.
- **Mood meters** — how the last message *read* (the appraisal: valence /
  intensity / novelty).
- **Formation banner** — names the single biggest trait move each turn in plain
  language ("grew more outgoing, energetic").

## How it maps onto the engine

The original UI design was authored for a different repo (Django + Gemma, with a
"beliefs" fast-loop). This repo has no beliefs and one set of dynamics: an
experience produces a signed push, applied to the OCEAN traits with diminishing
returns, anchored by a per-trait baseline. So the design's two loops are retold
truthfully as **the push (immediate) → the personality (accumulated)** — same
colors, same motion language, real numbers.

| Design piece | Real backing here |
|---|---|
| Chat reply | `web/reply.py` — the LLM (Gemma by default) if configured, trait-driven fallback otherwise (presentation only; never feeds the trait math) |
| "Beliefs" bars (fast, magenta) | **The push** — `llm_impact.push_from_text(text)` |
| "Traits" bars (slow, violet) | **OCEAN** — `personality["traits"]`, ghost tick at `temperament.mu`, moved by `updater.update_personality` |
| Orb spokes / particles | the five OCEAN traits (value + baseline) |
| Mood meters | the appraisal of the last message (`appraisal.appraise`) |
| Reflection banner | the biggest actual trait move this turn |
| Reset / switch | the start-screen picker (`personality.list_characters`) |
| "Turn NNN" | `experience_count` |

## Files

```
console.py               # repo-root launcher (python console.py)
web/
  server.py              # stdlib http.server: static files + JSON API
  engine_bridge.py       # orchestrates the unchanged engine into UI snapshots
  reply.py               # in-character chat reply (DeepSeek optional)
  static/
    index.html           # start screen + cockpit shell (design tokens)
    css/console.css       # styles, adapted from the handoff (center-zero bars)
    js/api.js             # fetch wrappers
    js/orb.js             # canvas orb (procedural, vanilla JS)
    js/console.js         # app logic: picker, cockpit, animation triggers
```

## API (all backed by the real engine)

```
GET  /api/config                      basis, trait questions, identity fields, levels
GET  /api/characters                  roster for the picker
GET  /api/state?name=                 first-paint snapshot
POST /api/select        {name}        open a character
POST /api/create/genesis {bio}        born from a biography
POST /api/create/manual  {identity, levels}   identity + 1–5 per trait
POST /api/turn          {name, message}        one experience -> new state + reply
```

A turn runs exactly `interactive.py`'s pipeline — `appraise → push_from_text →
update_personality → (encoder/memory, if installed) → save_character` — then
returns the new state so the front-end animates the bars to it.

## Notes

- Characters persist to `data/characters/<slug>.json` (and `*.memories.json`),
  the same store the CLI uses. That directory is per-machine runtime data; you
  generally don't want to commit it.
- `sentence-transformers` (encoder) and `numpy` (memory recurrence) are optional:
  if they're not installed, formation still works — the UI just skips the "seen
  before" count and the memory log that turn.
- Theme + the procedural orb read their colors from the shared CSS tokens, so
  light/dark stays in sync.
