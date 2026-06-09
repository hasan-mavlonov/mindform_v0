# Handoff: MindForm Console (live personality cockpit)

> For: a coding agent (Codex / Claude Code) integrating this design into the
> existing repo **`hasan-mavlonov/stable_mind_v0.1`** (Django + Gemma).

---

## Overview

This is the **MindForm Console** — the operating surface for the MindForm
personality engine. A user talks to an agent ("Atlas") and watches its internal
state change live: **beliefs adapt fast** (within one exchange) while **traits
evolve slow** (only on accumulated evidence), with occasional **reflection**
moments where a trait consolidates. It replaces / upgrades the repo's existing
basic console pages (`ui/templates/ui/talk.html`, `persona.html`,
`interactive_test.html`) with a single cohesive cockpit.

The whole point of the design is to make the two-clock model **visible and
alive** — no raw JSON dumped in the UI. Everything is a bar, a meter, an orbiting
particle, or a tone shift in the agent's replies.

---

## About the design files

The files in `prototype/` are a **design reference built in HTML/React (Babel
in-browser)**. They are a working prototype of the intended look and behavior —
**not production code to paste into the repo verbatim.** Your job is to
**recreate this design inside the existing Django app** using its established
patterns (Django templates under `ui/templates/ui/`, the existing `base.html`
shell, the CSS-variable token system already defined there), and to **back the
live visuals with the real engine** in `core/` instead of the prototype's
client-side simulation.

Two integration paths — pick based on how much you want to change the stack:

- **A. Server-rendered + fetch (recommended, lowest friction):** keep Django
  templates. Render the cockpit shell in a new template, and drive the live
  panels with small JSON endpoints (`fetch`) that call the real engine. Vanilla
  JS or Alpine is enough; you do **not** need to adopt React.
- **B. Mounted React island:** if you prefer, mount the prototype's components as
  a React island inside a Django template and point them at the same endpoints.
  Drop the in-browser Babel — precompile.

Either way, **the prototype's `console/engine.jsx` is a SIMULATION**. In the repo,
belief/trait/perception/reflection values must come from the real Python engine
(`core/agent.py`, `core/perception.py`, `core/consolidation.py`,
`core/reflection.py`, `core/state_manager.py`) reading/writing
`persona/*.json` — exactly the structures `talk.py` already loads.

---

## Fidelity

**High-fidelity.** Colors, typography, spacing, and motion are final and match
the repo's existing MindForm design language (same tokens as
`ui/templates/ui/base.html` and `landing.html`). Recreate the UI pixel-accurately
using the repo's existing CSS tokens. The numeric *behavior* of the engine in the
prototype is illustrative — the real numbers come from `core/`.

---

## How it maps onto the existing repo

| Prototype piece | Real backing in the repo |
|---|---|
| Chat with Atlas | `talk.py` → `answer_user_message()` (read-only) **or** `core/agent.py` `Agent.step()` for the write path. Current view: `ui/views.py::talk_view`, `run_talk()` in `ui/services.py`. |
| Belief bars (fast loop) | `persona/stable.json` → `beliefs[entity_type][entity][dimension] = {mean, confidence, n}` (see `talk.py::extract_entity_beliefs`). Updated by `core/consolidation.py`. |
| Trait bars (slow loop) | `persona/stable.json` → `personality` (core traits) evolved by `core/reflection.py` (bounded). |
| Perception (sparks on send) | `core/perception.py` — Gemma-powered structured perception extraction. |
| Reflection banner | A reflection event emitted by `core/reflection.py` when traits actually update. |
| Mood meters (valence/energy/stress) | `persona/dynamic.json` → `now` / mood fields. |
| "Turn NNN" counter | `InteractionTurn` count for the active session (`ui/models.py`). |
| Reset agent | `create_persona.py::run_create_persona()` (already wired as `ui/views.py::create_persona_view`). |

### Suggested endpoints (path A)

Add to `ui/urls.py` / `ui/views.py` (return JSON, no HTML):

```
POST /api/turn/        body: {message}  -> {reply, perception, beliefs[], traits[], mood, turn, reflection?}
GET  /api/state/       -> {beliefs[], traits[], mood, turn}     # for first paint / polling
POST /api/reset/       -> {ok: true}                            # re-init persona
```

`/api/turn/` should: run perception → consolidation (fast) → bounded reflection
(slow) via `core/agent.py`, persist via `core/state_manager.py`, then return the
**new** state so the front-end can animate the bars to their new values. Keep the
JSON server-side only — render it as bars, never show it to the user.

**Data contract the front-end expects** (shape the prototype animates against):

```json
{
  "reply": "string — Atlas's in-character message",
  "turn": 14,
  "mood": { "label": "guarded", "valence": -0.25, "energy": 0.5, "stress": 0.62 },
  "perception": { "topic": "remote work", "stance": 0.7, "pressure": 0.8 },
  "beliefs": [
    { "id": "consensus", "label": "blind consensus",
      "mean": -0.72, "base": -0.72, "confidence": 0.74, "n": 9 }
  ],
  "traits": [
    { "key": "conviction", "label": "Conviction", "glyph": "resists drift",
      "value": 0.80, "base": 0.80 }
  ],
  "reflection": { "trait": "conviction", "label": "Conviction", "delta": 0.028 }
}
```

- `mean` ∈ [-1, 1] (belief stance), `confidence`/`value`/energy/stress ∈ [0, 1],
  `valence` ∈ [-1, 1].
- `base` = the origin value, drawn as a ghost marker so drift from the starting
  identity is visible. Snapshot it when the persona is created.
- `reflection` is present only on turns where a trait actually moved.

---

## Screens / Views

It is **one screen**: a full-viewport cockpit (no page scroll on desktop;
columns scroll internally). Header on top, then a 3-column grid.

### Layout
- Root: `display:flex; flex-direction:column; height:100dvh`.
- **Header** (fixed height ~62px): brand (`MindForm` + pulsing dot + `Console`
  chip) · centered status (`Agent Atlas` · `Turn 001` · animated mood pill) ·
  right actions (theme toggle, "Reset agent").
- **Grid**: `display:grid; grid-template-columns: 1.12fr 0.92fr 0.96fr; gap:16px;
  padding:16px 26px 22px; flex:1; min-height:0`. Three cards:

**Column 1 — Chat**
- Card header: `Atlas` (serif, 30px) + an italic tagline that changes with the
  dominant trait + a "persistent" live badge.
- Scrolling message list. Agent messages: serif 21px, left magenta border,
  mono uppercase "ATLAS" label. User messages: right-aligned, italic, muted,
  right hairline border. Typing indicator = 3 bouncing magenta dots.
- Composer (pinned bottom, surface-2 bg): horizontally-scrolling suggestion
  chips, then textarea + magenta Send button.

**Column 2 — The identity field (canvas orb)**
- `<canvas>` filling the card: a breathing nucleus (the self), belief particles
  orbiting (distance ∝ inverse confidence, color lerps violet→magenta with
  stance magnitude, size ∝ confidence), a violet trait halo whose radius rides
  average trait magnitude, and 6 outer trait spokes (length ∝ trait value, with
  a ghost tick at `base`). On each sent message, ~26 magenta **sparks** fly
  inward and pulse the core (perception arriving).
- Reflection banner animates in over the top when a trait consolidates.
- Below the orb: three **mood meters** — Valence (signed, honey), Energy
  (magenta), Stress (magenta).

**Column 3 — Beliefs + Traits**
- **Beliefs** block: title + `fast loop · experience` badge. Each bar is
  **center-zero** (zero line at 50%): magenta fill from center to `mean`, a ghost
  marker at `base`, value `+0.82 / −0.34` at right. Track opacity ∝ confidence.
  New beliefs animate in; updated belief flashes (glow + name turns magenta).
- **Traits** block: title + `slow loop · personality` badge. Left-anchored violet
  bars, ghost tick at `base`, mono glyph subtitle per trait. On reflection the
  matching bar pulses (ring + glow).

### Responsive
- ≤1180px: 2-column (chat spans top, orb + state below).
- ≤820px: single column, page scrolls, `body{overflow:auto}`.

---

## Interactions & Behavior
- **Send** (Enter or button): push user msg → spawn orb sparks (`pulse`) → call
  `/api/turn/` → append Atlas reply → animate bars/traits/mood to returned
  values → flash the touched belief → if `reflection` present, pulse that trait
  + show banner ~5.2s.
- **Suggestion chips**: prefilled provocations that send on click (disabled while
  busy).
- **Theme toggle**: flips `data-theme` on `<html>`, persists to
  `localStorage('theme')` (mirror the boot script already in `base.html`).
- **Reset agent**: clears conversation + re-inits persona (`run_create_persona`).
- **Typing state** while awaiting the model.
- **Persistence**: prototype stores conversation + state in
  `localStorage('mindform_console_v2')` so refresh keeps place. In the repo this
  should instead come from the persisted persona + the Django session's
  `evaluation_session` turns; keep client storage only as an optimistic cache.

### Motion (durations / easing)
- Belief fill: `left/width 0.65s cubic-bezier(.4,0,.2,1)`.
- Trait fill: `width 1.1s cubic-bezier(.4,0,.2,1)` (deliberately slower than
  beliefs — reinforces the two-clock idea).
- Mood meters: `width 0.6s`. Reflection banner: `0.4s` slide-fade in.
- Brand/mood pulse dots: `2–2.4s ease-in-out infinite`.
- Respect `prefers-reduced-motion: reduce` (prototype already neutralizes
  animation there).

---

## State Management
Front-end state (per the prototype): `messages[]`, `engine` (`{turn, traits[],
beliefs[], mood, ...}`), `input`, `busy`, `pulse` (spark trigger), `flashId`
(belief just updated), `reflection` (transient). In the repo, treat the server
response from `/api/turn/` as the source of truth and animate toward it; the
prototype's local `stepEngine()` math is **not** needed once the real engine
returns state.

---

## Design Tokens

These already exist in `ui/templates/ui/base.html` — **reuse them**, don't invent
new ones. Exact values (light → dark):

```
--bone     #FAF7F1 → #0B0617      (app background)
--bone-2   #F2EDE3 → #120B24      (inset / input bg)
--ink      #1A0E2E → #F2EAD6      (primary text)
--ink-2    #2E1F4A → #C9C0AA      (secondary text)
--muted    #8B7BA0 → #9B8DB0
--muted-2  #6A597F → #B5A8C7
--hairline rgba(26,14,46,.10) → rgba(242,234,214,.10)
--belief   #F0386B → #FF6F94      (fast loop / experience — magenta)
--trait    #5B3FE0 → #9A7FFF      (slow loop / personality — violet)
--warm     #F5B341                (valence / honey accent)
--surface  #ffffff → #14102A      (card bg)
--track    rgba(26,14,46,.07) → rgba(242,234,214,.08)  (bar tracks)
```

Type scale:
```
--serif "Instrument Serif", Georgia, serif   — headings, agent name, chat messages
--body  "Newsreader", Georgia, serif         — body, taglines (italic)
--mono  "JetBrains Mono", ui-monospace, Menlo — labels, badges, values, eyebrows
```
Sizes used: chat msg 21px serif / 1.32; agent name 30px serif; block titles 26px
serif; mono labels 9–12px, letter-spacing 0.12–0.22em, UPPERCASE; bars 9px tall,
3px radius.

Accent is **Tweakable** in the prototype (belief/trait palette pairs). For the
repo this can be a fixed brand choice — default `#F0386B` / `#5B3FE0`.

---

## Assets
None external. The orb is procedural `<canvas>` (no images). Favicon is the
inline SVG already in the repo templates. Fonts load from Google Fonts (same
`<link>` as `base.html`). The architecture SVG at
`assets/diagrams/mindform_architecture.svg` is not used by this screen.

---

## Files in this bundle
```
prototype/
  MindForm Console.html        # full cockpit shell + all CSS (design tokens live here)
  console/
    engine.jsx                 # SIMULATED engine — reference for the math/behavior only;
                               #   replace with calls to core/ in the repo
    mind.jsx                   # canvas orb renderer (port as-is; it's presentation only)
    app.jsx                    # layout, chat wiring, animation triggers, Tweaks
    tweaks-panel.jsx           # design-time accent/theme controls (optional in prod)
```
Open `MindForm Console.html` in a browser to see the live behavior. The chat in
the prototype calls a hosted model helper; in the repo, route it through the
existing Gemma path (`core/llm.py` → `talk.py` / `core/agent.py`).

## Build / integration notes
- Drop the in-browser Babel transformer for production (path B only).
- Keep CSS in the shared template token system; don't fork the palette.
- The orb reads its accent colors from CSS vars / props — keep that so theme +
  accent stay in sync.
- Honor the existing `talk.py` contract if you want a safe read-only mode: it
  never writes state. The cockpit's *fast/slow* movement requires the write path
  (`Agent.step()` + consolidation + reflection).
```
