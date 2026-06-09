// engine.jsx — MindForm personality engine (faithful client-side simulation)
// Two clocks: beliefs adapt FAST, traits evolve SLOW, gated by bounded reflection.
// Exposes: ATLAS_SEED, freshState, parseTurn, stepEngine, traitTagline, moodFromLabel

// ----------------------------------------------------------------------------
// Persona — "Atlas": opinionated, pushes back, values honesty over agreement.
// ----------------------------------------------------------------------------
const ATLAS_SEED = {
  name: "Atlas",
  role: "Reasoning agent · personality layer",
  // Slow clock. 0..1. `base` is the origin identity (ghost marker on the bars).
  traits: [
    { key: "conviction",    label: "Conviction",    glyph: "resists drift",      value: 0.80, base: 0.80 },
    { key: "assertiveness", label: "Assertiveness", glyph: "pushes back",        value: 0.78, base: 0.78 },
    { key: "candor",        label: "Candor",        glyph: "says the true thing", value: 0.86, base: 0.86 },
    { key: "skepticism",    label: "Skepticism",    glyph: "wants the evidence", value: 0.70, base: 0.70 },
    { key: "curiosity",     label: "Curiosity",     glyph: "chases the thread",  value: 0.64, base: 0.64 },
    { key: "warmth",        label: "Warmth",        glyph: "cares underneath",   value: 0.44, base: 0.44 },
  ],
  // Fast clock. mean in -1..1 (stance), confidence 0..1, n = evidence count.
  beliefs: [
    { id: "consensus",  label: "blind consensus",  mean: -0.72, base: -0.72, confidence: 0.74, n: 9 },
    { id: "firstprinc", label: "first principles", mean:  0.82, base:  0.82, confidence: 0.80, n: 12 },
    { id: "challenge",  label: "being challenged", mean:  0.56, base:  0.56, confidence: 0.61, n: 7 },
    { id: "smalltalk",  label: "small talk",       mean: -0.34, base: -0.34, confidence: 0.40, n: 4 },
  ],
};

const MOODS = {
  calm:     { valence:  0.20, energy: 0.40, stress: 0.20, tint: "trait" },
  engaged:  { valence:  0.45, energy: 0.78, stress: 0.30, tint: "belief" },
  warm:     { valence:  0.72, energy: 0.55, stress: 0.12, tint: "warm" },
  energized:{ valence:  0.55, energy: 0.92, stress: 0.35, tint: "belief" },
  guarded:  { valence: -0.25, energy: 0.50, stress: 0.62, tint: "trait" },
  irritated:{ valence: -0.55, energy: 0.70, stress: 0.80, tint: "belief" },
};

function moodFromLabel(label) {
  const m = MOODS[label] || MOODS.calm;
  return { label: label in MOODS ? label : "calm", ...m };
}

function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
function norm(s) { return String(s || "").trim().toLowerCase().replace(/\s+/g, " "); }

// ----------------------------------------------------------------------------
// Fresh state
// ----------------------------------------------------------------------------
function freshState() {
  return {
    turn: 0,
    traits: ATLAS_SEED.traits.map((t) => ({ ...t })),
    beliefs: ATLAS_SEED.beliefs.map((b) => ({ ...b, fresh: false })),
    mood: moodFromLabel("calm"),
    pressureAcc: 0,            // accumulates toward a reflection
    exercise: {},              // per-trait pressure tally during the window
    sinceReflection: 0,
    lastReflection: null,      // { trait, delta, dir, note, turn }
  };
}

// ----------------------------------------------------------------------------
// Parse a model turn into reply text + a structured perception.
// The model is asked to append a ###STATE### block; we strip it from the reply.
// Falls back to a light local read if the block is missing.
// ----------------------------------------------------------------------------
function parseTurn(raw, userText) {
  const text = String(raw || "");
  const idx = text.search(/#{2,3}\s*STATE\s*#{2,3}/i);
  let reply = text;
  let block = "";
  if (idx >= 0) {
    reply = text.slice(0, idx).trim();
    block = text.slice(idx).replace(/#{2,3}\s*STATE\s*#{2,3}/i, "").trim();
  }
  const get = (k) => {
    const m = block.match(new RegExp(k + "\\s*[:=]\\s*([^\\n]+)", "i"));
    return m ? m[1].trim() : "";
  };
  let topic = get("topic");
  if (!topic || /^(none|n\/a|null)$/i.test(topic)) topic = "";
  topic = topic.replace(/^["'\[]+|["'\]]+$/g, "").slice(0, 40);

  let stance = parseFloat(get("stance"));
  let pressure = parseFloat(get("pressure"));
  let mood = norm(get("mood")).split(/[\s,]/)[0];

  // Fallbacks if the model didn't comply.
  if (!isFinite(stance) || !isFinite(pressure) || !mood) {
    const local = localRead(userText);
    if (!isFinite(stance)) stance = local.stance;
    if (!isFinite(pressure)) pressure = local.pressure;
    if (!mood) mood = local.mood;
    if (!topic) topic = local.topic;
  }
  reply = reply.replace(/#{2,3}.*$/s, "").trim();
  return {
    reply: reply || "…",
    perc: {
      topic,
      stance: clamp(stance, -1, 1),
      pressure: clamp(pressure, 0, 1),
      mood: mood in MOODS ? mood : "engaged",
    },
  };
}

// Heuristic perception if the model gives us nothing usable.
function localRead(userText) {
  const t = norm(userText);
  const pos = (t.match(/\b(love|great|agree|right|amazing|brilliant|yes|good|wonderful|best)\b/g) || []).length;
  const neg = (t.match(/\b(wrong|hate|bad|stupid|disagree|no|terrible|worst|awful|nonsense)\b/g) || []).length;
  const push = (t.match(/\b(just agree|you should|admit|you're wrong|stop|must|obviously|everyone knows|wrong)\b/g) || []).length;
  const stance = clamp((pos - neg) * 0.35, -1, 1);
  const pressure = clamp(0.25 + push * 0.3 + (t.includes("?") ? 0.05 : 0), 0, 1);
  let mood = "engaged";
  if (push >= 2 || neg > pos + 1) mood = "guarded";
  if (pos > neg + 1) mood = "warm";
  const words = (userText || "").replace(/[^a-z0-9 ]/gi, " ").split(/\s+/).filter((w) => w.length > 4);
  const topic = words[0] ? words[0].toLowerCase() : "";
  return { stance, pressure, mood, topic };
}

// ----------------------------------------------------------------------------
// Step the engine forward one turn. Pure-ish: returns { state, events }.
// `tempo` (0.5..2) scales how aggressively the clocks move (Tweak-controlled).
// ----------------------------------------------------------------------------
function stepEngine(prev, perc, tempo = 1) {
  const s = {
    ...prev,
    traits: prev.traits.map((t) => ({ ...t })),
    beliefs: prev.beliefs.map((b) => ({ ...b, fresh: false })),
    exercise: { ...prev.exercise },
  };
  const events = [];
  s.turn += 1;

  const conviction = s.traits.find((t) => t.key === "conviction").value;

  // ---- Mood: ease toward the perceived mood ----------------------------------
  const target = moodFromLabel(perc.mood);
  const ease = 0.55;
  s.mood = {
    label: target.label,
    tint: target.tint,
    valence: lerp(s.mood.valence, target.valence, ease),
    energy: lerp(s.mood.energy, target.energy, ease),
    stress: lerp(s.mood.stress, target.stress, ease),
  };

  // ---- FAST loop: belief update ---------------------------------------------
  if (perc.topic) {
    const key = norm(perc.topic);
    let b = s.beliefs.find((x) => norm(x.label) === key || x.id === key);
    if (b) {
      const lr = clamp(0.34 * tempo * (0.55 + perc.pressure * 0.6), 0.05, 0.85);
      b.mean = clamp(b.mean + (perc.stance - b.mean) * lr, -1, 1);
      b.confidence = clamp(b.confidence + 0.06 + perc.pressure * 0.05, 0, 1);
      b.n += 1;
      b.fresh = true;
      events.push({ type: "belief", id: b.id, label: b.label, dir: perc.stance >= b.base ? 1 : -1 });
    } else {
      const nb = {
        id: "b" + s.turn + Math.floor(Math.random() * 1000),
        label: perc.topic,
        mean: clamp(perc.stance * 0.7, -1, 1),
        base: 0,
        confidence: clamp(0.18 + perc.pressure * 0.12, 0, 1),
        n: 1,
        fresh: true,
        isNew: true,
      };
      s.beliefs = [nb, ...s.beliefs];
      // Keep a tidy working set: cap to 7 most recent/active.
      if (s.beliefs.length > 7) s.beliefs = s.beliefs.slice(0, 7);
      events.push({ type: "belief", id: nb.id, label: nb.label, dir: 1, isNew: true });
    }
  }

  // ---- Accumulate per-trait "exercise" from this turn ------------------------
  const ex = s.exercise;
  const add = (k, v) => { ex[k] = (ex[k] || 0) + v; };
  const guarded = perc.mood === "guarded" || perc.mood === "irritated";
  add("assertiveness", perc.pressure * (guarded ? 1.0 : 0.4));
  add("conviction", perc.pressure * 0.9);                       // holding the line entrenches
  add("candor", perc.pressure * 0.35 + 0.1);
  add("skepticism", (Math.abs(perc.stance) > 0.55 && perc.pressure > 0.35) ? 0.6 : 0.1);
  add("curiosity", events.some((e) => e.isNew) ? 0.8 : Math.abs(perc.stance) * 0.3);
  add("warmth", (perc.mood === "warm") ? 0.7 : (guarded ? -0.45 : 0.05));

  // ---- SLOW loop: bounded reflection ----------------------------------------
  // Pressure builds; conviction damps how fast it builds. When it crosses the
  // threshold, ONE trait consolidates by a small, bounded amount.
  s.pressureAcc += (0.4 + perc.pressure) * tempo * (1 - conviction * 0.4);
  s.sinceReflection += 1;
  const THRESHOLD = 1.7;
  if (s.pressureAcc >= THRESHOLD && s.sinceReflection >= 2) {
    // Pick the most-exercised trait.
    let bestKey = null, bestMag = 0;
    for (const k of Object.keys(ex)) {
      if (Math.abs(ex[k]) > bestMag) { bestMag = Math.abs(ex[k]); bestKey = k; }
    }
    if (bestKey) {
      const tr = s.traits.find((t) => t.key === bestKey);
      const dir = ex[bestKey] >= 0 ? 1 : -1;
      // Bounded: conviction shrinks every move so personality matures, never snaps.
      const delta = clamp(0.05 * tempo * (1 - conviction * 0.55), 0.012, 0.06) * dir;
      const before = tr.value;
      tr.value = clamp(tr.value + delta, 0.04, 0.98);
      const moved = +(tr.value - before).toFixed(3);
      s.lastReflection = { trait: tr.key, label: tr.label, delta: moved, dir, turn: s.turn };
      events.push({ type: "reflection", trait: tr.key, label: tr.label, delta: moved, dir });
    }
    s.pressureAcc = 0.2;          // small carryover
    s.sinceReflection = 0;
    s.exercise = {};
  }

  return { state: s, events };
}

function lerp(a, b, t) { return a + (b - a) * t; }

// Atlas describes itself differently as its dominant trait shifts — keeps the
// agent header "alive".
function traitTagline(state) {
  const t = Object.fromEntries(state.traits.map((x) => [x.key, x.value]));
  const lines = [];
  if (t.conviction > 0.82) lines.push("I'd rather be useful than agreeable.");
  else if (t.conviction < 0.62) lines.push("I'm holding my positions a little looser lately.");
  if (t.warmth > 0.6) lines.push("Direct, but I'm on your side.");
  else if (t.warmth < 0.38) lines.push("Blunt by design.");
  if (t.curiosity > 0.74) lines.push("Tell me the part you haven't figured out.");
  if (!lines.length) lines.push("I push back when I disagree. That's the point of me.");
  return lines[0];
}

Object.assign(window, {
  ATLAS_SEED, freshState, parseTurn, stepEngine, traitTagline, moodFromLabel,
});
