// console.js -- MindForm Console app logic (vanilla JS, no framework).
//
// Two views share one page: a START screen (pick or create a character) and the
// COCKPIT (talk to them and watch their OCEAN traits form). All numbers come from
// the real engine via /api; this file only renders them and drives the motion.

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const elc = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };

  // Experiences (not chit-chat): each shapes a different trait. Mirrors the
  // kinds of events the engine's push prompt is tuned on.
  const SUGGESTIONS = [
    "I went to a loud party and danced with strangers all night.",
    "I failed an important exam I had studied months for.",
    "I spent three days alone, avoiding everyone.",
    "I started learning something completely new and practiced every day.",
    "I volunteered all weekend helping people who needed it.",
    "I stood up to someone and said exactly what I thought.",
  ];

  const fmt = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);
  const pad3 = (n) => String(n).padStart(3, "0");
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  // center-zero geometry for a signed value in [-1, 1]
  const pct = (v) => (clamp(v, -1, 1) + 1) / 2 * 100;
  function centerFill(v) {
    const p = pct(v);
    return v >= 0 ? { left: 50, width: p - 50 } : { left: p, width: 50 - p };
  }
  // The push is small per experience; show it on a [-0.5, 0.5] visual scale so it
  // reads, while the exact number stays in the label.
  const PUSH_VIS = 2;

  const App = {
    config: null,
    snap: null,
    busy: false,
    orb: null,
    pushBars: {},   // key -> { root, fill, val }
    traitBars: {},  // key -> { root, fill, ghost, val, glyph }
    reflTimer: null,
  };

  // ---- theme ----------------------------------------------------------------
  function currentTheme() {
    return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
  }
  function applyThemeGlyph() {
    const g = currentTheme() === "dark" ? "☾" : "☀";
    [$("start-theme"), $("cockpit-theme")].forEach((b) => { if (b) b.textContent = g; });
  }
  function toggleTheme() {
    const next = currentTheme() === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("theme", next); } catch (e) {}
    applyThemeGlyph();
    if (App.orb) App.orb.syncColors();
  }

  // ---- view switching -------------------------------------------------------
  function showStart() {
    $("cockpit").classList.add("hidden");
    $("start").classList.remove("hidden");
    loadRoster();
  }
  function showCockpit() {
    $("start").classList.add("hidden");
    $("cockpit").classList.remove("hidden");
    if (!App.orb) App.orb = window.createOrb($("orb-canvas"));
    App.orb.resize();
  }

  // ===========================================================================
  // START SCREEN
  // ===========================================================================
  function initTabs() {
    const tabs = $("start-tabs");
    tabs.addEventListener("click", (e) => {
      const btn = e.target.closest(".tab");
      if (!btn) return;
      const name = btn.dataset.tab;
      tabs.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === btn));
      ["existing", "genesis", "manual"].forEach((p) =>
        $("panel-" + p).classList.toggle("active", p === name));
    });
  }

  function miniOcean(traits) {
    const wrap = elc("div", "mini-ocean");
    traits.forEach((t) => {
      const row = elc("div", "mini-row");
      row.appendChild(elc("span", "mini-k", t.key));
      const track = elc("div", "mini-track");
      track.appendChild(elc("span", "mini-zero"));
      const f = centerFill(t.value);
      const fill = elc("span", "mini-fill");
      fill.style.left = f.left + "%"; fill.style.width = f.width + "%";
      track.appendChild(fill);
      row.appendChild(track);
      wrap.appendChild(row);
    });
    return wrap;
  }

  async function loadRoster() {
    const host = $("roster");
    host.innerHTML = "";
    let chars = [];
    try { chars = (await API.characters()).characters || []; }
    catch (e) { host.appendChild(elc("p", "roster-empty", "Could not load characters.")); return; }

    if (!chars.length) {
      host.appendChild(elc("p", "roster-empty",
        "No one here yet — create someone with “From a story” or “Build one.”"));
      return;
    }
    chars.forEach((c) => {
      const card = elc("button", "char-card");
      card.appendChild(elc("h3", null, c.name));
      const facts = ["age", "origin", "culture"]
        .map((k) => c.identity[k]).filter(Boolean).join(" · ");
      card.appendChild(elc("p", "char-facts", facts || " "));
      card.appendChild(elc("p", "char-count",
        c.turn + (c.turn === 1 ? " experience" : " experiences")));
      card.appendChild(miniOcean(c.traits));
      card.addEventListener("click", () => selectCharacter(c.name));
      host.appendChild(card);
    });
  }

  function buildManualForm() {
    const grid = $("identity-grid");
    grid.innerHTML = "";
    App.config.identity_fields.forEach((f) => {
      const field = elc("div", "field");
      const label = elc("label", null, f.label);
      label.setAttribute("for", "id-" + f.key);
      const input = elc("input");
      input.type = "text"; input.id = "id-" + f.key; input.dataset.key = f.key;
      if (f.key === "name") input.placeholder = "required";
      field.appendChild(label); field.appendChild(input);
      grid.appendChild(field);
    });

    const list = $("slider-list");
    list.innerHTML = "";
    App.config.trait_questions.forEach((q) => {
      const row = elc("div", "slider-row");
      const head = elc("div", "slider-head");
      head.appendChild(elc("span", "slider-name", q.name));
      const valEl = elc("span", "slider-val");
      head.appendChild(valEl);
      row.appendChild(head);

      const input = elc("input");
      input.type = "range"; input.min = "1"; input.max = "5"; input.step = "1";
      input.value = "3"; input.dataset.key = q.key;
      row.appendChild(input);

      const poles = elc("div", "slider-poles");
      poles.appendChild(elc("span", null, q.low));
      poles.appendChild(elc("span", null, q.high));
      row.appendChild(poles);

      const describe = () => {
        const lvl = parseInt(input.value, 10);
        const word = lvl === 3 ? "balanced" : lvl < 3 ? q.low : q.high;
        valEl.textContent = lvl + " · " + word;
      };
      input.addEventListener("input", describe);
      describe();
      list.appendChild(row);
    });
  }

  async function doGenesis() {
    const bio = $("bio-input").value.trim();
    const note = $("genesis-note");
    if (!bio) { note.className = "form-note err"; note.textContent = "Write a sentence first."; return; }
    setFormBusy("genesis", true, "Bringing them to life…");
    try {
      const snap = await API.createGenesis(bio);
      const via = snap.created_via === "heuristic"
        ? "seeded heuristically" : "seeded by " + snap.created_via;
      enterWith(snap, `${snap.name} was born — ${via}.`);
    } catch (e) {
      note.className = "form-note err"; note.textContent = e.message || "Something went wrong.";
    } finally { setFormBusy("genesis", false); }
  }

  async function doManual() {
    const identity = {};
    $("identity-grid").querySelectorAll("input").forEach((i) => {
      const v = i.value.trim();
      if (v) identity[i.dataset.key] = v;
    });
    const note = $("manual-note");
    if (!identity.name) { note.className = "form-note err"; note.textContent = "A name is required."; return; }
    const levels = {};
    $("slider-list").querySelectorAll("input[type=range]").forEach((i) => {
      levels[i.dataset.key] = parseInt(i.value, 10);
    });
    setFormBusy("manual", true, "Creating…");
    try {
      const snap = await API.createManual(identity, levels);
      enterWith(snap, `${snap.name} is ready. Tell them what happens next.`);
    } catch (e) {
      note.className = "form-note err"; note.textContent = e.message || "Something went wrong.";
    } finally { setFormBusy("manual", false); }
  }

  function setFormBusy(which, busy, msg) {
    const btn = $(which + "-btn");
    btn.disabled = busy;
    const note = $(which + "-note");
    if (busy) { note.className = "form-note"; note.textContent = msg || ""; }
    else if (note.className === "form-note") note.textContent = "";
  }

  async function selectCharacter(name) {
    try {
      const snap = await API.select(name);
      enterWith(snap, `Continuing as ${snap.name}. Every message shapes who they become.`);
    } catch (e) { alert(e.message || "Could not open that character."); }
  }

  // ===========================================================================
  // COCKPIT
  // ===========================================================================
  function enterWith(snap, systemLine) {
    App.snap = snap;
    showCockpit();
    $("chat-scroll").innerHTML = "";
    buildPushBars(snap);
    buildTraitBars(snap);
    buildSuggestions();
    renderHeader(snap);
    renderMood(snap);
    syncOrb(snap, false);
    if (systemLine) addMessage("system", systemLine);
    const lead = leadIn(snap);
    if (lead) addMessage("agent", lead);
    $("chat-input").focus();
  }

  function leadIn(snap) {
    // A short in-character hello grounded in their dominant trait.
    const d = snap.dominant;
    return `Hi. I don't know quite who I am yet — right now I feel ${d.glyph}. Tell me what happens to me.`;
  }

  function renderHeader(snap) {
    $("hs-name").textContent = snap.name;
    $("hs-turn").textContent = pad3(snap.turn);
    const d = snap.dominant;
    $("hs-dominant-label").textContent = d.name + (d.dir >= 0 ? " ▲" : " ▼");
    $("hs-dominant").title = d.glyph;
    $("chat-name").textContent = snap.name;
    $("chat-tagline").textContent = d.glyph;
  }

  function buildSuggestions() {
    const row = $("suggest-row");
    if (row.childElementCount) return;   // build once
    SUGGESTIONS.forEach((s) => {
      const chip = elc("button", "chip", s);
      chip.addEventListener("click", () => { if (!App.busy) send(s); });
      row.appendChild(chip);
    });
  }

  function buildPushBars(snap) {
    const host = $("push-bars");
    host.innerHTML = ""; App.pushBars = {};
    snap.push.forEach((p) => {
      const root = elc("div", "cbar cbar--belief");
      const top = elc("div", "cbar-top");
      top.appendChild(elc("span", "cbar-name", p.name));
      const val = elc("span", "cbar-val", fmt(p.value));
      top.appendChild(val);
      root.appendChild(top);
      const track = elc("div", "cbar-track");
      track.appendChild(elc("span", "cbar-zero"));
      const fill = elc("span", "cbar-fill");
      track.appendChild(fill);
      root.appendChild(track);
      host.appendChild(root);
      App.pushBars[p.key] = { root, fill, val };
    });
    updatePushBars(snap, false);
    $("push-meta").innerHTML = "";
  }

  function updatePushBars(snap, flash) {
    snap.push.forEach((p) => {
      const b = App.pushBars[p.key];
      if (!b) return;
      const f = centerFill(p.value * PUSH_VIS);
      b.fill.style.left = f.left + "%";
      b.fill.style.width = f.width + "%";
      b.val.textContent = fmt(p.value);
      b.val.className = "cbar-val " + (p.value >= 0 ? "pos" : "neg");
      if (flash && Math.abs(p.value) > 0.001) {
        b.root.classList.remove("is-flash");
        void b.root.offsetWidth;            // restart the animation
        b.root.classList.add("is-flash");
      }
    });
    const src = snap.source ? `via ${snap.source}` : "";
    const seen = snap.seen != null ? ` · seen ${snap.seen} like it before` : "";
    $("push-meta").innerHTML = src ? `<b>${src}</b>${seen}` : "";
  }

  function buildTraitBars(snap) {
    const host = $("trait-bars");
    host.innerHTML = ""; App.traitBars = {};
    snap.traits.forEach((t) => {
      const root = elc("div", "cbar cbar--trait");
      const top = elc("div", "cbar-top");
      const name = elc("span", "cbar-name");
      name.appendChild(document.createTextNode(t.name));
      const glyph = elc("em", null, t.glyph);
      name.appendChild(glyph);
      top.appendChild(name);
      const val = elc("span", "cbar-val", fmt(t.value));
      top.appendChild(val);
      root.appendChild(top);
      const track = elc("div", "cbar-track");
      track.appendChild(elc("span", "cbar-zero"));
      const ghost = elc("span", "cbar-ghost");
      ghost.style.left = pct(t.base) + "%";
      ghost.title = "baseline";
      track.appendChild(ghost);
      const fill = elc("span", "cbar-fill");
      track.appendChild(fill);
      root.appendChild(track);
      host.appendChild(root);
      App.traitBars[t.key] = { root, fill, ghost, val, glyph };
    });
    updateTraitBars(snap, null);
  }

  function updateTraitBars(snap, pulseKey) {
    snap.traits.forEach((t) => {
      const b = App.traitBars[t.key];
      if (!b) return;
      const f = centerFill(t.value);
      b.fill.style.left = f.left + "%";
      b.fill.style.width = f.width + "%";
      b.ghost.style.left = pct(t.base) + "%";
      b.val.textContent = fmt(t.value);
      b.glyph.textContent = t.glyph;
      if (pulseKey && t.key === pulseKey) {
        b.root.classList.remove("is-pulse");
        void b.root.offsetWidth;
        b.root.classList.add("is-pulse");
        const ring = elc("span", "cbar-pulse");
        b.root.querySelector(".cbar-track").appendChild(ring);
        setTimeout(() => ring.remove(), 1400);
      }
    });
  }

  function renderMood(snap) {
    const a = snap.appraisal || { valence: 0, intensity: 0, novelty: 0 };
    const val = centerFill(a.valence || 0);
    const vEl = $("mood-valence");
    vEl.style.left = val.left + "%"; vEl.style.width = val.width + "%";
    $("mood-intensity").style.width = clamp(a.intensity || 0, 0, 1) * 100 + "%";
    $("mood-novelty").style.width = clamp(a.novelty || 0, 0, 1) * 100 + "%";
  }

  function syncOrb(snap, fresh) {
    if (!App.orb) return;
    const byKey = {};
    snap.traits.forEach((t) => { byKey[t.key] = t; });
    const E = byKey.E ? byKey.E.value : 0;
    const N = byKey.N ? byKey.N.value : 0;
    App.orb.setState({
      mood: { energy: (E + 1) / 2, stress: (N + 1) / 2 },
      traits: snap.traits.map((t) => ({ value: (t.value + 1) / 2, base: (t.base + 1) / 2 })),
      beliefs: snap.traits.map((t) => ({
        mean: t.value,
        base: t.base,
        confidence: clamp(0.4 + Math.abs(t.value) * 0.5, 0, 1),
        fresh: fresh,
      })),
    });
  }

  function addMessage(role, text) {
    const scroll = $("chat-scroll");
    const msg = elc("div", "msg msg-" + role);
    if (role !== "system") {
      msg.appendChild(elc("span", "msg-who", role === "user" ? "You" : App.snap.name));
    }
    msg.appendChild(elc("p", "msg-text", text));
    scroll.appendChild(msg);
    scroll.scrollTop = scroll.scrollHeight;
    return msg;
  }

  function showTyping() {
    const scroll = $("chat-scroll");
    const msg = elc("div", "msg msg-agent");
    msg.appendChild(elc("span", "msg-who", App.snap.name));
    const p = elc("p", "msg-text typing");
    p.appendChild(elc("i")); p.appendChild(elc("i")); p.appendChild(elc("i"));
    msg.appendChild(p);
    scroll.appendChild(msg);
    scroll.scrollTop = scroll.scrollHeight;
    return msg;
  }

  function showFormation(formation) {
    const mount = $("refl-mount");
    mount.innerHTML = "";
    if (!formation) return;
    const banner = elc("div", "refl-banner");
    banner.appendChild(elc("span", "refl-k", "Formed"));
    const body = elc("span", "refl-body");
    const b = elc("b", null, formation.name);
    body.appendChild(b);
    body.appendChild(document.createTextNode(
      " " + (formation.delta >= 0 ? "+" : "−") + Math.abs(formation.delta).toFixed(3)));
    body.appendChild(elc("em", null, " — " + formation.note));
    banner.appendChild(body);
    mount.appendChild(banner);
    clearTimeout(App.reflTimer);
    App.reflTimer = setTimeout(() => { mount.innerHTML = ""; }, 5200);
  }

  function setBusy(busy) {
    App.busy = busy;
    $("send-btn").disabled = busy;
    $("send-btn").textContent = busy ? "…" : "Send";
    $("suggest-row").querySelectorAll(".chip").forEach((c) => { c.disabled = busy; });
  }

  async function send(textArg) {
    const input = $("chat-input");
    const text = (textArg != null ? textArg : input.value).trim();
    if (!text || App.busy) return;
    input.value = ""; autoGrow(input);
    setBusy(true);

    addMessage("user", text);
    if (App.orb) App.orb.pulse();           // sparks: the experience arriving
    const typing = showTyping();

    try {
      const snap = await API.turn(App.snap.name, text);
      typing.remove();
      const prev = App.snap;
      App.snap = snap;

      addMessage("agent", snap.reply || "…");
      updatePushBars(snap, true);           // magenta flash: the push just landed
      const moved = snap.formation ? snap.formation.key : biggestMove(prev, snap);
      updateTraitBars(snap, moved);         // violet bars ease to new values
      renderMood(snap);
      renderHeader(snap);
      syncOrb(snap, true);
      showFormation(snap.formation);
    } catch (e) {
      typing.remove();
      addMessage("system", "— " + (e.message || "the engine didn't respond") + " —");
    } finally {
      setBusy(false);
      input.focus();
    }
  }

  function biggestMove(prev, next) {
    if (!prev) return null;
    const before = {}; prev.traits.forEach((t) => { before[t.key] = t.value; });
    let key = null, best = 0;
    next.traits.forEach((t) => {
      const d = Math.abs(t.value - (before[t.key] || 0));
      if (d > best) { best = d; key = t.key; }
    });
    return best > 0.005 ? key : null;
  }

  // ---- composer niceties ----------------------------------------------------
  function autoGrow(ta) {
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 120) + "px";
  }

  // ---- boot -----------------------------------------------------------------
  async function boot() {
    applyThemeGlyph();
    $("start-theme").addEventListener("click", toggleTheme);
    $("cockpit-theme").addEventListener("click", toggleTheme);
    $("switch-btn").addEventListener("click", showStart);
    $("cockpit-brand").addEventListener("click", showStart);
    $("genesis-btn").addEventListener("click", doGenesis);
    $("manual-btn").addEventListener("click", doManual);
    $("send-btn").addEventListener("click", () => send());

    const input = $("chat-input");
    input.addEventListener("input", () => autoGrow(input));
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
    });
    $("bio-input").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); doGenesis(); }
    });

    initTabs();
    try {
      App.config = await API.config();
    } catch (e) {
      App.config = { identity_fields: [], trait_questions: [] };
    }
    buildManualForm();
    loadRoster();
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
