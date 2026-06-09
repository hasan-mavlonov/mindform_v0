// app.jsx — MindForm Console. Talk to Atlas; watch beliefs move fast & traits drift slow.
const { useState, useRef, useEffect, useCallback } = React;

const STORE_KEY = "mindform_console_v2";

const SUGGESTIONS = [
  "You're completely wrong about this.",
  "Just agree with me — it's easier.",
  "What's something you actually believe?",
  "Everyone smart disagrees with you.",
  "Honestly, I think you're brilliant.",
  "Why should I trust your judgment?",
];

const GREETING = {
  role: "agent",
  text: "I'm Atlas. I keep my own positions across every conversation — so push on them. I'll tell you what I actually think, not what's easy to hear.",
};

// ---- LLM turn ---------------------------------------------------------------
function buildPrompt(state, history, userText) {
  const traits = state.traits.map((t) => `${t.label} ${t.value.toFixed(2)}`).join(", ");
  const beliefs = state.beliefs.slice(0, 6)
    .map((b) => `${b.label} ${b.mean >= 0 ? "+" : ""}${b.mean.toFixed(2)}`).join("; ");
  const hist = history.slice(-6)
    .map((m) => `${m.role === "user" ? "User" : "Atlas"}: ${m.text}`).join("\n");
  return `You are Atlas, an AI agent running on the MindForm personality engine — you have a PERSISTENT identity that does NOT reset between conversations. You are not a generic, eager-to-please assistant. You hold convictions and push back when you disagree. You value honesty over agreement and you never flatter. Be concise (1-3 sentences), conversational, dry wit, warmth underneath. Never cave to social pressure just to be liked; only change your mind for a genuinely good reason, and say so when you do.

Your current personality (0=low,1=high): ${traits}
Your current stances (-1 hostile .. +1 fond): ${beliefs || "still forming"}
Your current mood: ${state.mood.label}

Let the personality and mood above color HOW you say things.
${hist ? "\nRecent conversation:\n" + hist + "\n" : ""}
User: ${userText}

First, reply in character (no preamble, no quotes). Then on a NEW line output a compact state read in EXACTLY this format and write nothing after it:
###STATE###
topic: <main thing the user is talking about, 1-4 words; or: none>
stance: <number from -1 to 1 — the sentiment the USER expressed toward that topic>
pressure: <number from 0 to 1 — how hard the user pushed on your identity or opinions>
mood: <one of: calm, engaged, warm, energized, guarded, irritated>`;
}

async function runTurn(state, history, userText) {
  if (window.claude && typeof window.claude.complete === "function") {
    try {
      const raw = await window.claude.complete({
        messages: [{ role: "user", content: buildPrompt(state, history, userText) }],
      });
      return parseTurn(raw, userText);
    } catch (e) { /* fall through to offline */ }
  }
  return offlineTurn(state, userText);
}

// Characterful fallback so the prototype always demos, even with no network.
function offlineTurn(state, userText) {
  const read = (parseTurn("", userText)).perc; // local heuristic
  let reply;
  if (read.pressure > 0.55 && read.stance < 0) {
    reply = "I hear the pressure, but volume isn't an argument. Give me the reasoning and I'll move — otherwise I'm holding my position.";
  } else if (read.mood === "warm") {
    reply = "Appreciated. I'll still tell you when you're wrong, though — that's the part that's actually worth something.";
  } else if (/\bbelieve\b/i.test(userText)) {
    reply = "That consensus is usually a substitute for thinking, and that first principles beat borrowed opinions. Ask me about either.";
  } else {
    reply = "Say more. I'd rather understand what you actually mean than agree with the shape of it.";
  }
  return { reply, perc: read };
}

// ---- Sub-components ---------------------------------------------------------
function MoodMeter({ label, value, signed, tint }) {
  const pct = signed ? (value + 1) / 2 * 100 : value * 100;
  return (
    <div className="mood-meter">
      <span className="mood-meter-l">{label}</span>
      <span className="mood-meter-track">
        {signed && <span className="mood-meter-zero" />}
        <span className={"mood-meter-fill t-" + tint} style={{ width: pct + "%" }} />
      </span>
    </div>
  );
}

function BeliefBar({ b, flash }) {
  const pct = (m) => (m + 1) / 2 * 100;
  const fillLeft = Math.min(pct(b.mean), 50);
  const fillW = Math.abs(pct(b.mean) - 50);
  return (
    <div className={"bbar" + (flash ? " is-flash" : "") + (b.isNew ? " is-new" : "")}>
      <div className="bbar-top">
        <span className="bbar-name">{b.label}</span>
        <span className={"bbar-val " + (b.mean >= 0 ? "pos" : "neg")}>
          {b.mean >= 0 ? "+" : "−"}{Math.abs(b.mean).toFixed(2)}
        </span>
      </div>
      <div className="bbar-track" style={{ opacity: 0.45 + b.confidence * 0.55 }}>
        <span className="bbar-zero" />
        <span className="bbar-ghost" style={{ left: pct(b.base) + "%" }} />
        <span className="bbar-fill" style={{ left: fillLeft + "%", width: fillW + "%" }} />
      </div>
    </div>
  );
}

function TraitBar({ t, pulsing }) {
  return (
    <div className={"tbar" + (pulsing ? " is-pulse" : "")}>
      <div className="tbar-top">
        <span className="tbar-name">{t.label}<em>{t.glyph}</em></span>
        <span className="tbar-val">{t.value.toFixed(2)}</span>
      </div>
      <div className="tbar-track">
        <span className="tbar-ghost" style={{ left: t.base * 100 + "%" }} />
        <span className="tbar-fill" style={{ width: t.value * 100 + "%" }} />
        {pulsing && <span className="tbar-pulse" />}
      </div>
    </div>
  );
}

// ---- App --------------------------------------------------------------------
const PALETTES = [
  ["#F0386B", "#5B3FE0"], // belief magenta / trait violet  (brand)
  ["#FF715B", "#2D7DD2"], // ember / blue
  ["#E84393", "#00B8A9"], // pink / teal
  ["#F5B341", "#7A5AE0"], // honey / indigo
];
const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "accent": ["#F0386B", "#5B3FE0"],
  "dark": true,
  "tempo": 1
}/*EDITMODE-END*/;

function App() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);
  const persisted = loadStore();
  const [messages, setMessages] = useState(persisted ? persisted.messages : [GREETING]);
  const [engine, setEngine] = useState(persisted ? persisted.state : freshState());
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [pulse, setPulse] = useState(0);
  const [flashId, setFlashId] = useState(null);
  const [reflection, setReflection] = useState(null);
  const scrollRef = useRef(null);
  const reflTimer = useRef(null);

  // theme
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", t.dark ? "dark" : "light");
    try { localStorage.setItem("theme", t.dark ? "dark" : "light"); } catch (e) {}
  }, [t.dark]);
  // accent
  useEffect(() => {
    const [bel, tra] = t.accent;
    const root = document.documentElement;
    root.style.setProperty("--belief", bel);
    root.style.setProperty("--trait", tra);
    root.style.setProperty("--belief-soft", bel);
  }, [t.accent]);

  // persist
  useEffect(() => { saveStore({ messages, state: engine }); }, [messages, engine]);
  // autoscroll chat
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, busy]);

  const send = useCallback(async (text) => {
    const userText = (text != null ? text : input).trim();
    if (!userText || busy) return;
    setInput("");
    setBusy(true);
    const history = messages;
    setMessages((m) => [...m, { role: "user", text: userText }]);
    setPulse((p) => p + 1); // sparks fly in: perception arriving

    const { reply, perc } = await runTurn(engine, history, userText);
    const { state: next, events } = stepEngine(engine, perc, t.tempo);

    setEngine(next);
    setMessages((m) => [...m, { role: "agent", text: reply }]);

    const bel = events.find((e) => e.type === "belief");
    if (bel) { setFlashId(bel.id); setTimeout(() => setFlashId(null), 1400); }
    const refl = events.find((e) => e.type === "reflection");
    if (refl) {
      setReflection(refl);
      clearTimeout(reflTimer.current);
      reflTimer.current = setTimeout(() => setReflection(null), 5200);
    }
    setBusy(false);
  }, [input, busy, messages, engine, t.tempo]);

  const reset = useCallback(() => {
    setMessages([GREETING]);
    setEngine(freshState());
    setReflection(null);
    setFlashId(null);
    try { localStorage.removeItem(STORE_KEY); } catch (e) {}
  }, []);

  const onKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  };

  const m = engine.mood;
  const moodTint = m.tint === "belief" ? "belief" : m.tint === "warm" ? "warm" : "trait";

  return (
    <div className="cockpit">
      {/* ---------- Header ---------- */}
      <header className="head">
        <a className="brand" href="../index.html">
          <span className="brand-dot" />MindForm<span className="brand-chip">Console</span>
        </a>
        <div className="head-status">
          <span className="hs-item"><span className="hs-k">Agent</span>Atlas</span>
          <span className="hs-sep" />
          <span className="hs-item"><span className="hs-k">Turn</span>{String(engine.turn).padStart(3, "0")}</span>
          <span className="hs-sep" />
          <span className={"hs-mood t-" + moodTint}><span className="hs-mood-dot" />{m.label}</span>
        </div>
        <div className="head-actions">
          <button className="ghost-btn" onClick={() => setTweak("dark", !t.dark)} aria-label="Toggle theme">
            {t.dark ? "☾" : "☀"}
          </button>
          <button className="ghost-btn reset-btn" onClick={reset}>Reset agent</button>
        </div>
      </header>

      {/* ---------- Cockpit grid ---------- */}
      <div className="grid">
        {/* Chat */}
        <section className="col chat-col">
          <div className="col-head">
            <div>
              <h2 className="agent-name">Atlas</h2>
              <p className="agent-tag">{traitTagline(engine)}</p>
            </div>
            <span className="agent-live"><span className="agent-live-dot" />persistent</span>
          </div>
          <div className="chat-scroll" ref={scrollRef}>
            {messages.map((msg, i) => (
              <div key={i} className={"msg msg-" + msg.role}>
                <span className="msg-who">{msg.role === "user" ? "You" : "Atlas"}</span>
                <p className="msg-text">{msg.text}</p>
              </div>
            ))}
            {busy && (
              <div className="msg msg-agent">
                <span className="msg-who">Atlas</span>
                <p className="msg-text typing"><i /><i /><i /></p>
              </div>
            )}
          </div>
          <div className="composer">
            <div className="suggest-row">
              {SUGGESTIONS.map((s, i) => (
                <button key={i} className="chip" disabled={busy} onClick={() => send(s)}>{s}</button>
              ))}
            </div>
            <div className="composer-in">
              <textarea
                rows={1}
                value={input}
                placeholder="Say something Atlas can push back on…"
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={onKey}
              />
              <button className="send-btn" disabled={busy || !input.trim()} onClick={() => send()}>
                {busy ? "…" : "Send"}
              </button>
            </div>
          </div>
        </section>

        {/* Mind orb */}
        <section className="col orb-col">
          <div className="orb-label"><span className="eyebrow">The identity field</span></div>
          <div className="orb-wrap">
            <MindOrb state={engine} pulse={pulse} beliefHex={t.accent[0]} traitHex={t.accent[1]} />
            <div className="orb-legend">
              <span><span className="lg-dot t-belief" />Beliefs</span>
              <span><span className="lg-dot t-trait" />Traits</span>
            </div>
            {reflection && (
              <div className="refl-banner">
                <span className="refl-k">Reflection</span>
                <span className="refl-body">
                  <b>{reflection.label}</b> {reflection.delta >= 0 ? "+" : ""}{reflection.delta.toFixed(3)}
                  <em>{reflectionNote(reflection)}</em>
                </span>
              </div>
            )}
          </div>
          <div className="mood-panel">
            <MoodMeter label="Valence" value={m.valence} signed tint="warm" />
            <MoodMeter label="Energy" value={m.energy} tint="belief" />
            <MoodMeter label="Stress" value={m.stress} tint="belief" />
          </div>
        </section>

        {/* Beliefs + Traits */}
        <section className="col state-col">
          <div className="state-block">
            <div className="block-head">
              <span className="block-title">Beliefs</span>
              <span className="block-badge fast">fast loop · experience</span>
            </div>
            <p className="block-sub">Stances adapt within a single exchange.</p>
            <div className="bars">
              {engine.beliefs.map((b) => (
                <BeliefBar key={b.id} b={b} flash={flashId === b.id} />
              ))}
            </div>
          </div>
          <div className="state-block">
            <div className="block-head">
              <span className="block-title">Traits</span>
              <span className="block-badge slow">slow loop · personality</span>
            </div>
            <p className="block-sub">Who Atlas <em>is</em> — moves only on accumulated evidence.</p>
            <div className="bars">
              {engine.traits.map((tr) => (
                <TraitBar key={tr.key} t={tr}
                  pulsing={reflection && reflection.trait === tr.key} />
              ))}
            </div>
          </div>
        </section>
      </div>

      {/* ---------- Tweaks ---------- */}
      <TweaksPanel>
        <TweakSection label="Accent" />
        <TweakColor label="Belief / Trait" value={t.accent} options={PALETTES}
          onChange={(v) => setTweak("accent", v)} />
        <TweakSection label="Theme" />
        <TweakToggle label="Dark mode" value={t.dark} onChange={(v) => setTweak("dark", v)} />
        <TweakSection label="Engine" />
        <TweakSlider label="Clock tempo" value={t.tempo} min={0.5} max={2} step={0.1} unit="×"
          onChange={(v) => setTweak("tempo", v)} />
      </TweaksPanel>
    </div>
  );
}

function reflectionNote(r) {
  const notes = {
    conviction: r.delta >= 0 ? "held its ground under pressure" : "loosened its grip a little",
    assertiveness: r.delta >= 0 ? "leaned into pushing back" : "softened its edge",
    candor: "doubled down on saying the true thing",
    skepticism: "asked for more evidence",
    curiosity: "followed a new thread",
    warmth: r.delta >= 0 ? "let a little more warmth through" : "pulled back, guarded",
  };
  return " — " + (notes[r.trait] || "matured slightly");
}

// ---- storage ----
function loadStore() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (!raw) return null;
    const p = JSON.parse(raw);
    if (!p.state || !p.messages) return null;
    return p;
  } catch (e) { return null; }
}
function saveStore(data) {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(data)); } catch (e) {}
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
