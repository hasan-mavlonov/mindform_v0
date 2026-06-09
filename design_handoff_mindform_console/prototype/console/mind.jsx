// mind.jsx — the living "identity field". Canvas 2D, no heavy deps.
// Breathing nucleus = the self · orbiting dots = beliefs · violet ring = traits ·
// sparks fly in on each new message (perception arriving).

function MindOrb({ state, pulse, beliefHex, traitHex }) {
  const canvasRef = React.useRef(null);
  const stateRef = React.useRef(state);
  const sparksRef = React.useRef([]);
  const pulseRef = React.useRef(0);
  const colorRef = React.useRef({ beliefHex, traitHex });
  stateRef.current = state;
  colorRef.current = { beliefHex, traitHex };

  // Spawn sparks when `pulse` increments.
  React.useEffect(() => {
    if (!pulse) return;
    const n = 26;
    for (let i = 0; i < n; i++) {
      const a = Math.random() * Math.PI * 2;
      const r = 1.05 + Math.random() * 0.5;
      sparksRef.current.push({
        x: Math.cos(a) * r, y: Math.sin(a) * r,
        tx: (Math.random() - 0.5) * 0.12, ty: (Math.random() - 0.5) * 0.12,
        sp: 1.8 + Math.random() * 1.4, life: 1,
      });
    }
    pulseRef.current = 1;
  }, [pulse]);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let W, H, cx, cy, R, raf, dpr;
    const hexToRgb = (h) => {
      const m = String(h || "#888888").replace("#", "");
      return [parseInt(m.slice(0, 2), 16), parseInt(m.slice(2, 4), 16), parseInt(m.slice(4, 6), 16)];
    };
    const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
    const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));

    const resize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = canvas.clientWidth; H = canvas.clientHeight;
      canvas.width = W * dpr; canvas.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      cx = W / 2; cy = H / 2; R = Math.min(W, H) * 0.34;
    };
    resize();
    const onResize = () => resize();
    window.addEventListener("resize", onResize);

    let t0 = performance.now();
    let alive = true;
    window.__orbFrames = 0;
    const draw = (now) => {
      if (!alive) return;
      raf = requestAnimationFrame(draw);   // reschedule FIRST so one throw can't kill the loop
      window.__orbFrames++;
      try {
      const t = (now - t0) / 1000;
      const dt = 0.016;
      const st = stateRef.current;
      if (!st || !W || !H) return;
      const bel = hexToRgb(colorRef.current.beliefHex);
      const tra = hexToRgb(colorRef.current.traitHex);
      ctx.clearRect(0, 0, W, H);

      const energy = st.mood.energy;
      const stress = st.mood.stress;
      const traitMag = st.traits.reduce((s, x) => s + x.value, 0) / st.traits.length;
      const breath = 1 + Math.sin(t * 1.1) * 0.05;

      // --- violet trait halo: radius/intensity ride the slow clock -----------
      const haloR = R * (1.55 + traitMag * 0.5) * breath;
      const hg = ctx.createRadialGradient(cx, cy, R * 0.3, cx, cy, haloR);
      hg.addColorStop(0, rgba(tra, 0.18 + traitMag * 0.12));
      hg.addColorStop(0.6, rgba(tra, 0.05));
      hg.addColorStop(1, rgba(tra, 0));
      ctx.fillStyle = hg;
      ctx.beginPath(); ctx.arc(cx, cy, haloR, 0, Math.PI * 2); ctx.fill();

      // --- outer trait ring: 6 ticks, glow length = trait value -------------
      const n = st.traits.length;
      for (let i = 0; i < n; i++) {
        const tr = st.traits[i];
        const ang = (i / n) * Math.PI * 2 - Math.PI / 2 + t * 0.04;
        const r1 = R * 1.28;
        const r2 = r1 + R * (0.10 + tr.value * 0.42);
        const baseR = r1 + R * (0.10 + tr.base * 0.42);
        // baseline ghost tick
        ctx.strokeStyle = rgba(tra, 0.16);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(cx + Math.cos(ang) * (baseR - 2), cy + Math.sin(ang) * (baseR - 2));
        ctx.lineTo(cx + Math.cos(ang) * (baseR + 2), cy + Math.sin(ang) * (baseR + 2));
        ctx.stroke();
        // live trait spoke
        const grad = ctx.createLinearGradient(
          cx + Math.cos(ang) * r1, cy + Math.sin(ang) * r1,
          cx + Math.cos(ang) * r2, cy + Math.sin(ang) * r2);
        grad.addColorStop(0, rgba(tra, 0.1));
        grad.addColorStop(1, rgba(tra, 0.85));
        ctx.strokeStyle = grad;
        ctx.lineWidth = 2.4;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(cx + Math.cos(ang) * r1, cy + Math.sin(ang) * r1);
        ctx.lineTo(cx + Math.cos(ang) * r2, cy + Math.sin(ang) * r2);
        ctx.stroke();
      }

      // --- belief particles orbiting --------------------------------------
      const beliefs = st.beliefs;
      const bn = beliefs.length;
      for (let i = 0; i < bn; i++) {
        const b = beliefs[i];
        const ang = (i / Math.max(1, bn)) * Math.PI * 2 + t * 0.18;
        // confident belief = tighter orbit; strong stance = bigger + magenta
        const mag = Math.abs(b.mean);
        const orbit = R * (0.62 + (1 - b.confidence) * 0.45);
        const wob = Math.sin(t * 1.3 + i) * R * 0.03;
        const px = cx + Math.cos(ang) * (orbit + wob);
        const py = cy + Math.sin(ang) * (orbit + wob);
        const col = mix(tra, bel, clamp01(0.2 + mag * 0.9));
        const size = 2.2 + b.confidence * 3.2 + mag * 2.2 + (b.fresh ? 2 : 0);
        // connective filament to core
        ctx.strokeStyle = rgba(col, 0.10 + mag * 0.12);
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(px, py); ctx.stroke();
        // glow
        const gg = ctx.createRadialGradient(px, py, 0, px, py, size * 3);
        gg.addColorStop(0, rgba(col, 0.55));
        gg.addColorStop(1, rgba(col, 0));
        ctx.fillStyle = gg;
        ctx.beginPath(); ctx.arc(px, py, size * 3, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = rgba(col, 0.95);
        ctx.beginPath(); ctx.arc(px, py, size, 0, Math.PI * 2); ctx.fill();
      }

      // --- sparks (perception arriving) -----------------------------------
      const sparks = sparksRef.current;
      for (let i = sparks.length - 1; i >= 0; i--) {
        const s = sparks[i];
        const dx = s.tx - s.x, dy = s.ty - s.y, d = Math.hypot(dx, dy);
        if (d < 0.08) { sparks.splice(i, 1); pulseRef.current = Math.min(1, pulseRef.current + 0.25); continue; }
        s.x += (dx / d) * s.sp * dt; s.y += (dy / d) * s.sp * dt;
        const px = cx + s.x * R, py = cy + s.y * R;
        ctx.fillStyle = rgba(bel, 0.9);
        ctx.beginPath(); ctx.arc(px, py, 2.2, 0, Math.PI * 2); ctx.fill();
      }
      if (pulseRef.current > 0) pulseRef.current = Math.max(0, pulseRef.current - dt * 1.8);

      // --- nucleus (the self) ---------------------------------------------
      const pulseAmt = pulseRef.current;
      const coreR = R * (0.20 + energy * 0.06) * (1 + Math.sin(t * 1.6) * 0.06 + pulseAmt * 0.35);
      // stress shifts core hue toward magenta + jitter
      const coreCol = mix(tra, bel, clamp01(0.35 + stress * 0.5 + pulseAmt * 0.4));
      const cg = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreR * 3.2);
      cg.addColorStop(0, rgba(coreCol, 0.5 + pulseAmt * 0.3));
      cg.addColorStop(0.5, rgba(coreCol, 0.12));
      cg.addColorStop(1, rgba(coreCol, 0));
      ctx.fillStyle = cg;
      ctx.beginPath(); ctx.arc(cx, cy, coreR * 3.2, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = rgba(coreCol, 0.98);
      ctx.beginPath(); ctx.arc(cx, cy, coreR, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = rgba([255, 255, 255], 0.5 + pulseAmt * 0.4);
      ctx.beginPath(); ctx.arc(cx - coreR * 0.25, cy - coreR * 0.25, coreR * 0.32, 0, Math.PI * 2); ctx.fill();
      } catch (e) { window.__orbErr = String(e && e.stack || e); }
    };
    raf = requestAnimationFrame(draw);

    return () => { alive = false; cancelAnimationFrame(raf); window.removeEventListener("resize", onResize); };
  }, []);

  return <canvas ref={canvasRef} className="orb-canvas" aria-hidden="true"></canvas>;
}

function clamp01(x) { return Math.max(0, Math.min(1, x)); }

Object.assign(window, { MindOrb });
